# 26 — Mixture of Experts from First Principles

> Video: 16:49:58 — Mixture of Experts from First Principles
> Code: `torchfeather/model/moe/moe.py`
> Papers: GShard (2006.16668), Fine-Grained MoE Scaling (2402.07871),
> Auxiliary-Loss-Free Load Balancing (2408.15664), DeepSeek-V3 (2412.19437)

## Why this exists

Every parallelism in Part III attacked the *systems* side: the same model, spread
over more devices. MoE attacks the model itself, and it starts from an
observation about topic 02's two formulas.

```
compute:   C ≈ 6 · N_active · D          FLOPs scale with ACTIVE parameters
quality:   L ≈ f(N_total, D)             loss improves with TOTAL parameters
```

In a dense model these are the same `N`, so quality and cost are welded together.
**MoE breaks the weld:** make the FFN sparse so that each token touches only `k`
of `E` experts. Total parameters grow `E/k ×` while FLOPs per token stay fixed.

Our reference config gets a modest `2.88B / 1.31B = 2.2×`. DeepSeek-V3 gets
`671B / 37B = 18×` — a 671B-parameter model that costs 37B to run.

The price is paid entirely in **systems complexity**: a data-dependent routing
decision, a load-balancing problem that is simultaneously a modelling and a
scheduling problem, and a collective (all-to-all) we have not needed until now.
Topics 27–30 are that price.

## 1. Conditional computation

Replace the dense FFN with `E` parallel FFNs and a router:

```
dense:  y = FFN(x)

MoE:    s   = softmax_or_sigmoid( x W_gate )        ∈ R^E     the router
        T   = top_k(s)                              indices of k experts
        y   = Σ_{i∈T} g_i · E_i(x)   [+ E_shared(x)]
```

Each `E_i` is an ordinary SwiGLU FFN with its own weights. Parameter and FLOP
accounting (topic 02 §2.4):

```
total  = d·E              (router)
       + 3·d·d_ff^E·E_s   (shared)
       + 3·d·d_ff^E·E     (routed)

active = d·E + 3·d·d_ff^E·E_s + 3·d·d_ff^E·k
```

So the sparsity ratio is `≈ E/k` once the routed experts dominate. That is the
whole economic argument, in one ratio.

### 1.1 Token choice vs expert choice

Two ways to construct the assignment:

- **Token choice** (used here, and by almost everything): each *token* picks its
  top `k` experts. Every token is served, but an expert may be picked by any
  number of tokens → **load imbalance**.
- **Expert choice**: each *expert* picks its top `C` tokens. Perfect balance by
  construction, but some tokens may be picked by no expert at all (and, in a
  causal LM, expert choice leaks information across positions unless handled
  carefully).

Token choice moves the difficulty into load balancing, which §4 solves. That
turns out to be the better trade, because balance can be *encouraged* while
dropped tokens cannot be undone.

## 2. The router

```python
with torch.autocast(device_type=x.device.type, dtype=torch.float32):
    scores = self.gate(x)                        # [T, E]

if self.score_func == "sigmoid":
    scores = torch.sigmoid(scores)
elif self.score_func == "softmax":
    scores = F.softmax(scores, dim=1)
```

Three deliberate choices.

**fp32, forced.** The `autocast(dtype=torch.float32)` overrides the ambient bf16.
The router is a `d × E` matmul — negligible FLOPs — but its output decides a
*discrete* selection. Two experts whose scores differ by less than bf16
resolution would be ordered arbitrarily, and worse, *inconsistently* between the
forward pass and an activation-checkpointed recomputation. That produces
genuinely wrong gradients (the recomputed forward routes differently from the
one whose output was used). Cheap insurance against a nasty bug.

**Sigmoid rather than softmax** (the default here, and DeepSeek-V3's choice).
Softmax couples the experts: raising one score lowers all others, so the gates
must sum to 1 and the experts compete. Sigmoid scores each expert independently,
so a token can be "strongly relevant to three experts" — which is the natural
statement — and the router's gradient for expert `i` does not depend on expert
`j`'s score. With `route_norm` you can still normalize afterwards if you want
convex combination weights.

**`top_k` with `sorted=False`.** We only need the *set*, and the subsequent sort
is by expert id, not by score. Skipping the sort is free.

## 3. The gradient problem

`top_k` is an `argmax`: piecewise constant, derivative zero almost everywhere.
So (topic 11 §6.2):

```
∂y/∂s   flows ONLY through g_i for the SELECTED experts.
        An unselected expert receives EXACTLY ZERO gradient.
```

Which gives a positive feedback loop:

```
expert i is rarely selected
   → receives little gradient
   → improves slowly
   → becomes less useful than its rivals
   → is selected even less
   → ...
```

**Expert collapse.** Left alone, a handful of experts absorb nearly all traffic
and the rest are dead weight — you pay `E×` the memory for `~k` effective
experts. This is not a numerical instability that better initialization fixes; it
is a structural property of the derivative of `argmax`. It must be corrected
outside the gradient.

## 4. Load balancing

### 4.1 The classical answer: an auxiliary loss

GShard/Switch add a term to the objective:

```
L_aux = α · E · Σ_{i=1}^{E} f_i · P_i

f_i = fraction of tokens routed to expert i      (a count — not differentiable)
P_i = mean router probability for expert i        (differentiable)
```

Minimized when both are uniform (`1/E`). The `f_i · P_i` product is the trick:
`f_i` acts as a non-differentiable weight on the differentiable `P_i`, so the
gradient pushes down the probability of experts that are *currently* overloaded.

It works, and it has a real cost: **it competes with the language-modelling
objective.** The optimizer is trading perplexity for balance, and tuning `α`
means trading them explicitly.

### 4.2 The better answer: auxiliary-loss-free balancing

DeepSeek-V3's approach (2408.15664), and what this codebase implements. The idea:
balance the router with a **bias applied to selection only**, updated by a rule
rather than by gradient descent.

```python
if expert_bias is not None:
    # NOTE: The expert_bias is only used for routing.
    _, selected_experts_indices = torch.topk(
        scores + expert_bias, k=self.top_k, dim=1, sorted=False)
    # The gating value top_scores is still derived from the original scores.
    top_scores = scores.gather(dim=1, index=selected_experts_indices)
```

**This is the crux of the technique, and it is two lines.** Read them carefully:

```
selection:  topk(scores + expert_bias)     ← the bias steers WHICH experts
gating:     scores.gather(...)             ← the ORIGINAL score sets HOW MUCH
```

The bias changes the *assignment* but never enters the *value* that multiplies
the expert's output. Therefore:

- the gradient flowing to the router is computed from the true scores,
- no term is added to the loss,
- the language-modelling objective is completely undisturbed.

Balance becomes a **routing-time scheduling decision**, not an optimization
objective. That is why it is strictly better than an auxiliary loss rather than
merely different.

### 4.3 The update rule

From topic 24 §1.3, in a post-optimizer-step hook:

```python
expert_bias_delta = moe.load_balance_coeff * torch.sign(
    tokens_per_expert.mean() - tokens_per_expert)
expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
moe.expert_bias.add_(expert_bias_delta)
moe.tokens_per_expert.zero_()
```

- **`sign`, not magnitude** → a fixed step size, stable under any degree of
  imbalance.
- **zero-mean projection** → only *relative* bias affects a top-k, so the common
  component is unobservable; removing it prevents drift.
- **counts all-reduced first** → balance must be global, not per-rank.

### 4.4 Counting, and the two buffers

```python
self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32),
                     persistent=True)     # ← CHECKPOINTED
self.register_buffer("tokens_per_expert", torch.zeros(num_experts, dtype=torch.float32),
                     persistent=False)    # ← NOT checkpointed
```

The asymmetry is exactly right and worth internalizing:

- **`expert_bias` is `persistent=True`** — it is *learned* state, accumulated
  over the whole run. Losing it resets the router's balance (topic 24 §4.5).
- **`tokens_per_expert` is `persistent=False`** — it is a per-step accumulator,
  zeroed after every update. Saving it would be meaningless.

Both are fp32 regardless of the model dtype: they are counters and a control
signal, and bf16 accumulation of a count over thousands of tokens loses
integers.

And a lovely note about activation checkpointing:

```python
# NOTE: Activation Checkpointing has the side effect of double counting
# tokens_per_expert: first in the forward pass, and then in the backward pass.
# However, this has no effect on the expert bias update thanks to the
# torch.sign() operator. This count is halved in optimiser.py
with torch.no_grad():
    self.tokens_per_expert.add_(num_tokens_per_expert)
```

AC re-runs the forward during backward (topic 20 §10), so the counter increments
twice. **Because the update uses `sign`, scaling all counts by 2 changes
nothing** — `sign(2m − 2c) = sign(m − c)`. The halving in `optimizer.py` is for
the *logged* value's sake, not for correctness. A design that is accidentally
robust to an unrelated feature, and the comment explains why rather than just
patching it.

## 5. Shared experts

```python
self.shared_experts = (
    FeedForward(dim=dim, hidden_dim=hidden_dim * moe_args.num_shared_experts)
    if moe_args.num_shared_experts > 0 else None)
```

An always-on FFN that every token passes through, in addition to its `k` routed
experts. Two reasons, one modelling and one systems.

**Modelling.** Without it, every routed expert must independently learn the
common, universally-useful transformations — a massive duplication of capacity
across `E` experts. Factoring that shared knowledge into one always-on expert
frees the routed experts to genuinely specialize. It also means the model
degrades gracefully when routing is poor: there is always *one* good path.

**Systems.** This is a nice one:

```python
# NOTE: we execute the shared expert before scoring the output of the routed
# expert to "implicitly" overlap the shared expert compute with token combine
# communication
if self.shared_experts is not None:
    out = self.shared_experts(x)
```

The shared expert's input is `x` — available immediately, needing no routing and
no communication. So it is placed *between* the routed experts' compute and the
combine step, giving the all-to-all (topic 29) something real to overlap with.
Not a loop or a stream annotation: just **ordering the code so an independent
computation sits where the communication is in flight**. Under expert
parallelism this is worth a measurable fraction of the step.

## 6. Granularity

Fine-Grained MoE Scaling Laws (2402.07871) asks: at fixed active parameters and
fixed budget, is it better to have few large experts or many small ones?

```
granularity G  =  d_ff^dense / d_ff^E
```

The finding: **higher granularity is strictly better**, until routing overhead
dominates. Splitting the FFN into more, smaller experts and raising `k` to
compensate gives the router a much larger combinatorial space of expert
*combinations* (`C(E,k)` instead of `E`) at the same FLOP cost.

The trajectory in practice:

```
Switch (2021)      E=128, k=1        coarse
Mixtral (2023)     E=8,   k=2        coarse
DeepSeek-V2/V3     E=160/256, k=6/8, + 1-2 shared experts   fine-grained
```

The reference config's `E=8, k=1` is a small teaching configuration. The costs
of high granularity are all in the systems layer: more experts means a bigger
all-to-all with smaller messages (topic 29), and smaller `d_ff^E` means less
efficient matmuls per expert — which is exactly why grouped matmul (§7) exists.

## 7. The permutation machinery

The interesting implementation problem. After routing you have `T` tokens each
assigned to `k` experts, in arbitrary order. Running each expert on a
`for`-loop-selected subset would mean `E` tiny matmuls of unpredictable size —
launch-bound and terrible.

Instead: **sort the tokens by expert, do one grouped matmul, scatter back.**

### 7.1 Sort

```python
num_tokens_per_expert = torch.histc(selected_experts_indices.view(-1),
                                    bins=self.num_experts, min=0, max=self.num_experts)
token_indices_experts_sorted = torch.argsort(selected_experts_indices.view(-1),
                                             stable=True)
top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
token_indices_experts_sorted = token_indices_experts_sorted // self.top_k
```

Worked through with the source's own example (`T=8, E=12, k=4`), verified:

```
selected (flat):  [0,1,4,7, 1,2,5,8, 2,3,6,9, 3,4,7,10, 4,5,8,11, 5,6,9,0, 6,7,10,1, 7,8,11,2]
                   └─tok 0─┘ └─tok 1─┘ ...

histc:            [2, 3, 3, 2, 3, 3, 3, 4, 3, 2, 2, 2]      tokens per expert

argsort(stable):  [0,23, 1,4,27, 5,8,31, 9,12, 2,13,16, ...]
                   └e0┘  └─e1──┘ └─e2──┘ └e3─┘ └──e4───┘     grouped by expert!

// top_k:         [0, 5, 0,1,6, 1,2,7, 2,3, 0,3,4, ...]      → original token ids
```

Two pieces of index arithmetic to be sure of:

- **`argsort(..., stable=True)`** groups the flat `(token, slot)` positions by
  expert id. `stable=True` matters: it makes the order within an expert
  deterministic (by token index), so the forward pass and any recomputation agree
  — the same concern as the fp32 router in §2.
- **`// self.top_k`** converts a flat position back to a token index, because
  position `t·k + j` belongs to token `t`. Check it: expert 0 occupies flat
  positions `[0, 23]`, and `[0//4, 23//4] = [0, 5]` — tokens 0 and 5. ✓

### 7.2 Gather, grouped matmul, scatter

```python
token_indices_experts_sorted = token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim)
routed_input = torch.gather(x, dim=0, index=token_indices_experts_sorted)   # [T·k, d]

if self.score_before_experts:
    routed_input = (routed_input.to(torch.float32)
                    * top_scores_experts_sorted.reshape(-1, 1)).to(x.dtype)

routed_output = self.experts(routed_input, num_tokens_per_expert)

out = out.scatter_add(dim=0, index=token_indices_experts_sorted, src=routed_output)
```

Note `.expand(-1, dim)` — stride-0 broadcast, no copy (the same trick as topic 07
§step 4).

And **`scatter_add`, not `scatter`**: a token appears `k` times in the sorted
array (once per chosen expert), and its final output is the *sum* of those `k`
weighted expert outputs. `scatter_add` performs exactly the `Σ_{i∈T} g_i E_i(x)`
of §1, and it adds on top of the shared expert's output which is already in
`out`. One kernel for the combine.

### 7.3 The grouped matmul

```python
offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
h = F.silu(torch._grouped_mm(routed_input.bfloat16(), w1.bfloat16().transpose(-2,-1),
                             offs=offsets))
h = h * torch._grouped_mm(routed_input.bfloat16(), w3.bfloat16().transpose(-2,-1),
                          offs=offsets)
```

`torch._grouped_mm` performs `E` matmuls of *different sizes* in one kernel,
using `offs` to delimit each expert's row range:

```
num_tokens_per_expert = [2, 3, 3, 2, 3, 3, 3, 4, 3, 2, 2, 2]
offsets               = [2, 5, 8, 10, 13, 16, 19, 23, 26, 28, 30, 32]
                         ↑  ↑
                         │  └─ rows 2..4 → expert 1
                         └──── rows 0..1 → expert 0
```

This is the kernel that makes fine-grained MoE (§6) practical: it is why 256
small experts do not cost 256 kernel launches.

Two consequences that echo through the rest of Part IV:

- **`routed_input` must already be sorted by expert**, which is what §7.1 is for.
  Under expert parallelism the sort is done by the all-to-all instead (topic 29).
- **`offsets` is data-dependent**, so its values must reach the CPU to launch the
  kernel — a **device-to-host synchronization**. That is the sync that breaks
  FSDP's prefetch inference (topic 20 §7) and forces explicit prefetch.

### 7.4 `score_before_experts`

Multiply by the gate before or after the expert?

```
before:  E_i(g_i · x)
after:   g_i · E_i(x)
```

These differ — the FFN is not homogeneous (`silu` is nonlinear). The default is
`before`, whose systems advantage is that the scaling is fused into the gather
of `routed_input` and there is nothing left to do to `routed_output` before
scattering — which matters when the scatter is a network operation
(topic 29). Both are used in the literature; `after` is the more common formal
statement.

## 8. Dropless

Classic MoE implementations impose a **capacity** `C = f · k · T / E` per expert
and *drop* tokens beyond it, because static shapes were required for XLA/TPU.
Dropping is a real quality loss and interacts badly with token counting
(topic 11 §3.1).

This implementation is **dropless**: `torch._grouped_mm` accepts variable
per-expert row counts, so every token is processed no matter how skewed the
routing. That is why there is no `capacity_factor` in `MoEArgs`.

Dropless shifts the imbalance cost from *quality* to *wall-clock*: an overloaded
expert makes its owning rank slower, and everyone waits (topic 28). Which is
precisely why §4's balancing matters even though nothing is dropped.

## 9. Cost summary

For the reference config (`E=8, E_s=1, k=1, d=2048, d_ff^E=1408`):

| | value |
|---|---|
| params per MoE layer | 77.9 M |
| active per MoE layer | 17.3 M (22%) |
| sparsity ratio | 4.5× on the FFN, 2.2× overall |
| extra FLOPs vs dense | router: `2·d·E = 32 K`/token — negligible |
| extra memory | `E×` the expert weights, plus the permutation buffers |
| extra communication | **zero** without EP; 2 all-to-all per layer with EP (topic 29) |

Note the last row: **MoE with `P_ep = 1` needs no communication at all.** It is
purely a memory-for-quality trade. All-to-all appears only when the experts are
distributed, which is topic 28.

## 10. Check yourself

1. Give the two formulas from topic 02 that MoE exploits, and state precisely
   what it decouples.
2. Compute total and active parameters for `E=256, k=8, E_s=1, d=7168,
   d_ff^E=2048`. What sparsity ratio?
3. Contrast token choice with expert choice. What does each fail at, and why is
   token choice preferred?
4. Why is the router forced to fp32? Give the activation-checkpointing failure
   mode specifically.
5. Why sigmoid rather than softmax scoring?
6. Explain expert collapse as a property of the derivative of `argmax`.
7. Write the classical auxiliary loss and explain the role of the
   non-differentiable `f_i`.
8. State the two-line crux of auxiliary-loss-free balancing. Why does using the
   bias for selection but not for gating leave the objective undisturbed?
9. Why is `expert_bias` `persistent=True` and `tokens_per_expert`
   `persistent=False`?
10. Activation checkpointing double-counts `tokens_per_expert`. Why is that
    harmless?
11. Give both reasons for shared experts. Explain the overlap the code comment
    describes.
12. What is granularity, and why does higher granularity help? What limits it?
13. Walk through `argsort(stable=True)` then `// top_k` on the §7.1 example and
    say what each step produces.
14. Why `scatter_add` and not `scatter`?
15. What does `offs` do in `torch._grouped_mm`, and which distributed subsystem
    does its data-dependence break?
16. What does "dropless" mean, and where does the imbalance cost show up
    instead?
17. Why does MoE need zero extra communication at `P_ep = 1`?

## Next

→ [27 — Tensor parallelism for MoE](27-tensor-parallelism-for-moe.md)
