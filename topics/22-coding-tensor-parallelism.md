# 22 — Coding Tensor Parallelism

> Video: 13:44:57 — Coding Tensor Parallelism
> Code: `torchfeather/model/parallelize.py` (`apply_non_moe_tp`),
> `torchfeather/distributed/__init__.py` (`NoParallel`)

## Why this exists

Topic 21 derived what TP must do. Implementing it means expressing a **plan**: a
map from module names to `ParallelStyle`s. PyTorch then does the sharding and
inserts the collectives.

The plan is the most information-dense artefact in the codebase. Once you can
read it, you can read any parallel model. So this topic is a line-by-line reading
of a real plan for a real MLA + MoE transformer, and the mechanism behind it.

## 1. The mechanism: what a `ParallelStyle` actually does

The comment in the source is the best three-line description of TP anywhere:

```python
# Each "ParallelStyle" just exposes an _apply method that:
# - Shards the parameter of the module to which it is applied (if necessary)
# - Registers a pre_forward_hook to make sure the input conforms to the expected
#   input layout (the argument IS NOT the desired input layout, but the real layout)
# - Registers a forward_hook (called after the fwd method) to make sure the output
#   conforms to the expected output layout (the argument IS the desired output layout)
```

So `parallelize_module(model, mesh, plan)` walks the plan and, for each entry:

1. converts the module's parameters into DTensors with the right placement,
2. installs a **pre-forward hook** that redistributes the *input* into what the
   sharded computation needs,
3. installs a **forward hook** that redistributes the *output* into what the next
   module needs.

Every collective in TP comes from a hook doing a `redistribute` — and topic 19
§3's table says which collective that is. **There is no TP code. There are only
placements, and the collectives implied by changing them.**

### 1.1 The asymmetry in `input_layouts` vs `output_layouts`

Read that comment again, because this trips up everyone once:

```
input_layouts  = what the input ACTUALLY IS on arrival
output_layouts = what the output SHOULD BE on departure
```

`input_layouts` is a *description* (the hook computes the needed redistribution
from it); `output_layouts` is a *prescription*. Get them mentally swapped and
every plan you write will insert the wrong collectives.

## 2. The model-level plan

```python
parallelize_module(model, tp_mesh, {
    "tok_embeddings": RowwiseParallel(
        input_layouts=Replicate(),   # the input token IDs live on each rank
        output_layouts=Shard(1),     # sharded on the sequence dim, which is
                                     # what the norm expects
    ),
    "norm": SequenceParallel(),
    # loss_parallel requires the logits to be split on the vocabulary dimension
    "output": ColwiseParallel(
        input_layouts=Shard(1),
        output_layouts=Shard(-1) if loss_parallel else Replicate(),
        use_local_output=not loss_parallel,
    ),
})
```

### 2.1 The embedding is *row*-wise, and that is not a typo

`nn.Embedding`'s weight is `[V, d]`, and a lookup is a matmul against a one-hot
vector. Rowwise = shard the **vocabulary**:

```
rank p holds rows [p·V/P, (p+1)·V/P) of the embedding table
```

Rank `p` can only look up tokens whose id falls in its range; for other tokens it
contributes zeros. Summing across ranks gives the true embedding — a
`Partial(sum)`, hence the row-parallel pattern. The output hook then redistributes
`Partial → Shard(1)`, which is a **reduce-scatter** (topic 19 §3).

Two consequences:

- The `V × d = 210M`-parameter embedding is sharded `P` ways. At `V = 102400`
  that is worth having.
- The output arrives already sharded on the sequence, ready for the
  sequence-parallel norm. `Partial → Shard(1)` is a reduce-scatter, which is
  *half* the bytes of the `Partial → Replicate` all-reduce you would need
  without sequence parallelism. The comment "which is what the norm expects"
  names the reason.

### 2.2 `SequenceParallel()` on the norms

`SequenceParallel` = replicate the (tiny, `d`-element) norm weight, and keep the
activation `Shard(1)`. Norms are elementwise per token (topic 21 §5), so each
rank normalizes its own `S/P` tokens correctly with no communication at all.

### 2.3 The output projection and loss parallelism

```python
"output": ColwiseParallel(
    input_layouts=Shard(1),                                  # from the final norm
    output_layouts=Shard(-1) if loss_parallel else Replicate(),
    use_local_output=not loss_parallel,
),
```

`input_layouts=Shard(1)` says the input arrives sequence-sharded from `norm`;
the hook all-gathers it to `Replicate()` for the matmul. Then:

- **`loss_parallel=True`:** output stays `Shard(-1)` — vocabulary-sharded — and
  `use_local_output=False` keeps it a **DTensor** so the loss can dispatch a
  distributed cross-entropy. Saves the 6.7 GB gather of topic 21 §6.
- **`loss_parallel=False`:** `Replicate()` and a plain local tensor, so an
  ordinary `F.cross_entropy` works. Simpler, much more memory.

## 3. The block-level plan, line by line

```python
layer_plan = {
    "attention_norm": SequenceParallel(),
    "attention": PrepareModuleInput(
        input_layouts=(Shard(1), Replicate()),
        desired_input_layouts=(Replicate(), Replicate()),
    ),
    "attention.wkv_a":  NoParallel(use_local_output=False),   # not per-head
    "attention.wkv_b":  ColwiseParallel(use_local_output=False),  # per-head
    "attention.kv_norm": NoParallel(use_local_output=False),
    "attention.inner_attention": PrepareModuleInput(
        input_layouts=(Shard(1), Shard(1), Shard(1)),         # split Q,K,V per head
        desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
        use_local_output=False,
    ),
    "attention.wo":  RowwiseParallel(output_layouts=Shard(1)),
    "ffn_norm":      SequenceParallel(),
}
if transformer_block.attention.q_lora_rank == 0:
    layer_plan["attention.wq"] = ColwiseParallel(use_local_output=False)
```

### 3.1 `PrepareModuleInput` on `attention` — crossing the SP/TP boundary

```python
"attention": PrepareModuleInput(
    input_layouts=(Shard(1), Replicate()),
    desired_input_layouts=(Replicate(), Replicate()),
),
```

Two arguments, because `Attention.forward(self, x, freqs_cis)` takes two:

- **`x`**: arrives `Shard(1)` from the sequence-parallel `attention_norm`; the
  matmuls need the full sequence, so redistribute to `Replicate()`. **This is the
  all-gather that enters the TP region** (topic 21 §5).
- **`freqs_cis`**: `Replicate()` → `Replicate()`, a no-op. It is a buffer, not a
  parameter, and it must not be sharded (topic 04 §6.1).

`PrepareModuleInput` exists precisely for this: a module whose *parameters* need
no special handling but whose *inputs* must change layout. Note the arity must
match the forward signature exactly — a mismatch is an obscure tuple error.

### 3.2 Which MLA matrices get which style

Straight from topic 21 §3.1 — the rule is **"is it per head?"**:

| Module | Style | Why |
|--------|-------|-----|
| `wq` | `ColwiseParallel` | output is `h·(128+64)` — per head |
| `wkv_a` | **`NoParallel`** | output is `d_c + 64` — **shared** by all heads |
| `kv_norm` | **`NoParallel`** | operates on the shared latent |
| `wkv_b` | `ColwiseParallel` | output is `h·(128+128)` — per head |
| `wo` | `RowwiseParallel` | input is `h·128` — per head (rows) |

The source comments say it in fewer words:

```python
"attention.wkv_a": NoParallel(use_local_output=False),  # No need to shard because it's not per-head
"attention.wkv_b": ColwiseParallel(use_local_output=False),  # this is per-head, so we can shard
```

The `q_lora_rank > 0` branch applies the same test to the compressed query path:

```python
"attention.wq_a":  NoParallel(use_local_output=False),      # not per-head
"attention.q_norm": NoParallel(use_local_output=False),     # on the compressed vec
"attention.wq_b":  ColwiseParallel(use_local_output=False), # per head
```

`wq_a: d → q_lora_rank` produces a head-agnostic compressed query, so it
replicates; `wq_b: q_lora_rank → h·192` expands per head, so it shards. **The
same structural rule, applied consistently.** Once you see that MLA has a
"shared trunk, per-head branches" shape, the whole plan is determined.

### 3.3 `use_local_output=False` everywhere inside attention

Notice every attention entry sets it. `use_local_output=True` (the default)
converts the module's DTensor output to a plain local tensor at the boundary.
Inside attention we do **not** want that — the tensors must stay DTensors so
that:

- placements propagate to the next module, letting PyTorch insert (or skip)
  collectives correctly;
- the composition with context parallelism works. Recall topic 06 §7.3: the
  attention wrapper detects `isinstance(q, DTensor)` and does the TP→local→CP
  round-trip. If `use_local_output=True` had already stripped the DTensor, the
  wrapper could not recover the TP spec, and CP would rotate `K`/`V` on the wrong
  process group.

So `use_local_output=False` inside a parallel region and `True` at its edges.
Getting this wrong produces either redundant collectives or silently wrong
grouping.

### 3.4 `inner_attention` — the CP attachment point

```python
"attention.inner_attention": PrepareModuleInput(
    input_layouts=(Shard(1), Shard(1), Shard(1)),      # split Q,K,V per head
    desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
    use_local_output=False,
),
```

Input and desired layouts are **identical** — so this inserts no collective at
all. Why have it?

Because `Shard(1)` here is the **head** dimension (`q` is `[B, h, S, d_h]` at
this point, so dim 1 is the head), and `q/k/v` arrive already head-sharded from
the column-parallel projections. The entry exists to **declare and pin** that
layout, so that:

- PyTorch knows not to insert a gather (which it might otherwise infer),
- the module is a registered parallel module and therefore a valid hook point
  for `_ContextParallel` (topic 06 §7.1 — this is why the attention function was
  wrapped in a `Module` at all).

`PrepareModuleInput` with equal in/out layouts is a **type annotation for the
distributed layout**. It generates no code and prevents a lot of wrongness.

Note also the same `Shard(1)` means different things in different entries: the
*sequence* dim for `x` (`[B, S, d]`) and the *head* dim for `q` (`[B, h, S,
d_h]`). Placements index tensor dimensions, not semantics — read them against
the shape.

### 3.5 `wo` closes the TP region back into SP

```python
"attention.wo": RowwiseParallel(output_layouts=Shard(1)),
```

Row-parallel produces `Partial(sum)`. `output_layouts=Shard(1)` requests
sequence-sharded output, so the hook performs `Partial → Shard(1)` — a
**reduce-scatter**, not an all-reduce (topic 19 §3). Half the bytes, and the
result lands in exactly the layout `ffn_norm` (sequence-parallel) wants.

That is sequence parallelism, fully realized in one keyword argument:

```
attention_norm  Shard(1)  ──all-gather──►  TP region  ──reduce-scatter──►  Shard(1)  ffn_norm
```

Total: `V(P−1)/P + V(P−1)/P = 2V(P−1)/P` = exactly one all-reduce's worth
(topic 21 §5.1). No extra bytes, `P×` less activation memory in the norm
regions.

### 3.6 The dense FFN

```python
if not transformer_block.moe_enabled:
    layer_plan.update({
        # The input of this comes from `ffn_norm`, which is SP.
        # We are entering a TP region after a SP region, so we must have the
        # entire input as required by TP.
        "feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),), desired_input_layouts=(Replicate(),)),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        "feed_forward.w3": ColwiseParallel(),
    })
```

Exactly topic 21 §2: `w1`/`w3` column-parallel, `w2` row-parallel with a
reduce-scatter back to `Shard(1)`. The `PrepareModuleInput` all-gathers out of
the SP region, and its comment states the rule in full.

Note there is no `use_local_output=False` here — the FFN's internals are plain
elementwise ops that work fine on local tensors, and the boundaries are handled
by the two hooks. Contrast with attention, where CP needs the DTensor metadata.

MoE layers are excluded and handled by `apply_moe_ep_tp` — topics 27–30.

## 4. `NoParallel`: replication as a first-class style

There is no built-in "replicate this module" style, so the codebase defines one:

```python
class NoParallel(ParallelStyle):
    def _apply(self, module, device_mesh):
        return distribute_module(
            module, device_mesh,
            None,   # if this is None, the weights are replicated by default
            partial(self._prepare_input_fn, self.input_layout, self.desired_input_layout),
            partial(self._prepare_output_fn, self.output_layout, self.use_local_output,
                    self.output_grad_placements))
```

`partition_fn=None` means "leave the parameters replicated". The two hooks then
handle layout conversion:

```python
@staticmethod
def _prepare_input_fn(input_layout, desired_input_layout, mod, inputs, device_mesh):
    input_tensor = inputs[0]
    if not isinstance(input_tensor, DTensor):
        input_tensor = DTensor.from_local(input_tensor, device_mesh,
                                          (input_layout,), run_check=False)
    if input_layout != desired_input_layout:
        input_tensor = input_tensor.redistribute(
            placements=(desired_input_layout,), async_op=True)
    return (input_tensor, *inputs[1:])
```

Three details worth extracting, because you will write hooks like these:

- **`DTensor.from_local(..., run_check=False)`.** Wraps a plain local tensor with
  a placement *assertion*, no collective. `run_check=False` skips validating that
  ranks actually agree — a collective in itself. You are asserting the layout,
  not verifying it, which is fast and correct as long as your plan is right.
- **`async_op=True`** on the redistribute: issue the collective and let it
  overlap (topic 19 §4.1).
- **`(input_tensor, *inputs[1:])`.** Only the first argument is touched;
  everything else passes through. Which is why `NoParallel` works on `kv_norm`
  (one input) and on `wkv_a` alike.

Why a whole class for "do nothing"? Because in a DTensor world, **replication is
a placement, not an absence of one.** A module left entirely out of the plan
receives plain tensors and its output has no placement metadata, so the next
module's hook cannot reason about it. `NoParallel` keeps the module inside the
DTensor type system while leaving its weights whole. It is the `Replicate()`
placement, given a name.

## 5. The complete data flow through one block

```
x  Shard(1)                                    [B, S/P, d]      ← sequence-parallel
│
├─ attention_norm          SequenceParallel     Shard(1)     no comm
│
├─ PrepareModuleInput      Shard(1)→Replicate   [B, S, d]    ALL-GATHER  ◄── enter TP
│  │
│  ├─ wq       Colwise      → [B, S, h/P·192]   per head
│  ├─ wkv_a    NoParallel   → [B, S, 576]       replicated (shared latent)
│  ├─ kv_norm  NoParallel   → [B, S, 512]       replicated
│  ├─ wkv_b    Colwise      → [B, S, h/P·256]   per head
│  ├─ inner_attention       Shard(1)=head       LOCAL — no comm
│  └─ wo       Rowwise      Partial→Shard(1)    REDUCE-SCATTER  ◄── leave TP
│
├─ + residual              Shard(1)
├─ ffn_norm                SequenceParallel     Shard(1)     no comm
│
├─ PrepareModuleInput      Shard(1)→Replicate   ALL-GATHER  ◄── enter TP
│  ├─ w1, w3   Colwise      → [B, S, d_ff/P]
│  ├─ silu ⊙                 elementwise         LOCAL
│  └─ w2       Rowwise      Partial→Shard(1)    REDUCE-SCATTER  ◄── leave TP
│
└─ + residual              Shard(1)                          → next block
```

**Four collectives per block per forward pass** (2 AG + 2 RS), which is exactly
two all-reduces' worth of bytes (topic 21 §5.1) — the same as TP without
sequence parallelism, with `P×` less activation memory in the norm and residual
regions.

## 6. Writing your own plan — a checklist

For each `nn.Linear` in a new architecture:

1. **Does its output have an independent axis** (head, expert, FFN hidden)?
   → `ColwiseParallel` on that axis.
2. **Does its input have that same axis** (i.e. it is consuming a sharded
   activation)? → `RowwiseParallel`, and choose `output_layouts`:
   `Shard(1)` if the next region is sequence-parallel, `Replicate()` otherwise.
3. **Neither?** → `NoParallel`. Replicate the compute; it is cheap if the matrix
   is small. If it is *not* small, you have found a real bottleneck and need a
   different factorization.
4. **Between a Colwise and a Rowwise, is everything elementwise along the sharded
   dim?** (topic 21 §2.1) If not, the pair is invalid and you need a gather in
   the middle.
5. **At every region boundary**, insert `PrepareModuleInput` describing the
   *actual* incoming layout and the *desired* one.
6. **Inside a region**, `use_local_output=False`. **At the edges**, `True` unless
   another parallelism needs the metadata.
7. **Test against a single-rank reference** (topic 21 §8). A wrong plan does not
   crash.

## 7. Check yourself

1. What three things does a `ParallelStyle._apply` do?
2. `input_layouts` and `output_layouts` are asymmetric in meaning. State both
   precisely.
3. Why is `tok_embeddings` `RowwiseParallel`? What placement does its output
   start as, and which collective converts it to `Shard(1)`?
4. Why does the embedding's `output_layouts=Shard(1)` save bytes compared to
   `Replicate()`?
5. For each of `wq`, `wkv_a`, `kv_norm`, `wkv_b`, `wo`, give the style and the
   one-line reason.
6. Apply the same rule to `wq_a`, `q_norm`, `wq_b`.
7. Why `use_local_output=False` throughout attention? Give the CP-specific reason
   from topic 06 §7.3.
8. `inner_attention`'s input and desired layouts are identical. Give two reasons
   the entry exists anyway.
9. `Shard(1)` appears on both `x` and `q`. What does it mean in each case?
10. Explain how `RowwiseParallel(output_layouts=Shard(1))` implements sequence
    parallelism, and count the bytes against plain TP.
11. Why does `NoParallel` exist when the weights are replicated by default?
12. What does `DTensor.from_local(..., run_check=False)` do, and what does
    `run_check=True` cost?
13. Count the collectives per block per forward pass and compare to non-SP TP.
14. You add a new module: `nn.Linear(d, d)` followed by a softmax over its output
    dimension, followed by `nn.Linear(d, d)`. Can you Colwise→Rowwise it? If not,
    what is the minimum fix?

## Next

→ [23 — Context parallelism and ring attention](23-context-parallelism-ring-attention.md)
