# 14 — Pipeline Parallelism from First Principles

> Video: 06:06:17 — Pipeline Parallelism from First Principles
> Paper: GPipe (1811.06965)

## Why this exists

DDP (topic 12) gives throughput but no memory relief — every rank holds a full
replica. For a 70B model at 18 bytes/param that is 1.26 TB per rank, and no
number of ranks fixes it. Something has to shard the model itself.

The most obvious way to shard a model is the way it is written: **it is a stack
of `L` layers, so give each device `L/P_pp` of them.** That is pipeline
parallelism. It is the crudest of the four cuts and, for that reason, the one
with the fewest constraints — which is why it is the cut you put across your
*slowest* network links.

The interesting content of this topic is not the idea (it is obvious) but the
**bubble**: the precise fraction of the machine that pipeline parallelism wastes,
and the two independent ways to reduce it.

## 1. The cut

```
                  x → [L₀ … L₆] → [L₇ … L₁₃] → [L₁₄ … L₂₀] → [L₂₁ … L₂₆] → loss
                        rank 0        rank 1        rank 2        rank 3
```

Each rank holds `L/P_pp` layers, so per-rank parameter and optimizer memory
divides by `P_pp`. The cuts are the arrows, and by topic 11 §5 each one costs:

```
forward:   send activation h   [B, S, d]      point-to-point
backward:  recv gradient ∂L/∂h [B, S, d]      point-to-point
```

**Point-to-point, not a collective.** This is the defining property of PP and the
source of all its advantages. Only two ranks are involved per cut; nothing is
reduced; the message size depends on the *activation*, not on the model.

## 2. The naive version wastes `1 − 1/P_pp` of the machine

Run one batch through:

```
time ────────────────────────────────────────────────────────────────►
rank 0   F₀ ······································· B₀
rank 1   ··· F₁ ························· B₁ ···········
rank 2   ······ F₂ ············· B₂ ····················
rank 3   ········· F₃  B₃ ······························
```

Rank 1 cannot start until rank 0 finishes; rank 0 cannot do its backward until
rank 3's has propagated back. **Exactly one rank is busy at any instant.**
Utilization is `1/P_pp` — with `P_pp = 4` you bought four GPUs and got one.

This is worse than useless as stated. The fix is the whole technique.

## 3. Micro-batching: fill the pipe

Split the batch into `M` **micro-batches** and stream them. This is legal by
topic 11 §3 corollary (a): gradients are additive, so

```
∂L/∂θ = (1/M) Σ_m ∂L_m/∂θ
```

and the micro-batch gradients may be computed in any order, at any time, as long
as they all land in `.grad` before `optimizer.step()`.

```
time ──────────────────────────────────────────────────────────────────►
rank 0   F₁ F₂ F₃ F₄ ·  ·  ·  ·  B₁ B₂ B₃ B₄
rank 1   ·  F₁ F₂ F₃ F₄ ·  ·  B₁ B₂ B₃ B₄ ·
rank 2   ·  ·  F₁ F₂ F₃ F₄ B₁ B₂ B₃ B₄ ·  ·
rank 3   ·  ·  ·  F₁ F₂ F₃ F₄ B₁ B₂ B₃ B₄
         └── fill ──┘         └── drain ──┘
             (bubble)             (bubble)
```

Now all four ranks are busy in the middle. Only the **fill** and **drain** phases
are idle. That idle time is the *bubble*, and it is the central quantity of
pipeline parallelism.

## 4. The bubble, derived

Let `t_f` be the time for one micro-batch's forward on one stage and `t_b` the
time for its backward. Assume stages are balanced (§7 discusses when they are
not). Recall from topic 02 §3.1 that `t_b ≈ 2 t_f`.

For the GPipe schedule (all forwards, then all backwards):

**Forward phase.** Stage `P−1` receives micro-batch 1 only after it has passed
through `P−1` earlier stages, so it starts at time `(P−1)·t_f`. It then processes
`M` micro-batches back to back. The forward phase ends at

```
T_fwd = (P − 1) · t_f + M · t_f = (M + P − 1) · t_f
```

**Backward phase.** By symmetry (the backward wave travels from stage `P−1` to
stage `0`):

```
T_bwd = (M + P − 1) · t_b
```

**Total, and the ideal:**

```
T_total = (M + P − 1) · (t_f + t_b)
T_ideal =  M          · (t_f + t_b)         ← if the pipe were always full
```

```
┌──────────────────────────────────────────────────────────────┐
│                T_total − T_ideal        P − 1                │
│  bubble  =  ─────────────────────  =  ───────────            │
│                    T_total            M + P − 1              │
└──────────────────────────────────────────────────────────────┘
```

Note the bubble depends **only** on `M` and `P`, not on the model, the layer
shapes, or the interconnect. Concretely:

| `P_pp` | `M = 1` | `M = 4` | `M = 8` | `M = 16` | `M = 64` |
|--------|---------|---------|---------|----------|----------|
| 2 | 50% | 20% | 11% | 6% | 1.5% |
| 4 | 75% | 43% | 27% | 16% | 4.5% |
| 8 | 88% | 64% | 47% | 30% | 10% |
| 16 | 94% | 79% | 65% | 48% | 19% |

Two rules fall straight out:

> **Rule 1: `M ≫ P_pp`.** With `M = P_pp` you lose ~half the machine. A common
> target is `M ≥ 4·P_pp`, giving a bubble under 20%.
>
> **Rule 2: keep `P_pp` small.** The bubble grows with `P_pp` at fixed `M`. Use
> pipeline parallelism to make the model *fit*, then stop.

The reference code warns about exactly Rule 1:

```python
if n_microbatches < num_total_stages:
    logger.warning(
        f"Number of microbatches ({n_microbatches}) is less than the total number "
        f"of stages ({num_total_stages}) which may result in a bubble in the pipeline.")
```

### 4.1 The tension you cannot escape

`M = local_batch_size / microbatch_size`. To raise `M` you either raise the batch
size (which changes the optimization problem — there is a critical batch size
beyond which more data per step stops helping) or shrink the micro-batch (which
lowers arithmetic intensity per matmul and can make each stage slower).

So the bubble cannot simply be dialled to zero. It has to be attacked
structurally, which is topic 15: **rearrange the schedule**.

## 5. Memory: GPipe's second problem

Look at rank 0 in the §3 diagram. It runs `F₁ F₂ F₃ F₄` before any backward
happens, and each forward's activations must stay alive until its own backward
runs. So rank 0 holds **`M` micro-batches' worth of activations simultaneously**.

```
GPipe activation memory on stage 0  ∝  M × (activations of L/P_pp layers)
```

That is a disaster, and it directly contradicts §4's Rule 1: the thing that
shrinks the bubble is the thing that grows the memory.

```
   raise M  →  bubble ↓   memory ↑
   lower M  →  bubble ↑   memory ↓
```

Resolving that contradiction is the entire subject of the next topic. The key
observation, in one sentence: **stage 0 does not need to run all `M` forwards
before its first backward — it only needs to stay `P` micro-batches ahead.**
That is 1F1B, and it caps memory at `O(P)` instead of `O(M)` while leaving the
bubble unchanged.

## 6. Communication cost, and why PP goes on the slow link

Per micro-batch, per stage boundary, one direction:

```
V_pp = B_micro · S · d · bytes_per_elem
```

For `B_micro = 1, S = 16384, d = 2048`, bf16:

```
V_pp = 1 · 16384 · 2048 · 2 = 67 MB
```

Per training step, with `P_pp = 4` (3 boundaries), `M = 8`, both directions:

```
3 boundaries × 8 micro-batches × 2 directions × 67 MB = 3.2 GB
```

At 25 GB/s inter-node, that is **0.13 s per step** — and it overlaps with
compute, because while a stage sends micro-batch `m` downstream it can start
micro-batch `m+1`.

Now compare against tensor parallelism on the same activation (topic 21): TP
performs **two all-reduces per layer**, each moving `2V(P−1)/P ≈ 2 · 67 MB`:

```
27 layers × 2 collectives × 134 MB × (fwd + bwd) ≈ 14 GB per micro-batch
                                     × 8 micro-batches ≈ 116 GB per step
```

**Roughly 36× more traffic than PP, and it is on the critical path of every
layer** rather than once per stage boundary. That asymmetry is the quantitative
form of topic 01's bandwidth rule:

```
┌──────────────────────────────────────────────────────────────┐
│  PP:  O(P_pp) messages per step, point-to-point, overlappable│
│       →  put it ACROSS nodes (InfiniBand, 25 GB/s)           │
│                                                              │
│  TP:  O(L) collectives per step, on the critical path        │
│       →  keep it WITHIN a node (NVLink, 900 GB/s)            │
└──────────────────────────────────────────────────────────────┘
```

Also note `V_pp` is independent of the number of *parameters*. A 671B model and
a 7B model with the same `S` and `d` ship identical pipeline messages. PP's
communication does not grow with the thing it is sharding — a property no other
parallelism has.

## 7. Load balancing: the stages are not equal

The bubble analysis assumed equal `t_f` per stage. Real models are lopsided:

```
stage 0:  embedding (a gather: ~0 FLOPs) + L/P layers
stage P−1: L/P layers + final norm + output projection [d × V]
```

The output projection is a `2048 × 102400` matmul — **210M parameters, 12% of
the entire dense parameter count** (topic 02 §2.5) — and it produces a
`[B, S, V]` logits tensor. The embedding, by contrast, costs almost nothing.

Since the pipeline runs at the speed of its slowest stage, an overloaded last
stage inflates `t_f` for everyone. The fix is to give the first and last stages
*fewer transformer layers*, and the reference code parameterizes exactly that:

```python
# How many "layers" is the embedding layer equivalent to?
input_weight  = job_config.parallelism.pipeline_parallel_first_stage_less_layers
# How many "layers" is the output layer (norm + output) equivalent to?
output_weight = job_config.parallelism.pipeline_parallel_last_stage_less_layers

num_effective_layers = num_layers + input_weight + output_weight
```

The embedding and the output head are each charged as some number of equivalent
transformer layers, and the split is computed over `num_effective_layers`. With
`L = 27, P = 4, input_weight = 1, output_weight = 1`:

```
num_effective_layers = 29,  layers_per_stage = 7, extra = 1

stage 0: tok_embeddings + layers 0-6      (8 slots − 1 for embedding = 7 layers)
stage 1: layers 7-13                      (7 layers)
stage 2: layers 14-20                     (7 layers)
stage 3: layers 21-26 + norm + output      (7 slots − 1 for output = 6 layers)
```

Note `extra_layers` goes to the *earliest* stages (`if stage_idx < extra_layers`),
so remainders never pile onto the already-heavy last stage.

Tuning these two weights is a real and worthwhile activity: a 10% imbalance in
the slowest stage is a 10% throughput loss across the whole run. Measure per-stage
step time before guessing.

## 8. What PP gives and what it costs

| | |
|---|---|
| parameter + optimizer memory | ÷ `P_pp` ✓ |
| activation memory | ÷ `P_pp` per stage, but × `M` live micro-batches (GPipe) |
| communication per step | `O(P_pp · M)` point-to-point messages of `B·S·d` |
| tolerance of slow links | **excellent** — the only cut you should place across nodes |
| utilization cost | the bubble, `(P_pp − 1)/(M + P_pp − 1)` |
| throughput gain | **none** — PP makes the model fit, it does not add speed |
| implementation complexity | **highest of the four**: schedules, `detach` boundaries, explicit gradient sends |
| composes with | DP/FSDP (outer), TP/CP (inner) — topic 18 |

The last two rows are why PP is often the *last* parallelism you reach for: use
FSDP and TP first, and add PP only when the model still does not fit, or when
you have run out of intra-node bandwidth for TP.

## 9. Check yourself

1. Derive `T_total = (M + P − 1)(t_f + t_b)` for GPipe, explaining where the
   `P − 1` comes from in each phase.
2. Derive the bubble fraction. Why does it not depend on the model or the
   interconnect?
3. `P_pp = 8`. What `M` do you need for a bubble under 15%? What are the two
   costs of getting `M` that large?
4. GPipe's activation memory on stage 0 scales with `M`, but the bubble shrinks
   with `M`. State the contradiction, and the one-sentence idea that resolves it.
5. Compute `V_pp` for `B_micro = 2, S = 8192, d = 4096` in bf16, and the total
   step traffic for `P_pp = 8, M = 32`.
6. Give the quantitative argument for putting PP across nodes and TP inside a
   node.
7. PP's message size is independent of the parameter count. Why, and which other
   parallelism has that property?
8. Why do the first and last stages get fewer transformer layers? Compute the
   split for `L = 32, P = 4, input_weight = 1, output_weight = 2`.
9. Why does `extra_layers` go to the earliest stages rather than the last?
10. PP divides parameter memory by `P_pp` but adds no throughput. Reconcile that
    with the fact that large runs use it anyway.

## Next

→ [15 — Pipeline schedules: GPipe, 1F1B, zero bubble](15-pipeline-schedules.md)
