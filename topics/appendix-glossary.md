# Appendix — Glossary

Each entry gives the definition and the topic where it is derived.

## A

**Activation checkpointing (AC)** — Discard intermediate activations in the
forward pass and recompute them during backward. Trades `+2N` FLOPs for a large
memory saving (`6N → 8N`). Modes: `none`, `full`, `selective`. → t02 §3.4, t20 §10

**Adjoint** — The transpose of a linear operator. Since every collective is
linear, **the backward pass of a collective is its adjoint** — which is why the
backward collective is derivable rather than memorized. → t11 §1.4

**All-gather (AG)** — Every rank receives the concatenation of all ranks' shards.
`Shard(i) → Replicate()`. Moves `V(P−1)/P` bytes; adjoint is reduce-scatter.
→ t19 §1

**All-reduce (AR)** — Every rank receives the reduction of all ranks'
contributions. `Partial → Replicate()`. Moves `2V(P−1)/P` — exactly a
reduce-scatter followed by an all-gather. Self-adjoint. → t19 §2

**All-to-all (A2A)** — A distributed transpose: each rank sends a distinct chunk
to every other rank. `Shard(i) → Shard(j)`. Its own adjoint with split/concat
dims swapped. The collective of expert parallelism. → t19 §1, t29

**Arithmetic intensity (AI)** — FLOPs performed per byte moved to/from HBM.
Compared against the hardware **ridge point** (~295 FLOP/byte on an H100 in
bf16) to decide whether a kernel is compute- or memory-bound. Decode attention
sits at `AI = 1`. → t06 §3

**Auxiliary-loss-free balancing** — Balance an MoE router by adding a bias to the
scores **for selection only**, leaving the gating value from the original scores.
Updated by a `sign`-based rule in a post-step hook, so the loss is undisturbed.
→ t26 §4.2, t24 §1.3

## B

**Bubble** — Pipeline idle time during the fill and drain phases.
`(P_pp − 1)/(M + P_pp − 1)`. Depends only on `M` and `P_pp`, not on the model.
→ t14 §4

**Bucketing** — Concatenating many small gradient tensors into ~25–100 MB flat
buffers before all-reducing, to amortize per-collective latency. Buckets are
formed in reverse parameter order because that is the order gradients complete.
→ t12 §3.2

## C

**Capacity factor** — In classic MoE, `C = f · k · T / E` tokens per expert;
overflow is dropped. Absent from this implementation, which is **dropless**.
→ t26 §8

**Chinchilla** — The compute-optimal scaling result `D ≈ 20N`. Optimizes training
loss per *training* FLOP; inference cost pushes production models past it.
→ t02 §3.6

**Colwise / column parallel** — Shard a weight matrix by its **output**
dimension. `Replicate × Shard(1) → Shard(1)`. Needs no collective. → t08 §2.1

**Context parallelism (CP)** — Shard the sequence dimension across devices.
Everything is local except attention, which needs ring attention. Also folded
into the FSDP group (`fsdp = dp_shard × cp`). → t23

**Cotangent** — `∂L/∂h`, the row vector reverse-mode autodiff propagates.
→ t11 §1.2

## D

**DDP** — Data-parallel training with a full model replica per rank and one
gradient all-reduce per step. `P×` throughput, **no** memory relief. → t12

**Decoupled RoPE** — MLA's split of the head dimension into a content part (no
position, absorbable) and a positional part (RoPE'd, not absorbable, key shared
across heads). Forced by the incompatibility of absorption with RoPE. → t09 §5

**Device mesh** — An `n`-D array of ranks with named dimensions; slices are
process groups. `torchfeather` builds three named *views* over one flat world
mesh (`dataloading`, `dense`, `sparse`). → t18

**Dropless** — MoE with no capacity limit: `torch._grouped_mm` accepts variable
per-expert row counts, so no token is dropped. Shifts the cost of imbalance from
quality to wall-clock. → t26 §8, t28 §4

**DTensor** — A local tensor + a device mesh + a placement per mesh dimension.
`redistribute()` converts between placements, and the placement pair determines
the collective. → t19 §3

## E

**Expert parallelism (EP)** — Shard the expert index; each rank owns `E/P_ep`
whole experts. Requires an all-to-all dispatch and combine. Borrows ranks from
`dp_shard`, `cp` and (with `etp=1`) `tp` rather than adding a mesh axis. → t28

**Expert tensor parallelism (ETP)** — EP and TP composed on a 2-D `(ep, etp)`
mesh: shard the expert index *and* slice each expert's weights. `etp` must be
`tp` or `1`. → t30

## F

**FSDP / ZeRO-3** — Shard parameters, gradients and optimizer state across
ranks; all-gather a module's weights just before use and free them after.
Divides 18 bytes/param by `P`, at 1.5× DDP's communication. → t20

**`fully_shard`** — PyTorch's FSDP2 API. Called per `TransformerBlock`, which is
the right granularity. → t20 §2

## G

**Granularity** — How finely the FFN is split into experts (`G =
d_ff^dense / d_ff^E`). Higher is better for quality at fixed active parameters,
until routing/all-to-all overhead dominates. → t26 §6

**Grouped matmul** — One kernel performing `E` matmuls of different sizes, using
`offs` to delimit each expert's rows. Makes fine-grained MoE practical; its
data-dependent offsets force a device-to-host sync. → t26 §7.3

## H

**HSDP** — Hierarchical sharded data parallelism: a 2-D data-parallel mesh,
sharding within `dp_shard` and replicating across `dp_replicate`. Keeps
all-gathers local and pays one cross-replica all-reduce per step. → t20 §6

## L

**Loss parallelism** — Keep the `[B, S, V]` logits vocabulary-sharded and compute
cross-entropy distributively (two `[B, S]` reductions for the logsumexp) rather
than gathering 6.7 GB of logits. → t21 §6

## M

**MFU** — Model FLOPs Utilization: achieved FLOP/s divided by the cluster's peak.
Always state your convention (`6N` vs `8N`, masked vs unmasked attention).
→ t02 §3.5

**MLA (Multi-head Latent Attention)** — Compress `K`/`V` into a shared latent
`c_t ∈ R^{d_c}` and reconstruct per-head keys and values with per-head
up-projections. Caches `d_c + d_h^R` values per token instead of `2·h·d_h`,
independent of `h`. → t07, t09

**Micro-batch** — One of `M` slices of the local batch, processed independently;
gradients accumulate additively. The unit a pipeline schedule moves. → t14 §3

**Mixed precision** — bf16 parameters and compute, fp32 master weights, optimizer
state, and **gradient reduction** (`reduce_dtype`). → t20 §4

**Monoid (attention as)** — Partial attention results `(m, ℓ, o)` merge
associatively, so key blocks can be processed in any order on any device. The
property ring attention needs. → t23 §3.1

## N

**`NoParallel`** — A `ParallelStyle` that keeps a module's weights replicated
while keeping the module inside the DTensor type system. Used for MLA's
head-independent path (`wkv_a`, `kv_norm`). → t22 §4, t21 §3.1

**NTK-by-parts** — YaRN's frequency interpolation: classify RoPE planes by
rotations completed at the training length (`r_i`), leave fast planes untouched,
fully interpolate slow ones, ramp between. → t04 §4

## O

**Online softmax** — The recurrence maintaining `(max, denominator, numerator)`
so softmax attention can be computed over blocks and merged. Powers both
FlashAttention (tiles) and ring attention (devices). → t23 §3

**1F1B** — Pipeline schedule alternating one forward and one backward in steady
state. Same bubble as GPipe, `O(P)` memory instead of `O(M)`. Strictly dominates
GPipe. → t15 §3

## P

**`Partial(sum)`** — DTensor placement: every rank holds a full-shaped tensor
whose values are incomplete; the truth is the sum. Produced by row-parallel
matmuls; resolved by all-reduce (`→ Replicate`) or reduce-scatter (`→ Shard`).
→ t08 §2.2, t19 §3

**Pipeline parallelism (PP)** — Shard the layers. Point-to-point communication
whose message size is independent of model size, so it tolerates the slowest
link. Costs the bubble and adds no throughput. → t14

**Pre-norm** — `x ← x + F(Norm(x))`. The identity term in `∂x_{ℓ+1}/∂x_ℓ = I + J`
is why gradients cannot vanish geometrically with depth. → t05 §1.1

**Prefetch** — Issuing a collective before it is needed, exploiting known layer
order. What makes FSDP's 1.5× volume often free. Must be **explicit** under EP,
whose D2H sync breaks order inference. → t19 §4.3, t20 §7

## R

**Reduce-scatter (RS)** — Reduce across ranks, then keep only your own slice.
`Partial → Shard(i)`. Moves `V(P−1)/P`; adjoint is all-gather. → t19 §1

**`reshard_after_forward`** — FSDP flag: free gathered parameters after the
forward (min memory, 2 all-gathers) or keep them (1 all-gather, more memory).
Defaults to `False` under PP to avoid `M` redundant all-gathers. → t20 §3

**Ridge point** — `peak FLOP/s ÷ peak bytes/s`. ~295 FLOP/byte for an H100 in
bf16. Anything below it is memory-bound. → t06 §3.1

**Ring attention** — Context-parallel attention: rotate `K`/`V` blocks around a
ring, merging partial results with online softmax. Needs zigzag sharding to
balance causal work. → t23

**RoPE** — Rotary position embedding. The unique linear isometry satisfying
`⟨f(q,m), f(k,n)⟩ = g(q,k,n−m)`: a block-diagonal rotation by `m·θ_i` with
`θ_i = base^(−2i/d)`. → t03

**Rowwise / row parallel** — Shard a weight matrix by its **input** (contraction)
dimension. `Shard(1) × Shard(0) → Partial(sum)`. Needs a reduction. → t08 §2.2

## S

**Sequence parallelism (SP)** — Keep activations sharded on the sequence
dimension in the norm/residual regions between TP blocks. Costs **exactly zero**
extra bytes (an all-gather plus a reduce-scatter equals one all-reduce) and saves
`P_tp ×` on activation memory there. Always enable with TP. → t21 §5

**`Shard(i)`** — DTensor placement: the tensor is split along dim `i` across the
mesh dimension. Self-adjoint (sharded stays sharded). → t11 §2.4

**Shared expert** — An always-on FFN every token passes through, alongside its
`k` routed experts. Frees routed experts to specialize, and its
communication-free compute is placed where the combine all-to-all is in flight.
→ t26 §5

## T

**Tensor parallelism (TP)** — Shard the weight matrices; never gather them.
Column-parallel chained into row-parallel, valid because everything between them
is elementwise along the sharded dimension. Two collectives per block, on the
critical path — keep it inside a node. → t21

**Token choice** — Each token picks its top-`k` experts (vs *expert choice*,
where experts pick tokens). Every token is served, at the cost of load
imbalance. → t26 §1.1

## V

**VJP (vector–Jacobian product)** — The single primitive every op must supply:
`ḡ ↦ ḡ · ∂output/∂input`. Reverse-mode autodiff is nothing but a chain of VJPs.
→ t11 §1.3

## W

**Warmup–stable–decay (WSD)** — LR schedule with a linear warmup, a long stable
phase, and a decay to `min_lr_factor · lr`. Lets one stable run be branched into
several decayed models. → t24 §2.1

**Weight absorption** — Folding `W_UK` into `W_Q` and `W_UV` into `W_O` so MLA
attends directly against the cached latent. Raises decode `AI` from 1 to ~30 by
*spending* FLOPs to save bytes. Inference-only and frozen. → t10

## Y

**YaRN** — NTK-by-parts frequency interpolation **plus** an attention-temperature
correction (`√(1/t) = 0.1·ln s + 1`, folded into `softmax_scale`). Implementing
only the first half is not YaRN. → t04

## Z

**Zero bubble** — Pipeline schedules exploiting that the backward pass splits into
`B` (input gradient, latency-critical) and `W` (weight gradient, no downstream
dependency within the step). `W` is deferred to fill bubbles. ZBV achieves ~zero
bubble at 1F1B memory using a V-shaped stage placement. → t15 §5

**Zigzag sharding** — Context-parallel load balancing: split the sequence into
`2·P_cp` chunks and give rank `r` chunks `r` and `2P−1−r`, so every rank's causal
work is `2P+1` half-blocks — exactly constant. The source of the `2` in
`seq_len_divisor = tp · (cp · 2)`. → t23 §4.3
