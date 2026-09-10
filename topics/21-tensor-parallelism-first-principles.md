# 21 — Tensor Parallelism from First Principles

> Video: 12:13:08 — Tensor Parallelism from First Principles
> Code: `minimal_examples/tp_swiglu.py`, `minimal_examples/tp_attention.py`
> Paper: Megatron-LM (2104.04473)

## Why this exists

FSDP shards parameters but **gathers them back** to compute. So at the instant
layer `i` runs, one rank holds all of layer `i`'s weights and all of the
`[B, S, d_ff]` activation. If a single layer does not fit — a very wide FFN, a
huge vocabulary, a long sequence — FSDP cannot help.

Tensor parallelism never gathers. Each rank holds a **slice of every matrix** and
computes a **slice of every activation**, for the whole step. The model is
permanently distributed.

The mathematics is already done: topic 08 §2 derived both primitives from block
matrix multiplication. This topic applies them to a real transformer block, and
then adds the two refinements that make TP practical — **sequence parallelism**
and **loss parallelism** — both of which turn out to be free.

## 1. The two primitives, recalled

```
COLUMN parallel:   x · [W₁ | W₂]  =  [x W₁ | x W₂]
                   Replicate × Shard(1) → Shard(1)          no collective

ROW parallel:      [x₁ | x₂] · ⎡W₁⎤  =  x₁W₁ + x₂W₂
                                ⎣W₂⎦
                   Shard(1) × Shard(0) → Partial(sum)       all-reduce
```

The design rule follows immediately:

> **Chain a column-parallel layer into a row-parallel layer.** The intermediate
> activation is `Shard`, never gathered; the pair costs exactly one all-reduce.

Everything below is that rule applied twice per transformer block.

## 2. The FFN

```
FFN(x) = W₂ ( silu(x W₁) ⊙ (x W₃) )
```

Shard `W₁` and `W₃` by columns, `W₂` by rows:

```
rank p holds:   W₁^(p) ∈ R^{d × d_ff/P}
                W₃^(p) ∈ R^{d × d_ff/P}
                W₂^(p) ∈ R^{d_ff/P × d}
```

Trace it, with `x` replicated:

```
x W₁^(p)                       → [B, S, d_ff/P]     local, no comm
x W₃^(p)                       → [B, S, d_ff/P]     local, no comm
silu(·) ⊙ (·)                  → [B, S, d_ff/P]     local, no comm     ← §2.1
(·) W₂^(p)                     → [B, S, d]          Partial(sum)
all-reduce                     → [B, S, d]          Replicate
```

**One all-reduce per FFN.** The `d_ff`-wide activation — the largest tensor in
the block — is never materialized in full on any rank.

### 2.1 Why the nonlinearity survives — the load-bearing detail

The step marked `← §2.1` is the only non-trivial one. It works because `silu` and
`⊙` are **elementwise**:

```
silu( [a | b] ) = [silu(a) | silu(b)]
[a | b] ⊙ [c | d] = [a⊙c | b⊙d]
```

Elementwise functions **commute with column sharding**, because element `j` of
the output depends only on element `j` of the input. Rank `p` can apply `silu` to
its own slice and get exactly the slice of the true result.

Now consider what would break it. Put any operation that **mixes across the
hidden dimension** between `W₁` and `W₂`:

```
softmax over d_ff   →  needs Σ over all d_ff  →  must gather (or all-reduce twice)
LayerNorm over d_ff →  needs mean and var over all d_ff  →  same
```

Either would force an extra collective *inside* the FFN, doubling the
communication. The reason TP is cheap for transformers is that **the only
row-mixing operations in a transformer block are the matmuls themselves**, and
those are precisely what the block decomposition handles.

Write it as the general condition:

> **A column-parallel → row-parallel chain is valid iff every operation between
> them is elementwise along the sharded dimension.**

You will apply this test again to attention (§3), to MoE (topic 27), and to any
new architecture you try to parallelize.

## 3. Attention: shard by head

Attention has a natural independent axis: **the head**.

```
out = W_O · concat_i [ Attn(Q^(i), K^(i), V^(i)) ]
```

Head `i`'s computation touches no other head. So:

```
W_Q, W_K, W_V   → COLUMN parallel   (rank p gets heads  p·h/P … (p+1)·h/P)
W_O             → ROW parallel      (matching the concatenation)
```

Trace it:

```
x (replicated)  →  Q^(p), K^(p), V^(p)     [B, h/P, S, d_h]     local
                →  softmax(QKᵀ) V           [B, h/P, S, d_h]     LOCAL — no comm!
                →  concat heads             [B, S, (h/P)·d_h]    local
                →  W_O^(p)                  [B, S, d]            Partial(sum)
                →  all-reduce               [B, S, d]            Replicate
```

**One all-reduce per attention block**, and — this is the good part — **zero
communication inside the attention computation itself**. The `S × S` score
matrix, the softmax, the `AV` product: all entirely local, per head.

Apply the §2.1 test: between the column-parallel `W_Q/W_K/W_V` and the
row-parallel `W_O`, is everything elementwise along the sharded dimension? The
sharded dimension is the *head*, and softmax runs over the *sequence*, within a
head. So yes — softmax is "elementwise in the head index". That is why attention
tensor-parallelizes at all.

And hence the assertion in `parallelize_deepseekv3`:

```python
if n_heads % tp != 0:
    raise ValueError(f"tensor_parallel_degree ({tp}) must divide n_heads ({n_heads}).")
```

With `h = 16`, `P_tp ∈ {1, 2, 4, 8, 16}`. This is a hard ceiling on TP degree,
and it is why models intended for large TP are built with many heads.

### 3.1 MLA has a path TP cannot shard

MLA breaks the clean picture, and the reason is interesting. From topic 09, the
per-token quantities are:

```
wq      : d → h·(128+64)        PER HEAD          → shardable
wkv_b   : d_c → h·(128+128)     PER HEAD          → shardable
wkv_a   : d → (d_c + 64)        SHARED by all heads → NOT shardable
kv_norm : d_c                   SHARED             → NOT shardable
wo      : h·128 → d             per head (rows)   → row parallel
```

`wkv_a` produces the **latent** and the **shared RoPE key** — quantities that by
construction do not have a head dimension (that was the whole point of topic 09
§5). There is no head axis to shard along.

Options:

- **Shard `d_c` across ranks.** Then every rank needs the *whole* latent to
  reconstruct its own heads' keys, so you would have to all-gather it. Extra
  collective, no benefit.
- **Replicate `wkv_a` and `kv_norm`.** Every rank computes the same latent
  redundantly. Costs `P_tp ×` the FLOPs for that one small matrix, and zero
  communication.

Replication wins easily: `wkv_a` is `2048 × 576 = 1.18M` parameters, 8.6% of the
attention block (topic 02 §2.3). Paying 8× redundant work on 8.6% of a block is
cheaper than one extra all-gather per layer.

Hence the `NoParallel` entries in the plan (topic 22):

```python
"attention.wkv_a":   NoParallel(use_local_output=False),  # not per-head
"attention.wkv_b":   ColwiseParallel(use_local_output=False),  # per-head
"attention.kv_norm": NoParallel(use_local_output=False),
```

The general lesson: **tensor parallelism needs an independent axis, and not
every tensor has one.** When it does not, replicate the small thing rather than
contorting the plan.

## 4. Communication cost, and the hard limit on `P_tp`

Per transformer block:

```
forward:   2 all-reduces (attention + FFN)
backward:  2 all-reduces
           ─────────────
           4 all-reduces per block per micro-batch
```

Volume per all-reduce, on `[B, S, d]` in bf16 (topic 19 §2.3):

```
V   = B · S · d · 2 bytes
AR  = 2 V (P−1)/P
```

For `B = 1, S = 16384, d = 2048, P_tp = 8`:

```
V  = 67 MB
AR = 2 · 67 · 7/8 = 117 MB
per micro-batch:  4 × 27 blocks × 117 MB = 12.7 GB
```

Now the two links:

```
NVLink   900 GB/s  →  12.7 GB / 900  = 14 ms per micro-batch
InfiniBand 25 GB/s →  12.7 GB / 25   = 508 ms per micro-batch
```

A 36× difference, and — unlike DP or PP — **these collectives are on the critical
path of every single layer.** There is nothing to overlap them with: the next
layer's input *is* the all-reduce's output.

```
┌───────────────────────────────────────────────────────────────┐
│  Keep P_tp within one NVLink domain (typically ≤ 8).          │
│  This is not a guideline; crossing the node boundary with TP  │
│  typically costs 3–5× end-to-end.                             │
└───────────────────────────────────────────────────────────────┘
```

## 5. Sequence parallelism: a free `P×` on activation memory

Here is a real inefficiency in what we have built so far.

Between the FFN's all-reduce and the next block's attention sit the **norms**
(and dropout, if any). Those operate on the full replicated `[B, S, d]`
activation, and **every rank holds an identical copy**:

```
TP region      →  activations Shard(head or d_ff)   → 1/P per rank  ✓
norm region    →  activations Replicate()           → FULL on every rank  ✗
```

The norm regions are pure duplication — `P_tp ×` the necessary activation
memory, for the residual stream, which is the tensor you keep most copies of.

**The fix.** Norms are elementwise along the *sequence* dimension (RMSNorm
normalizes over `d`, independently per token). So shard the sequence there:

```
        ┌─ Shard(1) ─┐                    ┌─ Shard(1) ─┐
norm ──►│ each rank  │──all-gather──► TP ─│  Partial   │──reduce-scatter──► norm
        │ has S/P    │                     │            │                   Shard(1)
        └────────────┘                    └────────────┘
```

Transitions, straight from topic 19 §3's table:

```
entering TP:  Shard(1) → Replicate()     all-gather
leaving  TP:  Partial  → Shard(1)        reduce-scatter
```

### 5.1 It costs exactly nothing

The striking part. Compare the bytes:

```
without SP:   1 all-reduce            = 2V(P−1)/P
with SP:      1 all-gather            =  V(P−1)/P
            + 1 reduce-scatter        =  V(P−1)/P
                                       ─────────────
                                      = 2V(P−1)/P      ← IDENTICAL
```

Because an all-reduce **is** a reduce-scatter followed by an all-gather
(topic 19 §2.2), splitting it into its two halves and putting the norm region
*between* them moves zero extra bytes.

```
┌──────────────────────────────────────────────────────────────┐
│  Sequence parallelism reduces activation memory in the norm  │
│  regions by P_tp for exactly zero additional communication.  │
│  Always enable it with TP.                                   │
└──────────────────────────────────────────────────────────────┘
```

That is Megatron-LM's sequence-parallelism result, and it is one of the cleanest
wins in the whole field: you were already paying for both halves of the
all-reduce, you were just performing them back to back and throwing away the
sharded intermediate.

Note it also explains topic 18 §5's `seq_len % tp == 0` requirement — the
sequence must divide across TP ranks.

## 6. Loss parallelism: the `[B, S, V]` problem

The output projection produces logits of shape `[B, S, V]`. With
`V = 102400, S = 16384, B = 1`, in fp32 (cross-entropy needs fp32 — topic 13 §4):

```
1 × 16384 × 102400 × 4 bytes = 6.7 GB
```

Per micro-batch, on the last stage. `output` is column-parallel (it shards `V`),
so each rank naturally has `[B, S, V/P]` — 840 MB at `P = 8`. Gathering it to
compute cross-entropy would rebuild the whole 6.7 GB, plus its gradient.

**Do not gather.** Cross-entropy over a vocab-sharded logit tensor needs only two
small reductions:

```
CE = logsumexp_v(z) − z_target

logsumexp needs:   m = max_v z              →  all-reduce(MAX) of [B, S]
                   Σ_v exp(z − m)           →  all-reduce(SUM) of [B, S]
z_target:          gather one value per token from whichever rank owns it
```

Both reductions are on `[B, S]` — **`V/1` times smaller** than the logits:

```
[B, S] in fp32 = 16384 × 4 = 65 KB     vs     6.7 GB
```

So the plan keeps the logits sharded:

```python
"output": ColwiseParallel(
    input_layouts=Shard(1),
    output_layouts=Shard(-1) if loss_parallel else Replicate(),
    use_local_output=not loss_parallel,
),
```

`Shard(-1)` = sharded on the vocabulary dimension, and `use_local_output=False`
keeps it a DTensor so the loss function can dispatch the distributed
cross-entropy.

This is the largest single memory saving TP offers, and it is why
`disable_loss_parallel` exists as an explicit opt-out rather than a default.

## 7. TP versus FSDP — when to use which

| | FSDP | TP |
|---|---|---|
| parameters | sharded, **gathered to compute** | sharded, **never gathered** |
| activations | full (per micro-batch) | sharded (`1/P` in TP regions) |
| collective | 2 AG + 1 RS **per module** | 2 AR **per block**, on the critical path |
| overlappable | **yes** (prefetch: order is known) | **barely** — the next op needs the result |
| tolerates slow links | reasonably | **no** |
| degree limit | any | `n_heads` (and node size) |
| single layer too big? | **does not help** | **fixes it** |

```
Use FSDP as the default and as much of it as possible.
Add TP only when: a single layer does not fit, or the activation
memory (not the parameters) is the binding constraint, or you need
loss parallelism for a huge vocabulary.
Keep P_tp ≤ node size, always with sequence parallelism.
```

## 8. Verify it

`minimal_examples/tp_swiglu.py` and `tp_attention.py` do exactly this. The
pattern to internalize — and to reuse whenever you write a new parallel plan:

```python
# 1. run the module unsharded on every rank, on identical input  → reference
# 2. shard the weights by hand per §2/§3
# 3. run the sharded computation, ending in an all-reduce
# 4. assert the two agree to floating-point noise
```

```bash
torchrun --standalone --nproc-per-node=2 ./minimal_examples/tp_swiglu.py
torchrun --standalone --nproc-per-node=4 ./minimal_examples/tp_attention.py
```

(If `torchrun`'s rendezvous fails, use the `mp.spawn` + `file://` harness from
topic 01 §6. The examples' README also documents the static-rendezvous
workaround for IPv6 issues.)

Any parallel plan you write should come with this test. A TP bug does not crash
— shapes stay correct and the all-reduce succeeds — it just computes a different
function.

## 9. Check yourself

1. State both TP primitives with their DTensor placements and say which needs a
   collective.
2. Why chain column-parallel into row-parallel rather than the reverse?
3. Give the exact condition under which the chain is valid, and two operations
   that would violate it.
4. Why does `silu(x W₁) ⊙ (x W₃)` need no communication when `W₁` and `W₃` are
   column-sharded?
5. Attention shards by head. Why is the entire `S × S` score computation local?
6. Why must `n_heads % P_tp == 0`, and what does that imply for a 16-head model?
7. Why is MLA's `wkv_a` `NoParallel`? Give both the algebraic reason and the cost
   comparison.
8. Compute TP's per-micro-batch communication for `B=2, S=8192, d=4096, L=32,
   P_tp=8` and the time on NVLink versus InfiniBand.
9. Why can TP's collectives not be overlapped the way FSDP's can?
10. Show that sequence parallelism moves exactly as many bytes as plain TP.
    What does it buy, and by what factor?
11. Which two transitions does SP introduce, and which collective is each?
12. Compute the logits tensor size for `B=1, S=16384, V=102400` in fp32. What do
    the two loss-parallel reductions cost instead?
13. Give three situations that justify adding TP on top of FSDP.

## Next

→ [22 — Coding tensor parallelism](22-coding-tensor-parallelism.md)
