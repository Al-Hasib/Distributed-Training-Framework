# 23 — Context Parallelism and Ring Attention

> Video: 14:47:43 — Context Parallelism and Ring Attention
> Code: `minimal_examples/ring_attention.py`, `minimal_examples/ring_attention_intro.ipynb`,
> `torchfeather/distributed/utils.py` (`create_context_parallel_ctx`)

## Why this exists

Three of the four cuts are done: layers (PP), batch (DP/FSDP), matrices (TP).
One axis remains, and at long context it is the one that matters.

From topic 02: at `S = 16384` **67% of training FLOPs are attention scores**, and
activation memory is `O(L·B·S·d)` — 25 GB for a single sequence in our config.
At `S = 131072` neither the compute nor the memory fits, no matter how many
parameters you shard: the problem scales with `S`, and `B = 1` already.

So cut the sequence. Rank `r` holds `S/P_cp` tokens and everything about them.

The difficulty is that **attention is the one operation that couples positions**.
Solving it requires FlashAttention's online softmax, distributed around a ring —
and then a load-balancing trick to deal with causal masking, which is where the
mysterious `2` in `seq_len_divisor = tp · (cp · 2)` comes from.

## 1. What is easy, and what is not

Shard the sequence. Which operations still work?

| Operation | Couples positions? | Under `Shard(seq)` |
|-----------|--------------------|--------------------|
| embedding lookup | no | ✓ local |
| RMSNorm | no (normalizes over `d`) | ✓ local |
| all linear projections | no (applied per token) | ✓ local |
| SwiGLU, residual add | no | ✓ local |
| MoE routing + experts | no (per token) | ✓ local |
| **RoPE** | no, **but is position-dependent** | ⚠ needs *global* positions |
| **attention** | **yes** | ✗ needs remote `K`, `V` |
| cross-entropy loss | no | ✓ local |

Almost everything is free — a transformer is overwhelmingly per-token. Two things
need work, and they need different kinds of work: RoPE needs the right
*metadata* (§5), attention needs actual *communication* (§2–4).

## 2. The attention problem

Query at global position `m` must attend to all keys at positions `≤ m`. With the
sequence sharded, most of those keys are on other ranks.

```
P_cp = 4, S = 8, so 2 positions per rank

rank 0: positions 0,1     rank 1: positions 2,3
rank 2: positions 4,5     rank 3: positions 6,7

rank 3's query at position 7 needs keys 0…7 — three quarters of them remote.
```

Two ways out:

**(a) All-gather `K` and `V`.** Simple, and self-defeating: every rank then holds
the full `[B, h, S, d_h]` key and value tensors, which is the `O(S)` memory we
were trying to escape. It also all-gathers per layer.

**(b) Ring.** Keep `Q` fixed and **rotate `K`/`V` blocks around a ring of
ranks**, accumulating a partial attention result at each step. Peak memory stays
`O(S/P)` — you hold your own block plus one in flight.

Option (b) needs one thing to be true: **attention over a subset of keys must be
mergeable with attention over another subset.** It is not obvious that it is —
softmax has a denominator over *all* keys.

## 3. Online softmax: the mergeability that makes it work

### 3.1 The derivation

Softmax attention for one query over keys `j ∈ J`:

```
             Σ_{j∈J} exp(s_j − m) v_j
   out  =   ──────────────────────────  ,      m = max_{j∈J} s_j
             Σ_{j∈J} exp(s_j − m)
```

(the `m` subtraction is for numerical stability and cancels between numerator and
denominator).

Now suppose we have processed block `A` and hold three running values:

```
m   = max_{j∈A} s_j            the running maximum
ℓ   = Σ_{j∈A} exp(s_j − m)     the running denominator
o   = Σ_{j∈A} exp(s_j − m) v_j the running numerator
```

A new block `B` arrives with its own maximum `m_B = max_{j∈B} s_j`. Set
`m' = max(m, m_B)`. Every term in `o` and `ℓ` was scaled by `exp(−m)`; to
re-base it to `exp(−m')` multiply by

```
α = exp(m − m')
```

Then

```
ℓ' = α·ℓ + Σ_{j∈B} exp(s_j − m')
o' = α·o + Σ_{j∈B} exp(s_j − m') v_j
m' = max(m, m_B)
```

and after all blocks, `out = o / ℓ`.

```
┌───────────────────────────────────────────────────────────────┐
│  Attention is a MONOID over key blocks: partial results       │
│  (m, ℓ, o) merge associatively. So the blocks may be          │
│  processed in any order, on any device.                       │
└───────────────────────────────────────────────────────────────┘
```

This is FlashAttention's recurrence (topic 06 §6). FlashAttention runs it over
tiles in SRAM; ring attention runs the *same* recurrence over blocks that arrive
over the network. **Ring attention is FlashAttention with the block loop
distributed.**

### 3.2 It is exactly the code

```python
running_max    = torch.full(q.shape[:-1], float("-inf"))
running_sum    = torch.zeros_like(running_max)
running_values = torch.zeros_like(q)

for step in range(world_size):
    owner = (rank - step) % world_size
    if owner <= rank:
        scores = torch.matmul(q, current_k.transpose(-2, -1)) * scale
        if owner == rank:
            scores = apply_local_causal_mask(scores)

        block_max      = scores.amax(dim=-1)
        new_max        = torch.maximum(running_max, block_max)          # m'
        previous_scale = torch.exp(running_max - new_max)               # α
        probabilities  = torch.exp(scores - new_max.unsqueeze(-1))
        running_sum    = previous_scale * running_sum + probabilities.sum(dim=-1)
        running_values = (previous_scale.unsqueeze(-1) * running_values
                          + torch.matmul(probabilities, current_v))
        running_max    = new_max

    current_k, current_v = rotate_tensors((current_k, current_v), rank, world_size)

return running_values / running_sum.unsqueeze(-1)
```

Line for line: `new_max` = `m'`, `previous_scale` = `α`, and the two updates are
`ℓ'` and `o'`. Verified against `F.scaled_dot_product_attention`:

```
ring vs sdpa max err: 1.19e-07
```

fp32 rounding. **Ring attention computes exactly causal attention.**

### 3.3 The rotation

```python
def rotate_tensors(tensors, rank, world_size):
    next_rank, previous_rank = (rank + 1) % world_size, (rank - 1) % world_size
    received = tuple(torch.empty_like(t) for t in tensors)
    operations = []
    for tag, (send, recv) in enumerate(zip(tensors, received, strict=True)):
        operations.append(dist.P2POp(dist.isend, send, next_rank, tag=tag))
        operations.append(dist.P2POp(dist.irecv, recv, previous_rank, tag=tag))
    requests = dist.batch_isend_irecv(operations)
    for request in requests:
        request.wait()
    return received
```

Point-to-point, not a collective — like PP (topic 14 §1). Each rank sends to
`rank+1` and receives from `rank−1`, so all `P` links are busy simultaneously.

`batch_isend_irecv` posts all four operations at once. Posting them separately
would deadlock at `P = 2` (both ranks blocking on send with no one receiving) and
serialize at larger `P`. The `tag=tag` distinguishes the `K` message from the `V`
message.

And the real point: `isend`/`irecv` are **asynchronous**, so block `s+1`'s
rotation can overlap block `s`'s attention compute. Ring attention's
communication is nearly free at long context, because the `O((S/P)²)` compute per
step dominates the `O(S/P)` transfer.

## 4. Causal masking creates load imbalance — and the fix

### 4.1 The skip

```python
owner = (rank - step) % world_size
if owner <= rank:
    ... compute ...
else:
    pass    # causal_skip
```

If `owner > rank`, every key in that block is at a position *after* every one of
this rank's queries. Causal masking zeroes all of it. So skip the block entirely
— roughly halving total work.

### 4.2 The problem the skip creates

Count the blocks each rank actually computes:

```
rank 0: owner ∈ {0}             → 1 block
rank 1: owner ∈ {0,1}           → 2 blocks
rank 2: owner ∈ {0,1,2}         → 3 blocks
rank 3: owner ∈ {0,1,2,3}       → 4 blocks
```

Measured for several degrees:

```
P_cp=2: blocks/rank [1, 2]                    (max/min = 2.0×)
P_cp=4: blocks/rank [1, 2, 3, 4]              (max/min = 4.0×)
P_cp=8: blocks/rank [1, 2, 3, 4, 5, 6, 7, 8]  (max/min = 8.0×)
```

The ring is synchronous — every rank must finish before the rotation — so **the
whole thing runs at rank `P−1`'s speed**. Rank 0 idles for `(P−1)/P` of the time.
Utilization is `(P+1)/(2P) → 50%`.

Halving the work and then wasting half the machine is a net gain of zero. The
skip is worthless without balancing.

### 4.3 Zigzag sharding

**The fix.** Split the sequence into `2P` chunks instead of `P`, and give rank
`r` the pair

```
chunk r        (early in the sequence — little causal work)
chunk 2P−1−r   (late  in the sequence — much causal work)
```

```
P_cp = 4, so 8 chunks:

rank 0: chunks 0, 7        rank 1: chunks 1, 6
rank 2: chunks 2, 5        rank 3: chunks 3, 4
```

**Proof it balances.** Chunk `i` attends to chunks `0…i`, so its causal work is
proportional to `i+1`. Rank `r`'s total:

```
(r + 1) + ((2P − 1 − r) + 1)  =  r + 1 + 2P − r  =  2P + 1
```

**Independent of `r`.** Verified:

```
P_cp=2: zigzag [5, 5]
P_cp=4: zigzag [9, 9, 9, 9]
P_cp=8: zigzag [17, 17, 17, 17, 17, 17, 17, 17]
```

Perfectly constant. Utilization returns to ~100%, and the causal saving is kept.

```
┌───────────────────────────────────────────────────────────────┐
│  This is the "2" in  seq_len_divisor = tp · (cp · 2).         │
│  Load-balanced CP needs 2·P_cp chunks, so S must be           │
│  divisible by 2·P_cp.                                         │
└───────────────────────────────────────────────────────────────┘
```

Enabled by default in the reference code:

```python
_cp_options.enable_load_balance = cp_load_balance     # default True
```

and documented in `parallel_dims.seq_len_divisor`:

```python
# Context Parallel requires that seq_len be divisible by 2 * CP degree,
# when load balancing is enabled (by default).
```

## 5. RoPE under context parallelism — the bug you must not write

This is the §1 warning, and it is the practical trap of context parallelism.

Rank 2 holds *local* positions `0…S/P−1`, but its tokens are at *global*
positions determined by the sharding. Under zigzag they are not even contiguous
— rank 2 of 4 owns chunks 2 and 5, i.e. two disjoint global ranges.

RoPE rotates by `m·θ_i` where `m` is the **global** position (topic 03). Use the
local index and:

- every rank applies rotations as if its tokens started at position 0,
- relative distances *within* a chunk are still correct,
- relative distances *across* chunks are wrong,
- **the model trains anyway**, to a worse optimum, with no error message.

This is the "trains badly, silently" failure promised in topic 03 §Why.

### 5.1 How the framework prevents it

Elegantly: treat `freqs_cis` as **just another buffer to shard along the
sequence**.

```python
optional_context_parallel_ctx = dist_utils.create_context_parallel_ctx(
    cp_mesh=parallel_dims.get_mesh("cp"),
    cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
    cp_seq_dims=[1, 1]          + [0 for _ in model_parts],
    cp_no_restore_buffers={inputs, labels},
    cp_rotate_method=...,
    cp_load_balance=...,
)
```

Read it against §4.3:

- `inputs` and `labels` are `[B, S]` → sequence on dim **1**.
- `freqs_cis` is `[S, d/2]` → sequence on dim **0**.
- All are sharded with the **same** zigzag split, so each rank automatically
  holds precisely the frequency rows for the global positions it owns. No offset
  arithmetic anywhere.

That is why `freqs_cis` is threaded as a *function argument* through the model
(topic 04 §6.2) rather than read from a global — it has to be a shardable
tensor.

Two more details:

- **One entry per model part.** Under PP each stage has its own `freqs_cis`
  buffer (topic 17 §4.4), and each needs sharding.
- **`cp_no_restore_buffers={inputs, labels}`.** The context manager normally
  restores buffers to their unsharded state on exit — necessary for `freqs_cis`,
  which persists across steps. `inputs` and `labels` are discarded after the
  step, so restoring them is pure waste.

## 6. Cost

Per attention, per layer: `P_cp − 1` rotations, each carrying one rank's `K` and
`V` block.

```
V_rot = B · (S/P) · h · (d_qk + d_v) · 2 bytes
```

For `B=1, S=16384, P_cp=4, h=16, d_qk=192, d_v=128`:

```
V_rot = 1 · 4096 · 16 · 320 · 2 = 42 MB    per rotation
per layer: 3 rotations × 42 MB  = 126 MB
per micro-batch: × 27 layers    = 3.4 GB   (forward; ×~2 with backward)
```

Substantial, but:

- **Overlappable.** The compute per step is `O((S/P)² · d)` while the transfer is
  `O((S/P) · d)`. At long context, compute dominates by a factor of `S/P`, so
  with `isend`/`irecv` the transfer hides entirely. This is why CP works at long
  context and is a poor idea at short context.
- **Point-to-point**, so tolerant of moderate link quality — though `P_cp` inside
  a node is still preferable.

### 6.1 The MLA optimization

Topic 06 §5 promised this. What the ring carries depends on which attention form
runs:

```
reconstructed form (topic 07):  k [B,h,S,192] + v [B,h,S,128] = h·320 = 5120/token
latent form (topic 10):         k [B,1,S,576], v a slice of it =        576/token
                                                                   ─────────────
                                                                        8.9× less
```

The training path uses the reconstructed form, so it pays the full 5120. Rotating
the **latent** instead is a genuine available optimization — `d_c + d_h^R` is
head-independent and each rank could reconstruct locally after receiving it. It
requires the absorbed formulation inside the ring, which is why topic 10's
"inference-only" transformation is not purely an inference concern.

The general point is worth keeping: **an architecture that compresses its KV
representation compresses its context-parallel traffic by the same factor.** MLA
was designed for inference bandwidth and pays off again here.

### 6.2 `set_rotate_method`

```python
set_rotate_method(cp_rotate_method)
```

Two strategies:

- **`allgather`** — gather all `K`/`V` at once. More memory (`O(S)`), fewer
  larger messages. Can win at small `P_cp` or when latency dominates.
- **`alltoall`** — the true ring rotation. `O(S/P)` memory, `P−1` steps.

Ring/`alltoall` is the right default at scale; `allgather` is a fallback for
small degrees where the `O(S)` memory is affordable and the step latency is not.

## 7. Composition

**With FSDP.** Recall topic 18 §3.2: `fsdp = dp_shard × cp`. CP ranks hold
different tokens but identical parameters, so FSDP shards parameters across the
`cp` dimension too — **context parallelism gives you a free extra factor of
`P_cp` on parameter memory** on top of its activation saving. Hence:

```python
@property
def fsdp_enabled(self):
    return self.dp_shard_enabled or self.cp_enabled
```

**With TP.** The awkward case, and topic 06 §7.3's nine lines:

```python
tp_spec = None
if isinstance(q, DTensor):
    tp_spec = (q.device_mesh, q.placements)
    q, k, v = q.to_local(), k.to_local(), v.to_local()
...
if tp_spec is not None:
    out = DTensor.from_local(out, tp_spec[0], tp_spec[1], run_check=False)
```

`q` is sharded on **two** mesh dimensions at once: by head across `tp`, by
position across `cp`. The DTensor arriving at the attention wrapper is labelled
with its `tp` placement, and if CP's hook wraps that, the ring rotates blocks
among the **wrong ranks** — wrong numbers, no error. So: strip to local, let CP
do its thing on the `cp` mesh, re-wrap with the saved `tp` spec for the
`RowwiseParallel` `wo` downstream.

The secondary reason from the same comment: `F.pad` (the V-padding trick, topic
06 §7.2) has no registered DTensor sharding rule, so calling it on a DTensor
changes `V`'s placements and the CP handler then fails with *"inputs need to be
redistributed"*.

**With the loss.** CP ranks hold different tokens, so they contribute different
tokens to the loss — which is why the loss mesh is `batch × cp` (topic 18 §3.3).
And because each CP rank *loads* the full sequence before CP shards it,
`local_valid_tokens` must be divided by `cp` to avoid counting the same tokens
`P_cp` times (topic 13 §3).

## 8. When to use CP

```
Is S large enough that activations (not parameters) are the constraint?
    S ≥ 32k with B = 1 → yes, probably
    S ≤ 8k             → no; use FSDP/TP and a larger batch instead
Is the attention term a large share of FLOPs?  (S/(6d) from topic 02 §3.3)
Do you have S divisible by 2·P_cp·P_tp?
Can P_cp sit inside a node?
```

CP is the long-context tool. At short context its rotations do not overlap
(compute per step is too small) and it is strictly worse than a bigger batch.

## 9. Check yourself

1. List the transformer operations that are trivially local under `Shard(seq)`
   and the two that are not, saying what each of the two needs.
2. Why is all-gathering `K`/`V` self-defeating?
3. Derive the online-softmax merge rule for `(m, ℓ, o)`. Where does `α` come
   from?
4. In what sense is attention a monoid over key blocks, and why does that permit
   distribution?
5. Map `new_max`, `previous_scale`, `running_sum`, `running_values` onto the
   derivation's symbols.
6. Why `batch_isend_irecv` rather than separate `isend`/`irecv` calls? What
   happens at `P = 2` without it?
7. Why can the block `owner > rank` be skipped?
8. Compute blocks-per-rank for `P_cp = 8` without balancing. What is the
   utilization?
9. State zigzag sharding and **prove** it equalizes work. What constraint does it
   put on `S`?
10. Rank 2 of 4 under zigzag: which global position ranges does it own?
11. Explain the RoPE bug precisely, and how sharding `freqs_cis` as a CP buffer
    prevents it. Why is its `cp_seq_dim` `0` while `inputs`' is `1`?
12. Why is `freqs_cis` in `cp_buffers` but not in `cp_no_restore_buffers`?
13. Compute ring traffic per layer for `B=1, S=32768, P_cp=8, h=32, d_qk=128,
    d_v=128`. Why does it overlap well at that `S` and badly at `S = 2048`?
14. How much less would the ring carry under MLA's latent form, and why is that
    the same factor as the KV-cache saving?
15. Why does enabling CP also reduce *parameter* memory?
16. Under TP+CP, why must `q/k/v` be converted to local tensors? What is the
    observable symptom of skipping it?

## Next

→ [24 — Metrics, optimizers, schedulers and checkpointing](24-metrics-optimizers-schedulers-checkpointing.md)
