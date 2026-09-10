# 12 — Distributed Computation Graphs and DDP

> Video: 05:04:19 — Distributed Computation Graphs and DDP

## Why this exists

Topic 11 proved the rules. This topic applies them to the simplest possible
case — plain data parallelism — and in doing so establishes the *picture* we use
for every technique afterwards: **one logical computation graph, with each node
labelled by the device that owns it.**

DDP is worth its own chapter for two reasons. It is the only parallelism that
buys throughput (topic 01 §2), and its implementation contains the one systems
idea that every other parallelism reuses: **overlapping communication with
computation**.

## 1. The picture: one graph, many devices

Do not think of `P` separate programs. Think of **one** computation graph in
which each node carries a device annotation:

```
      device 0                    device 1
   ┌──────────────┐            ┌──────────────┐
   │ x₀ → f → h₀  │            │ x₁ → f → h₁  │       ← same f, different data
   │      ↑       │            │      ↑       │
   └──────┼───────┘            └──────┼───────┘
          └───────── θ (Replicate) ───┘               ← one logical node,
                          │                              materialized on both
                          ▼
                   Σ  gradients                       ← the adjoint of Replicate
```

Two kinds of edge exist:

- **Intra-device edges** — ordinary autograd, nothing to do.
- **Cross-device edges** — a *cut*. Topic 11 §5: every cut costs one forward
  communication and one backward communication, and the backward one is the
  adjoint of the forward one.

DDP has exactly one cut, and it is on the **parameters**, not the activations:

```
θ is Replicate()  ──(§2.1)──►  ∂L/∂θ is Partial(sum)  ──►  one all-reduce
x is Shard(batch) ──(§2.4)──►  ∂L/∂x is Shard(batch)  ──►  no communication
```

That is the entire derivation of DDP. There is no forward communication at all,
because the batch shards are genuinely independent and the parameters were
already replicated before the step began.

## 2. Why it is correct

From topic 11 §3, with `P` ranks each holding `n` of the `N = P·n` examples:

```
∂L/∂θ  =  (1/N) Σ_{i=1}^{N} ∂ℓ_i/∂θ
       =  (1/P) Σ_{p=1}^{P} [ (1/n) Σ_{i ∈ p} ∂ℓ_i/∂θ ]
          └────────────────────────────────────────────┘
                    average of the per-rank gradients
```

so a **mean** all-reduce of the local gradients yields the exact global-batch
gradient. Not an approximation: the same number a single enormous GPU would
produce, up to floating-point reassociation.

The equality in the second line requires all `n_p` equal. When they are not —
masking, packing, ragged batches, dropped MoE tokens — you must weight by token
counts (topic 11 §3.1). Topic 13 shows the reference implementation doing exactly
that.

### 2.1 What DDP does *not* change

```
per-rank memory:   still 18 bytes/param  (topic 01 §2)  — full model, full Adam state
per-rank FLOPs:    6N per local token
step time:         ~unchanged (if comms overlap)
throughput:        P×
```

DDP is a **throughput** technique only. It does nothing for memory — every rank
holds a complete replica. That is its limit, and it is what FSDP fixes (topic 20)
by sharding the replicated `θ` instead of replicating it.

## 3. The systems problem: hiding the all-reduce

The naive implementation is:

```python
loss.backward()                       # compute all gradients
for p in model.parameters():          # then communicate all of them
    dist.all_reduce(p.grad)
    p.grad /= world_size
optimizer.step()
```

Correct, and slow. Two independent problems.

### 3.1 Problem 1 — no overlap

Backward and all-reduce run in sequence, so step time is
`t_bwd + t_comm`. For a 7B model in bf16, the gradient volume is 14 GB per step;
on a 25 GB/s inter-node link, a ring all-reduce moves `2(P−1)/P · 14 GB ≈ 28 GB`
of traffic, taking ~1.1 s. If the backward pass itself takes 0.3 s, you are
spending 78% of the step waiting on the network.

**The fix uses a property of reverse-mode autodiff.** Gradients become ready
*progressively*, in reverse layer order: `∂L/∂W_L` is finished long before
`∂L/∂W_1`. So the all-reduce for the last layer can start while the first
layers are still computing.

```
time ──────────────────────────────────────────────────────►
compute:  [bwd L] [bwd L-1] [bwd L-2] … [bwd 2] [bwd 1]
comm:            [AR L]     [AR L-1]  …         [AR 2] [AR 1]
                  └── overlapped with compute ──┘        └─ exposed tail
```

Step time becomes `max(t_bwd, t_comm) + tail` instead of `t_bwd + t_comm`. This
is implemented with **autograd hooks**: a `post_accumulate_grad_hook` on each
parameter fires the moment that parameter's gradient is complete.

### 3.2 Problem 2 — too many small messages

A 7B model has thousands of parameter tensors, many of them small (every
RMSNorm weight is `d` values = 4 KB). Every collective pays a fixed latency
(a few microseconds intra-node, tens of microseconds inter-node) plus per-call
launch overhead. Thousands of 4 KB all-reduces is pure latency.

**The fix is bucketing.** Concatenate gradients into flat buffers of ~25 MB and
all-reduce one bucket at a time:

```
bucket 0: [ W_L.grad ‖ norm_L.grad ‖ W_{L−1}.grad ‖ … ]   25 MB → one all-reduce
bucket 1: [ … ]
```

Buckets are formed in **reverse parameter order**, because that is the order
gradients become ready — so a bucket fills up and fires as early as possible.
This is why DDP asks you not to change the parameter iteration order between
steps, and why `static_graph=True` (which lets it record the order once) is
faster.

The bucket size is a real trade-off:

```
small buckets  →  more overlap, more latency overhead
large buckets  →  less overhead, less overlap, more peak memory (the flat buffer)
```

25 MB is the empirical default. It is worth tuning on slow interconnects.

### 3.3 The bandwidth cost, for the record

A ring all-reduce of `V` bytes over `P` ranks moves `2V(P−1)/P` bytes per rank
and is bandwidth-optimal. Note it is **independent of `P`** to first order
(`2V(P−1)/P → 2V`), which is the property that makes data parallelism scale to
thousands of ranks. Compare with an all-gather, which moves `V(P−1)/P` — half as
much — a fact topic 20 exploits.

## 4. Gradient accumulation and `no_sync`

If you accumulate `M` micro-batches before stepping, all-reducing after every
micro-batch would be `M×` the necessary traffic. The gradients are additive
(topic 11 §3, corollary (a)), and summation commutes with the all-reduce, so you
can accumulate locally and communicate once:

```python
for m, (x, y) in enumerate(microbatches):
    last = (m == len(microbatches) - 1)
    ctx = contextlib.nullcontext() if last else model.no_sync()
    with ctx:                            # suppress the gradient hooks
        loss = criterion(model(x), y) / len(microbatches)
        loss.backward()
optimizer.step()
```

`no_sync()` disables the hooks so gradients merely accumulate into `.grad`. The
final micro-batch runs with hooks enabled, and the single all-reduce carries the
accumulated sum. This is a 1-line change with an `M×` reduction in communication
volume — and forgetting it is one of the most common causes of "why is my
multi-node run so slow".

## 5. What `torchfeather` actually uses, and why it is not `DistributedDataParallel`

Worth being clear about, because the reference framework has no `DDP` import
anywhere. It uses **DTensor** with `Replicate()` placements and FSDP2's
`fully_shard` (topic 20), even when the sharding degree is 1.

The reasons are about composition, not about DDP being wrong:

- **`nn.parallel.DistributedDataParallel` wraps the whole module** and owns the
  backward hooks. That makes it awkward to compose with pipeline parallelism
  (which needs to control when backward runs) and with tensor parallelism (which
  needs parameters to already be DTensors).
- **DTensor makes the placement explicit and composable.** A parameter can be
  `Shard(0)` on the `tp` mesh dim and `Replicate()` on the `dp_replicate` dim
  simultaneously. `DistributedDataParallel` has no vocabulary for that.
- **FSDP2 with `dp_shard = 1` degenerates to DDP.** Same communication pattern,
  same result, but reachable from the same code path as real sharding — so
  HSDP (topic 20 §6) is a config change rather than a different implementation.

So DDP in this course is best understood as **the degenerate case of the general
machinery**, and as the place where the overlap/bucketing ideas were invented.
Everything in §3 reappears in FSDP, where the collectives are prefetched rather
than merely overlapped.

## 6. Cost summary

| | DDP |
|---|---|
| forward communication | none |
| backward communication | 1 all-reduce of `∂L/∂θ`, `2V(P−1)/P` bytes |
| per-rank parameter memory | full model — **not reduced** |
| per-rank optimizer memory | full state — **not reduced** |
| per-rank activation memory | reduced by `P` (each rank has `1/P` of the batch) |
| throughput | `P×` if comms are hidden |
| scaling limit | model must fit on one device |
| composes with | everything (it is the outermost dimension of the mesh) |

## 7. Check yourself

1. DDP has exactly one cut in the computation graph. Where is it, and which two
   rules from topic 11 determine the forward and backward communication?
2. Why is there no forward communication in DDP?
3. Prove that mean-all-reducing local gradients gives the exact global-batch
   gradient. What assumption does the proof need, and when does it fail?
4. What property of reverse-mode autodiff makes overlap possible? In what order
   are buckets formed, and why?
5. Give the two independent reasons bucketing helps, and state the trade-off in
   bucket size.
6. A ring all-reduce of `V` bytes moves `2V(P−1)/P` per rank. Why does that make
   DDP scale to thousands of ranks?
7. You add 8-step gradient accumulation and your multi-node throughput barely
   improves. What did you forget, and what is the expected speedup from fixing
   it?
8. Give two reasons `torchfeather` uses DTensor `Replicate()` rather than
   `nn.parallel.DistributedDataParallel`.
9. DDP gives `P×` throughput and no memory relief. Which techniques in Part III
   fix the memory side, and why must they be applied *before* DDP is useful for
   large models?

## Next

→ [13 — Building the training loop](13-building-the-training-loop.md)
