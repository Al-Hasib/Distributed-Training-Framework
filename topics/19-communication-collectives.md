# 19 — Distributed Communication Collectives

> Video: 09:48:52 — Distributed Communication Collectives

## Why this exists

Topic 11 proved *which* collective each cut needs. This topic is the reference
for **what each one does, what it costs, and how to reason about it**. It is the
chapter you will come back to.

Three things to take away:

1. The **semantics** of eight primitives, precisely enough to predict output
   values.
2. The **cost model** — `α + β` — which tells you the ratio between an all-reduce
   and an all-gather without benchmarking. (The answer is 2, and it matters.)
3. The **DTensor redistribute table** — a `3 × 3` matrix that says which
   collective converts any placement into any other. Learn this table and you
   can read any parallelism plan.

## 1. The primitives

`P` ranks. Rank `r` starts with `x_r`. `V` = size of the full logical tensor.

### Point-to-point

```
send(t, dst) / recv(t, src)
```

Two ranks, no reduction. **This is pipeline parallelism's only collective**
(topic 14 §1). Everything else below involves a group.

### Broadcast

```
before:   r0: [a]   r1: [·]   r2: [·]   r3: [·]
after:    r0: [a]   r1: [a]   r2: [a]   r3: [a]
```

One source, all receive. Adjoint = reduce (topic 11 §2.1). Used to distribute
the seed (topic 13 §7) and to materialize replicated parameters.

### Reduce / All-reduce

```
all_reduce(SUM):
before:   r0:[0,0,0,0]  r1:[1,1,1,1]  r2:[2,2,2,2]  r3:[3,3,3,3]
after:    every rank:  [6,6,6,6]
```

`reduce` leaves the answer on one rank; `all_reduce` on all. Ops: `SUM`, `AVG`,
`MAX`, `MIN`, `PRODUCT`. **Self-adjoint** (topic 11 §2.2). This is DDP's
collective and tensor parallelism's.

### All-gather

```
before:   r0:[0,0,0,0]  r1:[1,1,1,1]  r2:[2,2,2,2]  r3:[3,3,3,3]
after:    every rank:  [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]
```

Concatenate everyone's shard, everywhere. **No reduction** — the output is `P×`
larger than the input. Adjoint = reduce-scatter. This is FSDP's forward
collective and tensor parallelism's `Shard → Replicate`.

### Reduce-scatter

```
y_r = arange(16) + 100·r          (each rank holds the full 16)
after (P=4):  r0:[600,604,608,612]  r1:[616,620,624,628]  ...
```

Sum across ranks, then keep only your own slice. Verify: element `i` of the sum
is `Σ_r (i + 100r) = 4i + 600`, and rank 0 keeps `i = 0..3` → `[600, 604, 608,
612]`. ✓

Adjoint = all-gather. This is FSDP's backward collective, and TP's
`Partial → Shard`.

### Scatter / Gather

The one-rank versions of the above (`scatter` = the inverse of `gather`). Rarely
used in training — the all- variants dominate because every rank needs the
result.

### All-to-all

```
rank r sends element q to rank q:   in_r[q] = q + 10r

after:   r0: [0, 10, 20, 30]      ← element 0 from every rank
         r1: [1, 11, 21, 31]      ← element 1 from every rank
```

A **distributed transpose**. Every rank sends a distinct chunk to every other
rank. Its own adjoint with the split/concat dims swapped (topic 11 §2.5), and it
is the collective of expert parallelism (topic 29) and of `Shard(i) → Shard(j)`.

### Barrier

Synchronization only, no data. Almost never needed in training — the collectives
already synchronize. Its main legitimate use is timing measurement; sprinkling
barriers to "fix" a hang hides the real bug.

## 2. The cost model

### 2.1 `α + β`

```
T  =  α · (number of communication steps)  +  (bytes moved) / B
```

- **`α`** — per-message latency: ~1–5 µs intra-node (NVLink), ~10–50 µs
  inter-node (InfiniBand).
- **`B`** — bandwidth: ~900 GB/s NVLink, ~25 GB/s InfiniBand.

Small messages are `α`-dominated, large messages `B`-dominated. The crossover is
around `α·B` bytes ≈ 25 KB inter-node, so almost everything in training is
bandwidth-bound — which is why bucketing (topic 12 §3.2) is about amortizing `α`
for the small tensors only.

### 2.2 Ring algorithms

The ring is the bandwidth-optimal pattern: arrange ranks in a circle, each
sending only to its neighbour, so all `P` links run concurrently.

**Ring all-gather.** `P − 1` steps; in each, every rank forwards a `V/P`-sized
chunk. Bytes received per rank:

```
V · (P − 1) / P        in  (P − 1)  steps
```

**Ring reduce-scatter.** The mirror image: `P − 1` steps, same volume.

```
V · (P − 1) / P        in  (P − 1)  steps
```

**Ring all-reduce = reduce-scatter, then all-gather.**

```
2V · (P − 1) / P       in  2(P − 1)  steps
```

Verified in code:

```
RS+AG == AR      True  ([6.0, 6.0, 6.0, 6.0] vs [6.0, 6.0, 6.0, 6.0])
```

**All-to-all.** `P − 1` steps, `V(P−1)/P` bytes sent per rank — the same *volume*
as an all-gather, but every byte is distinct data rather than a copy, so it
cannot be tree-optimized and is more sensitive to network topology.

### 2.3 The table, and the one number to remember

| Collective | steps | bytes per rank | vs all-reduce |
|------------|-------|----------------|---------------|
| broadcast | `O(log P)` | `V` | 0.5× |
| all-gather | `P − 1` | `V(P−1)/P` | **0.5×** |
| reduce-scatter | `P − 1` | `V(P−1)/P` | **0.5×** |
| **all-reduce** | `2(P − 1)` | `2V(P−1)/P` | 1× |
| all-to-all | `P − 1` | `V(P−1)/P` | 0.5× |

```
┌──────────────────────────────────────────────────────────────┐
│  An all-reduce costs exactly TWICE an all-gather or a        │
│  reduce-scatter, because it IS one of each.                  │
└──────────────────────────────────────────────────────────────┘
```

That single fact has a large consequence. Compare DDP with FSDP per step:

```
DDP  :  1 all-reduce of gradients          =  2V(P−1)/P
FSDP :  1 all-gather of params (forward)   =   V(P−1)/P
        1 all-gather of params (backward)  =   V(P−1)/P
        1 reduce-scatter of grads          =   V(P−1)/P
                                            = 3V(P−1)/P
```

**FSDP moves 1.5× the bytes of DDP** and in exchange divides parameter,
gradient, and optimizer memory by `P`. That is the trade, quantified — and it is
why FSDP is the default at scale while DDP survives for models that fit
(topic 20 §5).

Also note every volume has the factor `(P−1)/P → 1`. **Collective volume is
essentially independent of `P`.** Only the step count grows, so at large `P` you
pay in latency, not bandwidth. This is the property that lets data parallelism
scale to thousands of ranks.

## 3. The DTensor redistribute table

`DTensor` = a local tensor + a `DeviceMesh` + a list of `Placement`s (one per
mesh dimension). The three placements (topic 11, appendix-notation):

```
Shard(i)       the tensor is split along dim i; each rank holds 1/P
Replicate()    every rank holds the full tensor
Partial(sum)   every rank holds a full-shaped tensor; the truth is their sum
```

`redistribute()` converts between them, and **the collective is determined by the
(from, to) pair**:

```
        to →    Replicate()      Shard(j)              Partial(sum)
from ↓
Replicate()     —                local slice           divide by P
                                 (no comm)             (or zero on all but one)

Shard(i)        all-gather       i == j: —             (rare; not a natural op)
                                 i ≠ j: all-to-all

Partial(sum)    all-reduce       reduce-scatter        —
```

Five entries do real work, and each one is a technique:

| Transition | Collective | Where it appears |
|------------|------------|------------------|
| `Shard(i) → Replicate()` | all-gather | FSDP forward; TP `SequenceParallel → Colwise` |
| `Partial → Replicate()` | all-reduce | TP `RowwiseParallel` output |
| `Partial → Shard(i)` | reduce-scatter | FSDP gradients; TP + sequence parallelism |
| `Shard(i) → Shard(j)` | all-to-all | MoE dispatch; head↔sequence resharding |
| `Replicate() → Shard(i)` | **none** — local slice | free; just take your piece |

Two rows deserve comment:

- **`Replicate() → Shard(i)` is free.** Every rank already has the data; it just
  drops what it does not own. In the backward pass this becomes
  `Shard(i) → Replicate()`, i.e. an all-gather (topic 11 §2.4 — sharding is
  self-adjoint only when *both* sides are sharded). So a free forward can imply
  a paid backward. Always check both directions.
- **`Partial → Shard` is a reduce-scatter, not an all-reduce followed by a
  slice.** Same result, half the bytes. When you see a plan produce `Partial`
  output and the next layer wants `Shard`, that is a saved all-gather — the whole
  design goal of sequence parallelism (topic 22).

## 4. Overlap: making communication free

A collective that runs while the GPU computes costs nothing. Three mechanisms.

### 4.1 Async collectives

```python
work = dist.all_reduce(t, async_op=True)   # returns immediately
... do unrelated compute ...
work.wait()                                # block until complete
```

The reference examples use exactly this, with a helper that refuses to let a
`None` slip through:

```python
def require_work(work: dist.Work | None, name: str) -> dist.Work: ...
```

`async_op=True` returns `None` in some code paths (notably inside functional
collectives), and silently skipping a `wait()` gives you a race: the tensor is
read before the collective lands, producing intermittently wrong numbers. The
assertion turns that into a crash.

### 4.2 Separate streams

On CUDA, NCCL collectives run on their own stream. The dependency between compute
and communication is expressed with events, not by blocking the host. This is
what makes the topic 12 §3.1 overlap diagram real rather than aspirational — but
it also means **you must not read a tensor on the compute stream while a
collective on it is in flight**, and PyTorch's `wait()` inserts the event that
prevents that.

### 4.3 Prefetch

Overlap requires knowing the *future*. FSDP exploits the fact that layer order
is known: while computing layer `i`, all-gather layer `i+1`'s parameters. The
collective for layer `i+1` is issued before it is needed, so its latency is
hidden entirely.

```
compute:  [layer i] [layer i+1] [layer i+2]
comm:     [AG i+1 ] [AG i+2   ] [AG i+3   ]
```

This is why FSDP's 1.5× volume (§2.3) often costs *nothing* in wall-clock. The
number that matters is not the volume, it is the volume that fails to overlap.

## 5. Practical hazards

### 5.1 Collective order must match on every rank

Collectives are matched by **call order** within a group, not by name or tag. If
rank 0 does `all_reduce(A); all_gather(B)` and rank 1 does
`all_gather(B); all_reduce(A)`, you get a hang or — worse, on some backends —
data corruption.

The classic way to cause this is a rank-dependent branch:

```python
if loss > threshold:              # ← BUG: threshold reached on some ranks only
    dist.all_reduce(x)
```

Any conditional around a collective must have a condition that is **provably
identical on all ranks**: derived from the step number, the config, or a value
that was itself all-reduced. This is why topic 13 §6 gates metrics on
`should_log(self.step)` — a function of the step, hence identical everywhere.

### 5.2 Debugging a hang

```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL     # validates shapes/order, reports mismatches
TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # turns hangs into timeouts with a message
NCCL_DEBUG=INFO                    # topology and algorithm selection
```

Then find the rank that is *not* waiting in a collective — that is the one that
took a different path. `py-spy dump --pid <pid>` on each process gives you this
in seconds.

### 5.3 Reductions are not deterministic

Floating-point addition is not associative, and a ring all-reduce's summation
order depends on rank count and on the algorithm NCCL selects (which can vary
with message size). So the *same* job on the same data can produce slightly
different gradients across runs, and bit-exact reproducibility requires
`torch.use_deterministic_algorithms(True)` plus a fixed algorithm — and costs
performance. Topic 13 §7 exposes this as the `deterministic` flag.

The practical implication: do not chase `1e-6` differences between a 4-rank and
an 8-rank run. Do chase `1e-2`.

### 5.4 dtype and shape must match exactly

Every rank must pass the same shape, dtype, and device. A bf16 tensor on one rank
and fp32 on another is an error at best and a garbage reinterpretation at worst.
Under mixed precision, be explicit about which dtype a collective runs in —
gradient all-reduces in bf16 lose precision that fp32 reduction would keep, which
is why FSDP's `MixedPrecisionPolicy` has a separate `reduce_dtype`.

### 5.5 API churn

Running the §1 examples today prints:

```
FutureWarning: `torch.distributed.reduce_scatter_tensor` is deprecated.
Please use `torch.distributed.reduce_scatter_single` instead.
```

The collective API is still moving (the `_tensor`/`_single` family, the
functional collectives in `torch.distributed._functional_collectives`, and
DTensor's `redistribute` are three generations of the same idea). Prefer
**DTensor `redistribute`** in model code: it picks the collective for you from
the table in §3, and it composes across mesh dimensions. Reach for raw
collectives only for things DTensor does not model — pipeline send/recv, metric
reductions, the seed broadcast.

## 6. A runnable reference

```python
# coll.py — every primitive, on CPU with gloo.
import os, tempfile, torch, torch.distributed as dist, torch.multiprocessing as mp

def worker(rank, world, initfile):
    dist.init_process_group("gloo", init_method=f"file:///{initfile}",
                            rank=rank, world_size=world)
    P = world
    def show(name, t):
        if rank == 0: print(f"{name:<16} rank0 -> {t.tolist()}")

    x = torch.tensor([float(rank)] * 4)                     # rank r holds [r,r,r,r]

    a = x.clone(); dist.all_reduce(a, op=dist.ReduceOp.SUM); show("all_reduce sum", a)

    g = [torch.zeros(4) for _ in range(P)]
    dist.all_gather(g, x); show("all_gather", torch.cat(g))

    y = torch.arange(4 * P, dtype=torch.float) + rank * 100
    rs = torch.zeros(4); dist.reduce_scatter_tensor(rs, y.contiguous())
    show("reduce_scatter", rs)

    b = x.clone(); dist.broadcast(b, src=0); show("broadcast src0", b)

    a2a_in = torch.arange(P, dtype=torch.float) + rank * 10   # element q → rank q
    a2a_out = torch.zeros(P)
    dist.all_to_all_single(a2a_out, a2a_in.contiguous()); show("all_to_all", a2a_out)

    # all_reduce == reduce_scatter followed by all_gather
    rs2 = torch.zeros(4); dist.reduce_scatter_tensor(rs2, x.repeat(P).contiguous())
    gg = [torch.zeros(4) for _ in range(P)]; dist.all_gather(gg, rs2)
    ref = x.clone(); dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"RS+AG == AR      {torch.allclose(torch.cat(gg)[:4], ref)}")
    dist.destroy_process_group()

if __name__ == "__main__":
    f = os.path.join(tempfile.gettempdir(), "pg_coll")
    if os.path.exists(f): os.remove(f)
    mp.spawn(worker, args=(4, f.replace("\\", "/")), nprocs=4, join=True)
```

Actual output (`P = 4`):

```
all_reduce sum   rank0 -> [6.0, 6.0, 6.0, 6.0]
all_gather       rank0 -> [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3]
reduce_scatter   rank0 -> [600.0, 604.0, 608.0, 612.0]
broadcast src0   rank0 -> [0.0, 0.0, 0.0, 0.0]
all_to_all       rank0 -> [0.0, 10.0, 20.0, 30.0]
all_to_all       rank1 -> [1.0, 11.0, 21.0, 31.0]
RS+AG == AR      True
```

Predict each line from §1 before reading it. If `reduce_scatter` and
`all_to_all` surprise you, re-derive them — those two are where the confusion
lives.

## 7. Check yourself

1. Give the exact output of `all_gather` and of `reduce_scatter` for the inputs
   in §6, and derive `[600, 604, 608, 612]`.
2. Which collective is a distributed transpose, and which parallelism uses it?
3. Write the `α + β` model. At what message size does inter-node communication
   stop being latency-bound?
4. Show that a ring all-reduce moves `2V(P−1)/P` bytes, and why that is exactly
   twice an all-gather.
5. Compute DDP's and FSDP's per-step communication volume. What does FSDP buy for
   the extra 50%?
6. Collective volume is nearly independent of `P`. What *does* grow with `P`, and
   what does that imply at 4096 ranks?
7. Fill in the full `3 × 3` DTensor redistribute table from memory.
8. Which transition is free in the forward pass but implies a collective in the
   backward pass? Why?
9. Why is `Partial → Shard` a reduce-scatter rather than an all-reduce plus a
   slice? How many bytes does that save?
10. Give a rank-dependent conditional that would hang, and the rule for making
    conditionals around collectives safe.
11. Why is a 4-rank run not bit-identical to an 8-rank run? What size of
    discrepancy should worry you?
12. Why prefer DTensor `redistribute` over raw collectives in model code, and
    name three places raw collectives are still the right tool.

## Next

→ [20 — Implementing device meshes, DDP and FSDP](20-implementing-meshes-ddp-fsdp.md)
