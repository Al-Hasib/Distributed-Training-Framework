# 18 — Device Meshes and Combining PP with DP

> Video: 08:37:34 — Device Meshes and Combining PP with DP
> Code: `torchfeather/distributed/parallel_dims.py`

## Why this exists

You have `W` GPUs and five parallelism degrees. Rank 37 needs to answer:

- Which pipeline stage am I? Which data shard do I read?
- Which ranks do I all-reduce gradients with?
- Which ranks do I ring-rotate `K`/`V` with?
- Which ranks hold the other shards of `W_1`?

Answering these with rank arithmetic (`stage = rank // (dp*tp)`, and so on) is
where distributed training goes to die. The formulas are all *nearly* right, they
are scattered across the codebase, and a wrong one produces silently incorrect
training rather than a crash.

A **device mesh** replaces all of that with one object. And `torchfeather` does
something more interesting than the textbook version: it builds **three different
named views over the same physical ranks**, because dense layers, MoE layers, and
the dataloader need to group the same GPUs differently.

## 1. What a device mesh is

An `n`-dimensional array of ranks with **named** dimensions.

```python
mesh = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("pp", "dp", "tp"))
```

With `W = 8`, ranks fill the array in **row-major order** — the *last* dimension
varies fastest:

```
rank:  0  1  2  3  4  5  6  7
       │  │  │  │  │  │  │  │
(pp,dp,tp):
       (0,0,0) (0,0,1) (0,1,0) (0,1,1) (1,0,0) (1,0,1) (1,1,0) (1,1,1)
```

Slicing gives a sub-mesh whose `get_group()` is a real process group:

```python
mesh["tp"]           # this rank's TP peers
mesh["dp"].size()    # 2
mesh["dp"].get_local_rank()          # my coordinate on the dp axis
mesh["dp", "tp"]     # a 2-D sub-mesh
```

Everything a rank needs to know about its position is a mesh query. No arithmetic.

## 2. The ordering rule — the one design decision that matters

Because filling is row-major, **the last mesh dimension has physically adjacent
ranks**. Adjacent ranks are usually on the same node, behind NVLink.

```
Node 0 = ranks 0-7 (NVLink, ~900 GB/s)
Node 1 = ranks 8-15
   ...inter-node: InfiniBand, ~25 GB/s
```

Now recall the communication intensities from topic 14 §6:

| Parallelism | Traffic per step | Placement |
|-------------|------------------|-----------|
| TP | `O(L)` collectives, on the critical path of every layer | **innermost** (fastest link) |
| CP | `O(L)` ring steps, but overlappable with attention compute | inner |
| FSDP (`dp_shard`) | `O(L)` all-gathers, prefetchable | middle |
| DP replicate | 1 all-reduce per step, overlappable | outer |
| PP | `O(P_pp · M)` point-to-point, small, overlappable | **outermost** (slowest link) |

```
┌──────────────────────────────────────────────────────────────┐
│   mesh_dim_names = (pp, dp_replicate, dp_shard, cp, tp)      │
│                     ↑                                   ↑    │
│                slowest link                      fastest link│
└──────────────────────────────────────────────────────────────┘
```

Get this backwards — put `pp` innermost and `tp` outermost — and TP's per-layer
all-reduces cross the inter-node fabric. The job still runs. It is roughly 20×
slower on the communication, which typically means 3–5× slower overall. This is
the single highest-leverage configuration decision in distributed training, and
the mesh dimension order is where it is expressed.

## 3. `torchfeather`'s design: one world mesh, three views

The textbook approach builds one `n`-D mesh. The reference implementation builds
a **flat** world mesh and then `_unflatten`s it three different ways:

```python
self._world_mesh = init_device_mesh(device_type, (self.world_size,),
                                    mesh_dim_names=("world",))

dataloading_mesh = unflatten_mesh(self._world_mesh,
    ("pp", "batch", "cp", "tp"),
    (self.pp,  batch,  self.cp, self.tp))

loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")

dense_mesh = unflatten_mesh(self._world_mesh,
    ("pp", "dp_replicate", "fsdp", "tp"),
    (self.pp, self.dp_replicate, fsdp, self.tp))

sparse_mesh = unflatten_mesh(self._world_mesh,
    # This basically means "EP" can borrow from dp_shard, cp and TP.
    # dp_shard * cp * tp = efsdp * ep * etp
    ("pp", "dp_replicate", "efsdp", "ep", "etp"),
    (self.pp, self.dp_replicate, efsdp, self.ep, self.etp))
```

All three views have the same total size and cover the same physical ranks in the
same order. They differ only in **how the middle of the rank space is grouped**.

Why three?

- **`dataloading`** — answers "who is my data-parallel peer?" (`batch`) and
  "who do I share a sequence with?" (`cp`). The dataloader and the loss reduction
  use this view (topics 13 §3, 16 §3).
- **`dense`** — answers "who do I all-gather parameters with?" (`fsdp`) for
  ordinary dense layers.
- **`sparse`** — answers the same question for MoE layers, where the grouping is
  different because experts are sharded across `ep` (topics 28–30).

The insight: **the same GPU is simultaneously a member of several different
logical groups, and which grouping is correct depends on which parameter you are
talking about.** A single `n`-D mesh cannot express that; multiple views over one
world mesh can.

### 3.1 The derived degrees

```python
batch = self.dp_replicate * self.dp_shard
fsdp  = self.dp_shard * self.cp
efsdp = fsdp * self.tp // (self.etp * self.ep)
```

and the master constraint:

```python
assert dp_replicate * dp_shard * cp * tp * pp == self.world_size
```

Note `ep` and `etp` are **absent** from that product. They do not consume ranks
of their own — they *borrow* from `dp_shard`, `cp`, and `tp`. That is what
`efsdp = fsdp·tp / (etp·ep)` encodes.

### 3.2 `fsdp = dp_shard × cp` — the non-obvious one

Why does the FSDP group include the context-parallel dimension?

FSDP shards **parameters** across ranks that hold *different data*. Ask which
dimensions satisfy that:

| dim | different data? | same parameters? | can FSDP shard over it? |
|-----|-----------------|------------------|--------------------------|
| `dp_shard` | yes | yes | **yes** — by definition |
| `cp` | yes (different sequence slices) | **yes** — CP does not touch parameters | **yes** |
| `tp` | no (same tokens) | no (already sharded) | no |
| `pp` | no (same micro-batch) | no (different layers) | no |

Context parallelism splits the *sequence*, so CP ranks process different tokens
while holding identical copies of every weight. Those identical copies are pure
waste — so fold `cp` into the FSDP group and shard the parameters across it too.

**You get context parallelism's memory saving on activations *and* an extra
factor of `P_cp` on parameter memory, for free.** This is a real and easily-missed
optimization, and it is why `fsdp_enabled` returns true when *either*
`dp_shard > 1` **or** `cp > 1`:

```python
@property
def fsdp_enabled(self):
    return self.dp_shard_enabled or self.cp_enabled
```

### 3.3 The `loss` mesh

```python
loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")
```

`batch × cp` flattened into one dimension. Justified in topic 13 §3: these are
exactly the dimensions over which the batch's *tokens* are distributed. `tp`
ranks hold the same tokens; `pp` stages hold different layers and only one of
them computes a loss.

`_flatten` matters because the reduction should be a **single** collective over
`|batch|·|cp|` ranks, not a nested pair.

### 3.4 The `"fake"` backend trick

```python
def _mesh_exist(self, name, degree):
    if name == "fsdp":
        # Always keep fsdp mesh with real backend so fully_shard()
        # can apply MixedPrecisionPolicy even at degree 1.
        return True
    if name == "efsdp":
        return self.ep > 1
    return degree > 1

def unflatten_mesh(world_mesh, dim_names, dim_degrees):
    backend_override = {name: "fake" for name, degree in zip(dim_names, dim_degrees)
                        if not self._mesh_exist(name, degree)}
    return world_mesh._unflatten(0, dim_degrees, dim_names,
                                 backend_override=backend_override)
```

Degree-1 dimensions get a **fake** backend: the mesh dimension exists with size
1, and collectives on it are no-ops, so no real process group is created. Two
benefits:

- **Uniform code.** Every view always has all its dimensions. No
  `if tp_enabled: ... else: ...` branching at every use site.
- **No wasted NCCL communicators.** Real process groups cost memory and setup
  time; a job with `cp = 1, ep = 1` should not pay for them.

The `fsdp` exception is a good detail: `fully_shard()` is applied even at degree
1, not to shard anything, but because it is also the hook that installs the
mixed-precision policy (topic 20). So that dimension needs a real backend
regardless.

And `get_optional_mesh` returns `None` for a fake dimension, which is what makes
`get_optional_mesh("pp")` the natural "is PP on?" query used throughout the
trainer.

## 4. Worked configurations

Verified against the real validator.

### 4.1 8 GPUs: PP × FSDP

```
pp=2, dp_replicate=1, dp_shard=4, cp=1, tp=1     (2·1·4·1·1 = 8 ✓)

batch = 1×4 = 4       fsdp = 4×1 = 4       loss = 4×1 = 4
dataloading:  2 × 4 × 1 × 1 = 8
dense:        2 × 1 × 4 × 1 = 8
```

Note `dp_shard = -1` in the config auto-solves:

```python
if dp_shard < 0:
    self.dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
```

"Use whatever ranks are left for FSDP" — the sane default, since FSDP is the
parallelism you want as much of as you can get.

Physical layout on one 8-GPU node:

```
ranks 0-3  →  pp stage 0,  fsdp ranks 0-3
ranks 4-7  →  pp stage 1,  fsdp ranks 0-3
```

`pp` is outermost, so the pipeline cut falls between rank groups. On a single
node that is arbitrary; across two nodes it puts the pipeline boundary on the
inter-node link, which is exactly right.

### 4.2 64 GPUs (8 nodes): everything on

```
pp=2, dp_replicate=1, dp_shard=4, cp=2, tp=4, ep=4, etp=4    (2·1·4·2·4 = 64 ✓)

batch = 1×4 = 4          fsdp  = 4×2 = 8
loss  = 4×2 = 8          efsdp = 8×4/(4×4) = 2
seq_len_divisor = tp × (cp × 2) = 4 × 4 = 16

dataloading:  2 × 4 × 2 × 4       = 64
dense:        2 × 1 × 8 × 4       = 64
sparse:       2 × 1 × 2 × 4 × 4   = 64
```

Read the three views. Dense layers shard parameters over 8 ranks (`fsdp`); MoE
layers shard experts over 4 ranks (`ep`) and expert weights over 4 more (`etp`),
leaving only 2 for `efsdp`. Same 64 GPUs, three groupings.

`tp = 4` is innermost, so TP groups are 4 adjacent ranks — within a node. ✓

### 4.3 128 GPUs: HSDP × TP

```
pp=1, dp_replicate=2, dp_shard=8, cp=1, tp=8    (1·2·8·1·8 = 128 ✓)

batch = 2×8 = 16      fsdp = 8      loss = 16
dense: 1 × 2 × 8 × 8 = 128
```

`tp = 8` fills a node exactly; `dp_shard = 8` shards parameters across 8 nodes;
`dp_replicate = 2` gives two such groups that only all-reduce once per step.
This is **HSDP** (topic 20 §6), and it is the standard shape for a large
multi-node run.

### 4.4 What gets rejected

```python
if ep > 1:
    assert etp == tp or etp == 1, (tp, etp)
    if etp == tp:
        # EP would borrow all cp and some dp_shard degree
        assert ep % cp == 0 and (dp_shard * cp) % ep == 0
    elif etp == 1:
        # EP would borrow all cp and tp and some dp_shard degree
        assert ep % (cp * tp) == 0 and (dp_shard * cp * tp) % ep == 0
```

`ep = 3` with `cp = 2` is rejected (`3 % 2 ≠ 0`) — the expert-parallel group
cannot be carved out of the available rank structure. Note `etp` is restricted to
exactly two values: `tp` (experts get the same tensor parallelism as dense
layers) or `1` (experts get none, so `tp`'s ranks are freed for `ep` instead).
Anything in between would require a rank grouping that does not tile.

Fail loudly at startup. A mesh that "sort of" works is the worst outcome.

## 5. `seq_len_divisor`

```python
@property
def seq_len_divisor(self):
    # Sequence Parallel requires that seq_len be divisible by TP degree.
    # Context Parallel requires that seq_len be divisible by 2 * CP degree,
    # when load balancing is enabled (by default).
    return self.tp * (self.cp * 2)
```

Two independent divisibility requirements, multiplied:

- **`tp`** — sequence parallelism shards activations along the sequence in the
  norm/dropout regions between TP blocks (topic 22).
- **`2 · cp`** — the `2` is the **zigzag load-balanced** sharding of ring
  attention (topic 23): each CP rank takes *two* chunks, one from the front half
  and one from the back half, so that causal masking gives every rank equal work.

So `tp = 4, cp = 2` requires `S` divisible by 16. Worth checking before a run:
an indivisible `S` produces either a shape error deep inside a kernel or, with
silent truncation, a subtly wrong loss.

## 6. Querying the mesh

```python
def get_optional_mesh(self, dims: str | list[str]) -> DeviceMesh | None:
    if any(not self._mesh_exist(dim, self._meshes[dim].size()) for dim in dims):
        return None                    # dimension is degree-1 / fake
    if len(dims) == 1:
        return self._meshes[dims[0]]
    for global_mesh in self._global_meshes.values():
        if set(dims).issubset(set(global_mesh.mesh_dim_names)):
            return global_mesh[tuple(dims)]
    raise ValueError(f"Invalid mesh name combinations {dims}.")
```

Two behaviours worth noting:

- **Multi-dimensional slices must live in one view.** `get_mesh(["fsdp", "tp"])`
  works (both are in `dense`); `get_mesh(["batch", "fsdp"])` raises, because no
  single view contains both. That is not a limitation, it is the type system
  telling you the combination is not a meaningful group: `batch` and `fsdp` are
  two different *decompositions* of the same ranks, so their "intersection" is
  not well-defined.
- **`get_optional_mesh` vs `get_mesh`.** The former returns `None` for a disabled
  dimension (used for "is this on?" checks like the `pp_mesh` argument to
  `clip_grad_norm_`); the latter raises. Use `get_mesh` when the dimension is
  required by the code path you are in, so a misconfiguration fails at the top
  rather than producing a no-op collective.

## 7. Choosing a configuration

A workable procedure, in order:

```
1. TP = the largest degree that stays inside one node (usually 8),
        and only if the model needs it. TP is the most expensive per-layer.
2. PP = the smallest degree that makes the model fit, given TP and FSDP.
        Remember the bubble: keep it small, and ensure M ≥ 4·P_pp.
3. CP  = only if the sequence is long enough that activations dominate
        (topic 02 §4.2). Remember it also buys FSDP an extra factor.
4. FSDP (dp_shard) = all remaining ranks, up to the point where the
        all-gather stops overlapping.
5. DP replicate = whatever is left; use HSDP shape when dp_shard would
        otherwise span too many nodes.
6. Check: dp_replicate·dp_shard·cp·tp·pp == world_size
          S % (tp · 2 · cp) == 0
          M = local_batch/micro_batch ≥ 4·P_pp
```

Then measure. The mesh makes the configuration *expressible*; only a profile
tells you it is *good*.

## 8. Check yourself

1. Ranks fill a mesh row-major. Which mesh dimension has physically adjacent
   ranks, and which parallelism belongs there?
2. Give the full recommended dimension order and justify each position by
   communication volume.
3. Why does `torchfeather` build three views over one world mesh instead of one
   5-D mesh? Give a concrete question each view answers.
4. Derive `fsdp = dp_shard × cp`. Why can FSDP shard over the `cp` dimension but
   not over `tp` or `pp`?
5. What extra memory saving does §3.2 buy, and why is it easy to miss?
6. Why is the `loss` mesh `batch × cp` flattened, and why `_flatten` rather than
   a nested slice?
7. What does the `"fake"` backend accomplish? Why is `fsdp` exempt even at
   degree 1?
8. `ep` and `etp` are absent from the world-size product. Explain what they
   borrow from and write the `efsdp` formula.
9. Why is `etp` restricted to `tp` or `1`?
10. Derive `seq_len_divisor = tp · 2 · cp`, explaining both factors, especially
    the `2`.
11. `get_mesh(["batch", "fsdp"])` raises. Explain why that is correct rather
    than a missing feature.
12. You have 256 GPUs in 32 nodes of 8 and a model needing 4-way TP to fit.
    Propose a full configuration and check every constraint.

## Next

→ [19 — Distributed communication collectives](19-communication-collectives.md)
