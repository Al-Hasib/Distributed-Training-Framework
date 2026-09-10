# 28 — Expert Parallelism

> Video: 18:36:24 — Expert Parallelism
> Code: `torchfeather/distributed/expert_parallel.py` (`ExpertParallel`)

## Why this exists

Topic 27 ended with a wall: tensor parallelism divides expert memory by
`P_tp ≤ 8`, and MoE's whole purpose is to make `E` large. At `E = 256` that is
not enough by an order of magnitude.

The fifth and final cut: **shard the expert dimension.** Rank `r` owns
`E / P_ep` experts — whole, unsliced.

This is the cut that changes the *shape* of the computation rather than just its
distribution. Under every previous parallelism, data flowed through a fixed
graph and collectives repaired the cuts. Under EP, **which device a token visits
depends on the token's content**. The routing decision becomes a communication
pattern, and that pattern is an all-to-all.

## 1. The cut

```python
@staticmethod
def _partition_fn(_name, mod, device_mesh):
    # shard on the expert dimension
    for name, param in mod.named_parameters(recurse=False):
        dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
        mod.register_parameter(name, dist_param)
```

`Shard(0)` — dimension 0 of `[E, ·, ·]` is the expert index. Six words of code
for the whole idea.

```
E = 12, P_ep = 4:

rank 0 owns experts 0,1,2      rank 1 owns experts 3,4,5
rank 2 owns experts 6,7,8      rank 3 owns experts 9,10,11
```

Every expert's `w1`, `w2`, `w3` are **complete** on their owning rank. Contrast
with topic 27, where all ranks had all experts in slices. The two are orthogonal,
which is why they compose (topic 30).

Memory:

```
TP:  E · 3 · d · d_ff^E / P_tp        still O(E)
EP:  E · 3 · d · d_ff^E / P_ep        O(E / P_ep) — and P_ep can be 64
```

With `E = 256, P_ep = 64`: 4 experts per rank. Now it fits.

## 2. The consequence: tokens must travel

A token whose router picked expert 7 must be processed by rank 2, because rank 2
is the only rank with expert 7's weights. And every rank has tokens for every
expert.

```
rank 0 has tokens for experts 0..11, but owns only 0,1,2
rank 1 has tokens for experts 0..11, but owns only 3,4,5
...
```

So each rank must send a *different subset* of its tokens to each other rank, and
receive a different subset from each. That is precisely the definition of an
**all-to-all** (topic 19 §1).

```
       ┌────── dispatch (all-to-all) ──────┐
tokens │  rank r sends its expert-j tokens │  local experts   ┌── combine ──┐
   ────┤  to the rank owning expert j      ├── grouped_mm ────┤  all-to-all │──► out
       └───────────────────────────────────┘                  │  (adjoint)  │
                                                              └─────────────┘
```

And by topic 11 §2.5 the combine is *literally the backward* of the dispatch —
the same collective with the split and concat dimensions exchanged. The
implementation writes one and gets the other by swapping two arguments (topic 29
§4).

## 3. Why EP "borrows" ranks

From topic 18 §3.1, `ep` is absent from the world-size product:

```python
assert dp_replicate * dp_shard * cp * tp * pp == self.world_size   # no ep, no etp
efsdp = fsdp * tp // (etp * ep)
```

EP is not a new axis of the machine; it is a **regrouping** of ranks that already
exist. The reason is that the expert-parallel group must consist of ranks that
hold *different tokens* — which is exactly what `dp_shard`, `cp` and (with
`etp=1`) `tp` provide.

```python
if etp == tp:
    # EP would borrow all cp and some dp_shard degree
    assert ep % cp == 0 and (dp_shard * cp) % ep == 0
elif etp == 1:
    # EP would borrow all cp and tp and some dp_shard degree
    assert ep % (cp * tp) == 0 and (dp_shard * cp * tp) % ep == 0
```

Read what "borrow" costs. With `etp = 1`, the `tp` ranks stop tensor-parallelizing
the experts and become expert-parallel ranks instead — so the experts get **no**
tensor parallelism, and the ranks that were splitting matrices are now splitting
experts. That is a genuine trade, not free extra parallelism, and it is why
topic 30 exists to give you the other option.

This is also why the mesh needs a separate **`sparse` view**: the same GPU is in
one group for dense parameters (`fsdp`) and a different group for expert
parameters (`ep`, `etp`, `efsdp`). One `n`-D mesh cannot say that.

## 4. Load imbalance becomes a wall-clock problem

Topic 26 §4 framed load balancing as a *modelling* problem — dead experts learn
nothing. Under EP it becomes a *scheduling* problem too, and this is the sharpest
consequence of expert parallelism.

The implementation is **dropless** (topic 26 §8), so an expert receiving 3× its
share of tokens does 3× the work. Its owning rank takes 3× as long. And the
all-to-all is a **synchronization point**: every rank waits for the slowest.

```
perfectly balanced:  every rank does T·k/P_ep tokens        step time = t
one rank at 3×:      that rank does 3·T·k/P_ep              step time = 3t
                     all other ranks idle for 2t
```

So the same imbalance that used to cost *quality* now costs *throughput*
directly, and the cost is set by the **maximum** over ranks, not the average.
Two consequences:

- **Balancing is mandatory, not optional.** Topic 26 §4.2's bias mechanism is
  what keeps EP efficient. Its `sign`-based update and the global all-reduce of
  the counts (topic 24 §1.3) are load-balancing *for the scheduler*.
- **Fine granularity helps.** More, smaller experts (topic 26 §6) means the
  law of large numbers smooths the per-rank load: with 4 experts per rank a
  single hot expert dominates; with 64 the variance averages out. Granularity is
  a systems argument as well as a modelling one.

Recall also topic 13 §8.1's principle: *anything that makes one rank slower makes
every rank slower.* Expert imbalance is the largest such source in an MoE run.

## 5. Cost: EP versus TP for MoE

The comparison that decides your configuration.

| | TP for MoE (t27) | EP (this topic) |
|---|---|---|
| what is sharded | each expert's `d_ff^E` | the expert index |
| experts per rank | all `E`, sliced | `E/P_ep`, whole |
| memory | `O(E)/P_tp` | `O(E/P_ep)` |
| max degree | `min(n_heads, node)` ≈ 8 | up to `E` |
| what moves | activations: all-gather + reduce-scatter | tokens: 2 all-to-all |
| bytes moved | `2·B·S·d·(P−1)/P` per layer | `≈ 2·B·S·k·d·(P−1)/P` per layer |
| sensitive to imbalance | no | **yes** |
| kernel efficiency | full-width `d_ff/P` matmul | full `d_ff^E`, variable rows |

The volume comparison is the interesting row. EP moves roughly `k×` more bytes
than TP, because each token is sent to `k` experts:

```
TP:  every rank sees every token once           →  O(B·S·d)
EP:  every token travels to k experts           →  O(B·S·k·d)
```

At `k = 8` that is 8× the traffic. **EP is more expensive in communication and
more sensitive to imbalance — and it is the only option that scales to large
`E`.** So the rule is:

```
Use EP because you must (memory), and use exactly as much as you must.
Prefer granularity for balance, and keep P_ep on a fast link.
```

## 6. Where EP touches the rest of the framework

EP is the most invasive of the five cuts. Every one of these has already been
covered; collected here because you will need them together.

| Subsystem | What EP changes | Where |
|-----------|-----------------|-------|
| **Device mesh** | a separate `sparse` view: `(pp, dp_replicate, efsdp, ep, etp)` | t18 §3 |
| **FSDP** | routed experts wrapped on `dp_replicate_efsdp_mesh`, not `fsdp` | t20 §8 |
| **FSDP sharding** | `Shard(1)` fallback when `efsdp · ep > E` (empty shards otherwise) | t20 §8 |
| **FSDP prefetch** | must be **explicit**: EP's D2H sync breaks order inference | t20 §7 |
| **Optimizer** | parameter groups by mesh, or fused DTensor dispatch fails | t24 §1.2 |
| **Grad clipping** | the `ep` dimension counted exactly once (`ep_enabled` flag) | t13 §5 |
| **Checkpointing** | `expert_bias` must be saved (`persistent=True`) | t24 §4.5 |
| **Seeding** | `ep` can be added to `distinct_seed_mesh_dims` | t13 §7 |
| **Balancing** | now a throughput requirement, not just a quality one | §4 above |

Note the third row is a genuinely subtle one. EP shards dim 0 (experts) and FSDP
would too by default, so with `E = 8, P_ep = 4` each rank has 2 local experts and
FSDP sharding those across 4 `efsdp` ranks leaves two ranks with **nothing** — no
gradient to contribute, degenerate collectives. Sharding dim 1 (the hidden
dimension) instead divides evenly. The condition `efsdp_degree * ep_degree >
num_experts` is exactly "the expert axis has run out of parallelism to give".

## 7. The style, assembled

```python
class ExpertParallel(ParallelStyle):
    def __init__(self):
        self.input_splits = None       # how many rows to send to each EP rank
        self.output_splits = None      # how many rows to receive from each
        self.input_shape = None        # for _unpermute in the combine
        self.permuted_indices = None

    def _token_dispatch(self, mod, inputs, device_mesh): ...   # topic 29
    def _token_combine(self, _mod, routed_output, device_mesh): ...  # topic 29

    @staticmethod
    def _partition_fn(_name, mod, device_mesh):
        for name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(name,
                nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])))

    def _apply(self, module, device_mesh):
        return distribute_module(module, device_mesh,
                                 partition_fn=ExpertParallel._partition_fn,
                                 input_fn=self._token_dispatch,
                                 output_fn=self._token_combine)
```

The shape of every `ParallelStyle` (topic 22 §1): a partition function for the
parameters, an input hook, an output hook. What makes this one different is that
the hooks are **stateful** — `_token_dispatch` computes `input_splits`,
`output_splits`, `input_shape` and `permuted_indices` and stashes them on the
style object for `_token_combine` to reuse.

That statefulness is forced by the design: the combine must send tokens back
along exactly the paths the dispatch sent them, and those paths are
data-dependent. There is no way to recompute them without redoing the routing.

It is also a hazard: the style instance is per-module (one `ExpertParallel()` per
MoE layer), so the state is not shared between layers — but it *is* shared
between micro-batches within a layer, which is only safe because dispatch and
combine of one micro-batch always bracket each other. Under a pipeline schedule
that interleaved two micro-batches' MoE layers this would be a bug; the schedules
in topic 15 do not, since a stage's forward runs to completion before the next
micro-batch starts.

## 8. Check yourself

1. Give the one-line sharding that defines EP, and say which dimension of what
   shape it refers to.
2. Compare TP and EP memory for `E = 256, P_tp = 8, P_ep = 64`. Why is TP
   insufficient?
3. Why must tokens move under EP? Derive that the required pattern is an
   all-to-all rather than a broadcast or an all-gather.
4. Why is the combine "free" once the dispatch is written?
5. Why is `ep` absent from the world-size product? Which dimensions does it
   borrow from, and what does borrowing `tp` cost?
6. Why does EP require a separate `sparse` mesh view?
7. Under dropless EP, why does imbalance cost throughput rather than quality?
   Compute the step time for one rank at 3× load.
8. Give two independent reasons fine granularity helps under EP.
9. Derive EP's communication volume and show it is `≈ k×` TP's. When is that
   still the right choice?
10. `efsdp · ep > E`. What goes wrong with FSDP's default `Shard(0)`, and what is
    the fix?
11. Why must FSDP prefetch be explicit under EP? Trace the causal chain.
12. Why are `_token_dispatch` and `_token_combine` stateful? Under what pipeline
    schedule would that statefulness be unsafe, and why is it safe here?

## Next

→ [29 — All-to-all token dispatch and combine](29-all-to-all-dispatch-combine.md)
