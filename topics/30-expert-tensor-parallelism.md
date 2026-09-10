# 30 — Expert Tensor Parallelism

> Video: 19:27:38 — Expert Tensor Parallelism
> Code: `torchfeather/distributed/expert_parallel.py`
> (`ExpertTensorParallel`, `ReordererSequenceParallel`, `apply_moe_ep_tp`)

## Why this exists

Two ways to distribute an MoE FFN, and neither is sufficient alone:

```
TP  (topic 27):  all E experts on every rank, each sliced   → memory O(E)/P_tp
EP  (topic 28):  E/P_ep experts per rank, unsliced          → memory O(E/P_ep)
                                                              but the whole
                                                              expert must fit
```

At DeepSeek-V3 scale you need both: `E = 256` requires EP for the count, and
each expert is still `3 · 7168 · 2048 = 44M` parameters, which — multiplied by
the several experts per rank and by 58 MoE layers — wants slicing too.

**Expert tensor parallelism** applies both cuts simultaneously on a 2-D
`(ep, etp)` mesh. This is the final composition of the course, and it also
forces the last piece of machinery: `ReordererSequenceParallel`, which exists
purely to resolve a redundancy that appears when TP ranks are *borrowed* for EP.

## 1. The 2-D sharding

```python
@staticmethod
def _partition_fn(_name, mod, device_mesh):
    for name, param in mod.named_parameters(recurse=False):
        # colwise TP: shard out_dim (dim 1) for w1/w3;
        # rowwise TP: shard out_dim (dim 2) for w2
        _tp_shard_dims = {"w1": 1, "w2": 2, "w3": 1}
        shard_dim = _tp_shard_dims.get(name)
        placements = ((Shard(0), Shard(shard_dim)) if shard_dim is not None
                      else (Shard(0), Replicate()))
        mod.register_parameter(name, nn.Parameter(
            distribute_tensor(param, device_mesh, placements)))
```

Two placements, one per mesh dimension:

```
             ep dim        etp dim
w1  [E, d_ff, d]   Shard(0)   Shard(1)      expert index  ×  d_ff  (colwise)
w2  [E, d, d_ff]   Shard(0)   Shard(2)      expert index  ×  d_ff  (rowwise)
w3  [E, d_ff, d]   Shard(0)   Shard(1)      expert index  ×  d_ff  (colwise)
anything else      Shard(0)   Replicate()
```

`Shard(0)` is topic 28's expert cut; `Shard(shard_dim)` is topic 27's weight cut
— the same `{w1: 1, w2: 2, w3: 1}` map. **The two are literally composed**, with
no new algebra, because they cut orthogonal axes.

```
E = 8, P_ep = 4, P_etp = 2:

              etp 0            etp 1
   ep 0   E0,E1 : d_ff[0:h/2]  E0,E1 : d_ff[h/2:h]
   ep 1   E2,E3 : d_ff[0:h/2]  E2,E3 : d_ff[h/2:h]
   ep 2   E4,E5 : ...          E4,E5 : ...
   ep 3   E6,E7 : ...          E6,E7 : ...
```

Memory per rank: `E · 3 · d · d_ff^E / (P_ep · P_etp)`. Both factors multiply.

## 2. The two operating points

`etp` is not free to choose. From topic 18 §4.4:

```python
if ep > 1:
    assert etp == tp or etp == 1, (tp, etp)
```

Exactly two options, and they are genuinely different strategies.

### `etp == tp` — experts get the same TP as dense layers

```
ep borrows from:   cp, dp_shard
    assert ep % cp == 0 and (dp_shard * cp) % ep == 0
```

The `tp` ranks keep tensor-parallelizing, both for the dense layers *and* for
the experts. `ep` is carved out of `cp × dp_shard`. This is the
`ExpertTensorParallel` path.

- Each rank's slice is `1/(P_ep · P_tp)` of the expert weights — the smallest.
- The all-to-all runs on the `ep` mesh only, so `P_tp` ranks each hold the same
  tokens and each sends its own weight-slice's worth of work.

### `etp == 1` — TP ranks are repurposed as EP ranks

```
ep borrows from:   cp, tp, dp_shard
    assert ep % (cp * tp) == 0 and (dp_shard * cp * tp) % ep == 0
```

Experts get **no** tensor parallelism; the ranks that were splitting matrices now
split experts instead. This is the `ExpertParallel` path with a larger `P_ep`,
and it needs `ReordererSequenceParallel` (§4).

- Larger `P_ep` for the same world size — better for very large `E`.
- Each expert's weights must fit whole on one rank.

### Choosing

```
Experts individually large (big d_ff^E)?      →  etp = tp   (slice them)
Experts numerous and individually small?      →  etp = 1    (more EP)
E large AND experts large?                    →  etp = tp, and raise P_ep
                                                 by taking from dp_shard
```

Note `etp` strictly between `1` and `tp` is rejected. The reason is structural,
not a missing feature: the rank grouping must **tile** the world mesh, and
`(pp, dp_replicate, efsdp, ep, etp)` has to multiply out to `world_size` with
`efsdp = fsdp · tp / (etp · ep)` an integer (topic 18 §3.1). Intermediate values
of `etp` do not generally tile.

## 3. The dispatch, doubled

```python
def _token_dispatch(self, mod, inputs, device_mesh):
    routed_input, num_tokens_per_expert = inputs
    # NOTE: This no-op marking of the tensor as Partial is due to the same
    # reason as in `TensorParallel`.
    routed_input = DTensor.from_local(
        routed_input, self.etp_mesh, (Replicate(),)
    ).to_local(grad_placements=(Partial(),))
    return super()._token_dispatch(mod, (routed_input, num_tokens_per_expert),
                                   self.ep_mesh)
```

`ExpertTensorParallel` subclasses `ExpertParallel`, and its dispatch does exactly
two things:

1. **The `Partial` gradient marking on the `etp` mesh** — topic 27 §2, verbatim.
   The grouped matmul runs on local shards, so each `etp` rank produces only a
   partial input gradient; the round-trip declares it so backward all-reduces
   across `etp`. Forward no-op, backward all-reduce.
2. **Delegates to `super()` on the `ep` mesh** — topic 29's two all-to-alls,
   unchanged.

```python
def _token_combine(self, _mod, routed_output, device_mesh):
    return super()._token_combine(_mod, routed_output, self.ep_mesh)
```

Two mesh dimensions, two concerns, cleanly separated:

```
etp  →  gradient reduction of the input   (a weight-slicing concern)
ep   →  all-to-all of the tokens          (an expert-placement concern)
```

Note the assertion in `_apply`:

```python
def _apply(self, module, device_mesh):
    assert device_mesh.mesh_dim_names == ("ep",), (device_mesh.mesh_dim_names,)
    return distribute_module(module, self.ep_etp_mesh,
                             partition_fn=ExpertTensorParallel._partition_fn, ...)
```

The style is *invoked* with the `ep` mesh (matching `ExpertParallel`'s interface)
but *distributes* on the 2-D `ep_etp_mesh` it was constructed with. The assertion
pins that contract, because passing the wrong mesh here would shard on the wrong
axes with no error.

## 4. `ReordererSequenceParallel` — resolving a redundancy

The last piece, and it exists only in the `etp == 1` case. The comment states
the purpose:

```python
if ep_mesh is not None and not etp_enabled:
    # If TP is borrowed for EP, then split the tokens across TP ranks so that
    # the reorderer, the all-to-all comms, and routed experts computation are
    # effectively running Sequence Parallel (split along the folded bs*slen dim)
    moe_layer_plan.update({"moe.reorderer": ReordererSequenceParallel()})
```

### 4.1 The problem

The MoE module boundary all-gathers its input (topic 27 §4), so **every `tp`
rank holds the full `[T, d]` activation**. That is right for the dense TP path,
where each rank then does its own weight slice.

But with `etp == 1` the experts are *not* sliced across `tp`. Those ranks are EP
ranks now. So if every one of them routes and dispatches the full `T` tokens:

- all `P_tp` ranks compute the same routing,
- all `P_tp` ranks dispatch the same tokens,
- the all-to-all carries `P_tp ×` the necessary data,
- and the experts compute each token `P_tp` times.

Pure `P_tp`-fold redundancy in both compute and communication.

### 4.2 The fix

Split the tokens across the `tp` ranks, so each handles `T/P_tp` of them:

```python
def _prepare_input_fn(self, _mod, inputs, device_mesh):
    top_scores, selected_experts_indices = inputs
    num_tokens, _ = top_scores.shape

    def _split_along_first_dim(x):
        assert x.is_contiguous()
        if num_tokens % device_mesh.size() != 0:
            raise ValueError("Uneven split of tokens is not supported yet. "
                             "Requires TP degree dividing batch size * seq len.")
        local_num_tokens = num_tokens // device_mesh.size()
        offset = device_mesh.get_local_rank() * local_num_tokens
        return x[offset : offset + local_num_tokens]

    return _split_along_first_dim(top_scores), _split_along_first_dim(selected_experts_indices)
```

A plain contiguous slice of the folded `B·S` dimension — sequence parallelism
(topic 21 §5) applied to the routing metadata rather than to activations. No
collective: every rank already has the whole tensor and simply keeps its share.

### 4.3 The index fix-up, which is the interesting part

The reorderer now produces **local** token indices — rank 1 of 2 with `T = 16`
sees tokens `0…7` and emits indices in that range, but those tokens are globally
`8…15`. Downstream, `torch.gather(x, 0, token_indices)` indexes the **full**
`[T, d]` activation (which was all-gathered and is still full-size). Local
indices would gather the wrong rows.

```python
def _prepare_output_fn(self, mod, outputs, device_mesh):
    top_scores, token_indices_experts_sorted, num_tokens_per_expert = outputs
    tokens_per_rank = top_scores.shape[0] // mod.top_k
    local_rank = device_mesh.get_local_rank()
    # Transform from local indices to global indices.
    token_indices_experts_sorted += tokens_per_rank * local_rank
    return top_scores, token_indices_experts_sorted, num_tokens_per_expert
```

One addition. Note the recovery of `tokens_per_rank` from
`top_scores.shape[0] // mod.top_k` — the output has been flattened to
`[T/P_tp · k]`, so dividing by `top_k` recovers the token count.

This is the same class of bug as RoPE under context parallelism (topic 23 §5):
**a rank holding a slice must know its global offset.** Two entirely different
subsystems, one recurring hazard. Whenever you shard something, ask what indices
downstream code will use, and against which tensor.

### 4.4 Why `num_tokens_per_expert` is computed twice

This finally explains a comment in `MoE.forward` that looks redundant:

```python
# NOTE: the reason we need to compute num_tokens_per_expert again is:
#   1st computation in router is to update self.tokens_per_expert which would
#      be the same across all TP ranks.
#   2nd computation in reorderer is for the actual routing and experts
#      computation which would be sharded over TP ranks if
#      expert_tensor_parallel_degree == 1.
#   If tensor_parallel_degree == expert_tensor_parallel_degree, they agree
#      because no `ReordererSequenceParallel` is applied.
```

Two counts, two purposes:

- **In the router**, over *all* `T` tokens. This feeds `tokens_per_expert`, which
  drives the load-balancing bias (topic 26 §4). The bias must be identical on
  every `tp` rank, because `router.gate` and `expert_bias` are replicated
  parameters/buffers — the invariant of topic 27 §3.2.
- **In the reorderer**, over this rank's `T/P_tp` tokens. This feeds the actual
  all-to-all splits and the grouped matmul, which operate on the local shard.

With `etp == tp` no split happens and the two agree, so the recomputation is
harmless. With `etp == 1` they genuinely differ, and using the wrong one gives
either a wrong bias (divergent replicated buffers) or wrong all-to-all splits
(a crash or corrupted routing).

An apparently redundant computation that is load-bearing in exactly one
configuration. Worth remembering as a pattern: **when a value is used for both a
replicated purpose and a sharded purpose, compute it twice rather than trying to
convert.**

## 5. The three regimes, assembled

```python
if ep_mesh is None:
    # TP only. Every rank holds all experts, but each expert's weights are
    # sliced (col/row-wise) across TP ranks.
    experts_mesh, experts_plan = tp_mesh, TensorParallel()

elif tp_mesh is None or not etp_enabled:
    # EP only or EP+TP, but no ETP. Experts are split across EP ranks; each
    # expert's weights stay whole (no col/row slicing).
    # `ReordererSequenceParallel` is applied in this case
    experts_mesh, experts_plan = ep_mesh, ExpertParallel()

else:
    # ETP. Experts are split across EP ranks, and each expert's weights are
    # further sliced (col/row-wise) across the TP ranks in the same EP group.
    experts_mesh, experts_plan = ep_mesh, ExpertTensorParallel(ep_etp_mesh=ep_etp_mesh)

parallelize_module(moe_module.experts, experts_mesh, experts_plan)
```

| Regime | experts/rank | weights sliced? | all-to-all | Reorderer SP | memory |
|--------|--------------|-----------------|------------|--------------|--------|
| TP only | all `E` | yes, across `tp` | **no** | no | `O(E)/P_tp` |
| EP (`etp=1`) | `E/P_ep` | no | yes, on `ep` | **yes** | `O(E/P_ep)` |
| ETP (`etp=tp`) | `E/P_ep` | yes, across `etp` | yes, on `ep` | no | `O(E/(P_ep·P_etp))` |

Three regimes, one dispatch, selected by two config values. Everything else in
the MoE path is shared.

## 6. Course complete — the whole picture

```
                     ONE TRAINING STEP
                            │
   ┌────────────────────────┼────────────────────────┐
   │                     the cuts                    │
   ├─────────────────────────────────────────────────┤
   │ layers    → PP    p2p send/recv        slow link│
   │ batch     → DP    all-reduce           slow link│
   │ params    → FSDP  all-gather + RS      medium   │
   │ matrices  → TP    AG + RS per block    FAST only│
   │ sequence  → CP    ring p2p             medium   │
   │ experts   → EP    all-to-all × 2       medium   │
   │ expert wts→ ETP   + AR on etp          fast     │
   └─────────────────────────────────────────────────┘
                            │
              every cut costs exactly one forward
              collective and one backward collective,
              and the backward one is the ADJOINT
              of the forward one.
```

The five things that, together, are the whole subject:

1. **Reverse-mode autodiff is a product of Jacobians**, and every collective is a
   linear map, so **the backward collective is the adjoint of the forward one** —
   derivable in one line, never memorized (topic 11).
2. **Block matrix multiplication is tensor parallelism**, waiting to be
   distributed (topic 08).
3. **Count FLOPs, count bytes, take the ratio, compare to the hardware.** The
   roofline discipline decides MLA (topic 06), sequence parallelism (topic 21),
   FSDP vs DDP (topic 20), and every collective placement (topic 18).
4. **Which cut to use is set by which resource is binding**; where to place it is
   set by the interconnect hierarchy (topics 01, 18, 25).
5. **Composition is where the bugs live.** Fourteen pairwise interactions
   (topic 25 §3), and the recurring hazards are *global indices for locally-held
   data* (RoPE under CP, the reorderer under ETP) and *replicated parameters with
   divergent gradients* (the MoE gate under TP).

### What to do next

- **Run the minimal examples.** `block_matrix_multiply.py`, `pp_gpipe.py`,
  `tp_swiglu.py`, `tp_attention.py`, `ring_attention.py`. Each is short and each
  makes one derivation concrete.
- **Reproduce the verifications.** Every numerical claim in this course is
  checkable in seconds, and running them is how the material stops being prose.
- **Ablate a real configuration.** Topic 25 §6: bring up one parallelism at a
  time and compare *gradients*, not losses.
- **Then design something new.** The point of deriving rather than memorizing is
  that a new architecture — a different attention, a different sparsity pattern —
  gets the same treatment: find the independent axis, cut it, work out the
  adjoint, count the bytes, and check where it lands against the ridge point.

## 7. Check yourself

1. Give both placements for `w1`, `w2`, `w3` under ETP and say which topic each
   comes from.
2. Compute per-rank expert memory for `E=256, d=7168, d_ff^E=2048, P_ep=8,
   P_etp=8`.
3. State both operating points for `etp` and what `ep` borrows from in each.
4. Why is `etp` strictly between `1` and `tp` rejected?
5. `ExpertTensorParallel._token_dispatch` does two things. Name each, say which
   mesh it acts on, and which topic derived it.
6. Why does `_apply` assert `device_mesh.mesh_dim_names == ("ep",)` while
   distributing on a 2-D mesh?
7. Explain the redundancy `ReordererSequenceParallel` removes. Quantify it in
   both compute and communication.
8. Why is the token split a plain slice with no collective?
9. Why must `token_indices_experts_sorted` be offset by
   `tokens_per_rank * local_rank`? Which earlier bug is this the same class as?
10. Why is `num_tokens_per_expert` computed twice? Give both purposes and say
    when the two values differ.
11. Fill in the three-regime table from memory.
12. State the five ideas of the course, and for each, name a topic where it
    decided a design.

## Appendices

- [Notation and symbols](appendix-notation.md)
- [Papers and references](appendix-references.md)
- [Glossary](appendix-glossary.md)

← [Back to the course index](README.md)
