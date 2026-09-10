# 20 — Implementing Device Meshes, DDP and FSDP

> Video: 10:49:41 — Implementing Device Meshes, DDP and FSDP
> Code: `torchfeather/distributed/model_parallel.py`, `torchfeather/model/parallelize.py`,
> `torchfeather/distributed/activation_checkpoint.py`

## Why this exists

DDP replicates 18 bytes per parameter on every rank (topic 01 §2). For a 70B
model that is 1.26 TB, and adding GPUs does not help.

FSDP fixes it with a single idea: **a rank only needs a parameter while it is
computing with that parameter.** Store `1/P` of everything; gather a layer's
weights just before you use it; throw them away immediately after.

This topic derives the memory saving stage by stage, then reads the real
implementation — where every argument is a memory-vs-communication trade with a
non-obvious default.

## 1. ZeRO, derived

Per parameter, mixed-precision Adam holds:

```
bf16 parameters        2 bytes    needed for the forward pass
fp32 gradients         4 bytes    accumulated in backward
fp32 Adam m, v         8 bytes    optimizer state
fp32 master weights    4 bytes    the fp32 copy the optimizer updates
                      ─────────
                      18 bytes
```

Now ask, for each item, **when is it actually needed?**

| Item | Needed when | Shardable? |
|------|-------------|------------|
| Adam `m`, `v`, master | only inside `optimizer.step()`, for parameters this rank updates | **yes** — pick a rank per parameter |
| gradients | only to feed the optimizer | **yes** — reduce-scatter instead of all-reduce |
| parameters | during forward and backward of *this layer* | **yes** — gather just in time |

Three answers, three ZeRO stages:

```
              params  grads   opt     total (P=64)   70B model
DDP            2       4      12      18             1.26 TB
ZeRO-1         2       4      12/P     6.19          433 GB
ZeRO-2         2       4/P    12/P     2.25          157 GB
ZeRO-3 (FSDP)  2/P     4/P    12/P     0.28          19.7 GB
```

ZeRO-3 divides *everything* by `P`. A 70B model goes from not fitting on any
machine to 19.7 GB per GPU, leaving 60 GB for activations. **This is the
technique that makes large-model training possible on commodity clusters**, and
PyTorch's `fully_shard` (FSDP2) implements it.

### 1.1 The mechanics

For each wrapped module, per forward pass:

```
1.  all-gather this module's parameter shards      → full params, transiently
2.  compute the forward
3.  free the gathered params                        ("reshard after forward")

backward:
4.  all-gather again (they were freed)
5.  compute grads
6.  reduce-scatter the gradients                    → each rank keeps its shard
7.  free the gathered params
```

Steps 1 and 4 are `Shard(0) → Replicate()`; step 6 is `Partial → Shard(0)`.
Straight off topic 19 §3's table. Nothing new — FSDP is that table applied per
module.

### 1.2 The cost

From topic 19 §2.3:

```
DDP  : 1 all-reduce                     = 2V(P−1)/P
FSDP : 2 all-gathers + 1 reduce-scatter = 3V(P−1)/P     →  1.5× DDP
```

**50% more traffic for a `P×` memory reduction.** And because layer order is
known in advance, the all-gathers can be *prefetched* (topic 19 §4.3), so in
practice the extra volume is often entirely hidden. The number that matters is
never the volume; it is the volume that fails to overlap.

## 2. Granularity: what `fully_shard` is called on

```python
if tok_embeddings is not None:
    fully_shard(tok_embeddings, **fsdp_config, reshard_after_forward=...)

for transformer_block in layers.values():
    fully_shard(transformer_block, **fsdp_config, reshard_after_forward=...)

if norm is not None and output is not None:
    fully_shard([norm, output], **fsdp_config, reshard_after_forward=...)

fully_shard(model, **fsdp_config)          # the root
```

Each `fully_shard` call creates one **communication group**: its parameters are
gathered and freed together. So the granularity choice is a direct trade:

```
too coarse (whole model in one group):
    one enormous all-gather → the full model is resident → no memory saving,
    and nothing to overlap with

too fine (every nn.Linear):
    hundreds of tiny all-gathers → α-dominated (topic 19 §2.1), and the
    prefetch pipeline has too little work per step to hide latency

just right: one TransformerBlock
    ~14 MB per group for our config — comfortably bandwidth-bound, and while
    block i computes, block i+1's all-gather is in flight
```

One `TransformerBlock` per group is the right answer for essentially every
transformer, and it works because the blocks are *identical and sequential* —
perfectly regular prefetching.

Note `fully_shard([norm, output])` takes a **list**: those two modules form one
group because they always execute back to back at the end, so gathering them
together is one collective instead of two.

The final `fully_shard(model)` wraps the root. It shards nothing new (the
children are already wrapped) but it is the object that owns the top-level
prefetch schedule and the root's own parameters, if any.

## 3. `reshard_after_forward` — the interesting default

```python
match reshard_after_forward_policy:
    case "always": reshard_after_forward = True
    case "never":  reshard_after_forward = False
    case "default":
        # For PP, by default do not reshard after forward to avoid per-microbatch
        # all-gathers, which can be expensive and non-overlapped
        reshard_after_forward = not pp_enabled
```

The trade:

```
True  → free params after forward, all-gather again in backward
        min memory, 2 all-gathers per step
False → keep the gathered params alive until backward
        1 all-gather per step, but a full layer's params resident throughout
```

Now the pipeline-parallel subtlety, which is the point of this section. Under PP
a stage runs its forward **`M` times** (once per micro-batch) before its
backwards. With `reshard_after_forward=True`:

```
F₁ [AG, compute, free]  F₂ [AG, compute, free]  …  F_M [AG, compute, free]
     └── M all-gathers of the same parameters, one per micro-batch ──┘
```

`M×` the communication for zero benefit — the parameters are identical every
time. Worse, these all-gathers sit on the critical path of the pipeline's fill
phase, where there is nothing to overlap them with.

So under PP, hold the parameters: **one all-gather serves all `M`
micro-batches.** The memory cost is one layer's parameters, which is small
compared to the `M` micro-batches of activations PP already holds.

This is a genuinely non-obvious interaction between two independent
parallelisms, and it is exactly the kind of thing that makes composed
parallelism hard: neither FSDP nor PP in isolation suggests this default.

### 3.1 The last-layers exception

```python
# As an optimization, do not reshard_after_forward the last layers by default
# since FSDP would prefetch them immediately after the forward pass
fully_shard([norm, output], **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always")
```

`norm` and `output` are the *last* modules in the forward pass and therefore the
*first* in the backward pass. Freeing them at the end of forward means
immediately re-gathering them nanoseconds later. Pointless — so keep them unless
the user explicitly asked for `"always"`.

A small optimization, and a good illustration of the general rule: **the cost of
resharding depends on how soon the parameter will be needed again.**

## 4. Mixed precision

```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=param_dtype,      # bf16 — compute dtype
    reduce_dtype=reduce_dtype,    # fp32 — gradient reduction dtype
    cast_forward_inputs=False,
)
```

Two separate dtypes, for two separate reasons:

- **`param_dtype = bf16`.** Parameters are stored sharded in fp32 (the master
  copy) and all-gathered *as bf16*, which also **halves the all-gather volume**.
  Compute runs on tensor cores at full rate.
- **`reduce_dtype = fp32`.** The gradient reduce-scatter accumulates in fp32.
  This matters: a bf16 reduction over `P = 512` ranks loses meaningful precision
  in the sum (bf16 has 8 mantissa bits, so accumulating 512 values of similar
  magnitude systematically underflows the tail). Reducing in fp32 costs 2× the
  gradient bytes and is worth it. Topic 19 §5.4 flagged this; here is where it
  is configured.

`cast_forward_inputs=False` means FSDP will not silently cast the module's
*inputs* — the model is expected to hand over correctly-typed activations
itself. Automatic casting is convenient and hides dtype bugs; this codebase
prefers the explicit `maybe_enable_amp` context in the trainer.

Recall from topic 18 §3.4 that the `fsdp` mesh dimension keeps a **real** backend
even at degree 1 precisely so this policy can be applied — mixed precision comes
in through the FSDP wrapper, not separately.

## 5. `disable_fsdp_gradient_division` — one invariant, three places

```python
def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """Disable FSDP's automatic gradient division for all FSDP modules.

    Sets gradient_divide_factor=1.0 so gradient scaling can be handled manually
    (e.g., by normalizing loss with global token count).
    """
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)
```

By default FSDP's reduce-scatter divides by the group size, implementing the
`1/P` of `∂L/∂θ = (1/P) Σ_p ...`. Here that is **disabled**, for exactly the
reason topic 11 §3.1 gave: dividing by `P` is only correct when every rank
contributed the same number of tokens.

This is the third component to make the same choice:

```
components/loss.py    reduction="sum"                 — do not average
pipeline_parallel.py  scale_grads=False               — do not divide by M
model_parallel.py     set_gradient_divide_factor(1.0) — do not divide by P
                              ↓
train.py              loss_sum / global_valid_tokens  — the ONE normalization
```

Four files, one invariant: **exactly one division, by the global valid token
count.** Any component that also normalizes would silently double-divide,
shrinking the effective learning rate by `P` or `M` — a bug that looks like
"training is mysteriously slow to converge".

If you take one architectural lesson from this topic, take this one: when
several layers of a framework each *could* normalize, the design must state
plainly which one *does*, and the others must be explicitly switched off with a
comment saying why.

## 6. HSDP: two-dimensional data parallelism

```python
if parallel_dims.dp_replicate_enabled:
    dp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])   # 2-D
else:
    dp_mesh = parallel_dims.get_mesh("fsdp")                     # 1-D
```

Give FSDP a **2-D** mesh and it shards along the inner dimension and replicates
along the outer:

```
                  fsdp (shard) →
   dp_replicate   ┌────┬────┬────┬────┐
        ↓         │ r0 │ r1 │ r2 │ r3 │   replica 0: params sharded 4 ways
                  ├────┼────┼────┼────┤
                  │ r4 │ r5 │ r6 │ r7 │   replica 1: same sharding
                  └────┴────┴────┴────┘
```

Communication becomes hierarchical:

```
all-gather / reduce-scatter   within a replica (P_sh ranks)   — often intra-node
all-reduce of grad shards     across replicas (P_dp ranks)    — once per step
```

**Why this beats flat FSDP at scale.** Flat FSDP over 512 ranks needs a ring
all-gather with `511` steps — pure `α`-cost (topic 19 §2.2), and every step
crosses the inter-node fabric. HSDP with `P_sh = 8, P_dp = 64` does its
all-gathers over 8 intra-node ranks (7 fast steps) and pays one cross-node
all-reduce per step, which overlaps with the backward pass.

The rule of thumb: **shard as widely as memory requires and no wider; replicate
the rest.**

```
P_sh = ceil(18 bytes/param × N / available memory per GPU)
P_dp = W / (P_sh · P_cp · P_tp · P_pp)
```

## 7. Explicit prefetch when EP is on

```python
# NOTE: set up explicit prefetching when EP is enabled, as D2H syncs in EP
# could interfere with implicit prefetching in FSDP
if ep_degree == 1:
    return

tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])
for block, next_block in zip(transformer_blocks, next_transformer_blocks):
    if next_block is not None:
        if next_block.moe_enabled:
            block.set_modules_to_forward_prefetch(
                [next_block, next_block.moe.experts])   # two groups!
        else:
            block.set_modules_to_forward_prefetch([next_block])
```

FSDP normally infers the prefetch order by *recording the execution order on the
first iteration*. Expert parallelism breaks that inference, because MoE routing
requires a **device-to-host synchronization** — the number of tokens per expert
is data-dependent and must reach the CPU to size the subsequent grouped matmul
(topic 29). A D2H sync stalls the CPU, so the CPU falls behind and stops issuing
the next layer's all-gather early enough to matter.

The fix is to state the schedule explicitly. Note the MoE case prefetches **two**
groups (`next_block` and `next_block.moe.experts`), because §8 wraps the routed
experts in their own FSDP group on a different mesh.

The backward direction is set up symmetrically, in reverse order.

This is a good example of a cross-cutting concern: a property of the *MoE
implementation* (a data-dependent shape) degrades a heuristic in a completely
different subsystem (FSDP's prefetcher), and the fix lives in a third place.

## 8. MoE experts get their own FSDP group

```python
# - the router and the shared experts are sharded together with the TransformerBlock
# - the routed experts are sharded with the dp_replicate_efsdp_mesh
if transformer_block.moe_enabled and ep_degree > 1:
    fsdp_mod_ep_config = fsdp_config.copy()
    fsdp_mod_ep_config["mesh"] = dp_replicate_efsdp_mesh      # a DIFFERENT mesh

    efsdp_degree = dp_replicate_efsdp_mesh["efsdp"].size()
    if efsdp_degree * ep_degree > moe.experts.num_experts:
        def _shard_on_dim1(_param): return Shard(1)
        _experts_shard_placement_fn = _shard_on_dim1

    fully_shard(moe.experts, **fsdp_mod_ep_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_experts_shard_placement_fn)
```

Two things here, both previewing Part IV.

**Different parameters, different mesh.** The routed experts are already sharded
across `ep`, so the ranks that hold *the same* expert weights are the `efsdp`
group, not the `fsdp` group. This is precisely why topic 18 built a separate
`sparse` mesh view — the same GPU belongs to different FSDP groups depending on
which parameter you are asking about.

**The `Shard(1)` fallback**, explained by the source comment:

> EP first shards the expert dimension, leaving each EP rank with
> `num_experts / ep_degree` local experts. If the EFSDP shard degree exceeds
> that local expert count, FSDP's default `Shard(0)` leaves some EFSDP ranks
> with empty or very small shards. Shard dim 1 instead so every EFSDP rank
> partitions the hidden dimension.

Expert weights are `[num_experts, hidden, dim]`. Default `Shard(0)` splits the
*expert* axis — but EP already did that, so with `E = 8, P_ep = 4` each rank has
2 local experts, and sharding 2 experts across `efsdp = 4` ranks leaves two
ranks with nothing. An empty shard is not merely inefficient: those ranks have no
gradient to contribute and the collective degenerates. Sharding dim 1 (the
hidden dimension, size 1408) divides evenly instead.

The condition `efsdp_degree * ep_degree > num_experts` is exactly "the expert
axis has run out of parallelism to give".

## 9. DDP, for completeness

```python
def apply_ddp(model, dp_mesh, enable_compile, enable_compiled_autograd):
    if enable_compile:
        torch._dynamo.config.optimize_ddp = (
            "python_reducer_without_compiled_forward" if enable_compiled_autograd
            else "ddp_optimizer")
    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)
```

`replicate` is FSDP2's DDP-equivalent (topic 12 §5). Note `bucket_cap_mb=100`
rather than the classic 25 — modern interconnects and larger models favour
bigger buckets (topic 12 §3.2).

And the guard in `parallelize_deepseekv3`:

```python
elif parallel_dims.dp_replicate_enabled:
    if parallel_dims.world_mesh.ndim > 1:
        raise RuntimeError("DDP has not supported > 1D parallelism")
    apply_ddp(model, parallel_dims.world_mesh, ...)
```

DDP is only reachable when *nothing else* is enabled. The moment you add any
other parallelism, `fsdp_enabled` is true (recall topic 18 §3.2: it is true if
`dp_shard > 1` **or** `cp > 1`) and you get FSDP with `dp_shard = 1` — same
communication pattern, but composable.

## 10. Activation checkpointing

The other memory lever. `apply_ac` is applied *before* FSDP:

```python
if job_config.activation_checkpoint.mode != "none":
    apply_ac(model, job_config.activation_checkpoint,
             model_compile_enabled=model_compile_enabled,
             op_sac_save_list=_op_sac_save_list, ...)
```

Three modes:

- **`none`** — save everything. `6N` FLOPs, maximum memory.
- **`full`** — save only block boundaries; recompute everything inside.
  `8N` FLOPs (topic 02 §3.4), minimum memory.
- **`selective`** — save the *expensive* ops, recompute the cheap ones. The
  sweet spot, and the list is what `_build_op_sac_save_list` builds:

```python
save_ops = {op.default for op in get_default_op_list().compute_intensive_ops}
extra_ops = [
    torch.ops.aten.linear.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops._c10d_functional.all_to_all_single.default,
    torch.ops.aten.max.default,
]
```

The principle: **save the output of anything whose recomputation is expensive;
recompute anything cheap.** Matmuls and attention are expensive — save them.
Elementwise ops, norms, and reshapes are cheap — recompute them.

The two `_c10d_functional` entries are the interesting ones. `reduce_scatter` and
`all_to_all` are **collectives**, and recomputing a collective means re-running
communication *and* requires every rank to reach it in the same order (topic 19
§5.1). Saving their outputs avoids both problems. This is a distributed-specific
addition to what would otherwise be a purely local decision.

## 11. Ordering, all together

```python
def parallelize_deepseekv3(model, parallel_dims, job_config):
    assert job_config.training.seq_len % parallel_dims.seq_len_divisor == 0

    if parallel_dims.tp_enabled:
        if n_heads % tp != 0: raise ValueError(...)
        apply_non_moe_tp(model, tp_mesh, loss_parallel=...)        # 1. TP

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(model, tp_mesh, ep_mesh, ep_etp_mesh, ...) # 2. EP/ETP

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(...)                                              # 3. AC

    if model_compile_enabled:
        apply_compile(model, job_config.compile)                   # 4. compile

    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        apply_fsdp(...)                                            # 5. FSDP
    elif parallel_dims.dp_replicate_enabled:
        apply_ddp(...)
```

The order is forced, innermost transformation first:

1. **TP/EP** rewrite parameters into DTensors. Everything after must tolerate
   DTensors.
2. **AC** wraps module *forwards*. It must see the final (TP-aware) forward, or
   recomputation would replay the un-parallelized version.
3. **Compile** must see the final forward too, including the AC wrapper.
4. **FSDP** wraps outermost, because it must intercept parameter access for
   *whatever* the module ended up being.

And recall from topic 17 §3 that all of this runs *inside* the per-stage loop, so
PP is outside even FSDP. The full nesting:

```
PP  (module tree cut)
 └─ FSDP  (parameter access)
     └─ compile
         └─ AC
             └─ TP / EP  (parameter layout)
                 └─ the model
```

Assertions worth noting: `seq_len % seq_len_divisor` (topic 18 §5) and
`n_heads % tp == 0` — TP shards attention by head, so heads must divide evenly
(topic 22).

## 12. Check yourself

1. Derive the ZeRO-1/2/3 memory table. For a 70B model at `P = 64`, what does
   each stage give per GPU?
2. Write out FSDP's seven steps per module and label each collective with its
   DTensor placement transition.
3. FSDP moves 1.5× DDP's bytes. Show the arithmetic and say why it is often free
   in wall-clock terms.
4. Why is one `TransformerBlock` the right FSDP granularity? Give the failure
   mode at each extreme.
5. Why does `fully_shard` take `[norm, output]` as a list?
6. Explain why `reshard_after_forward` defaults to `False` under PP. Quantify the
   waste if it were `True`.
7. Why are `norm`/`output` exempt from resharding, and what general rule does
   that illustrate?
8. Why does `reduce_dtype` differ from `param_dtype`? What goes wrong reducing
   gradients in bf16 at `P = 512`?
9. Name the four places that could normalize gradients and say which one does.
   What is the symptom if two of them do?
10. Draw the HSDP mesh and give the two collectives with their groups. Why does
    HSDP beat flat FSDP at `P = 512`?
11. Why must FSDP prefetch be explicit when EP is enabled? Trace the causal
    chain from MoE routing to a stalled all-gather.
12. Why do routed experts use `dp_replicate_efsdp_mesh`? Why does
    `efsdp_degree · ep_degree > num_experts` force `Shard(1)`?
13. Why does DDP raise on `world_mesh.ndim > 1`, and what is used instead?
14. Why are `reduce_scatter_tensor` and `all_to_all_single` in the selective-AC
    save list? Give both reasons.
15. State the full nesting order of PP, FSDP, compile, AC, TP and justify each
    adjacency.

## Next

→ [21 — Tensor parallelism from first principles](21-tensor-parallelism-first-principles.md)
