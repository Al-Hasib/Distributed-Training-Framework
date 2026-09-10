# 27 — Tensor Parallelism for MoE

> Video: 18:02:52 — Tensor Parallelism for MoE
> Code: `torchfeather/distributed/expert_parallel.py` (`TensorParallel`, `apply_moe_ep_tp`)

## Why this exists

Before distributing the *experts* (topic 28), distribute their *weights* — the
straightforward thing, since we already know how to tensor-parallelize a SwiGLU
FFN (topic 21 §2). Under `TensorParallel`, **every rank holds all `E` experts,
but each expert's matrices are sliced.**

The interesting content is not the sharding. It is the two **gradient
corrections** that MoE forces, both of which are instances of topic 01 §3's
warning that a tensor's gradient layout is not always its forward layout. Both
are one-line fixes for bugs that produce no error and no crash — only a model
that trains slightly wrong.

## 1. The sharding

```python
def _partition_fn(self, _name, module, device_mesh):
    # w1 shape = (experts, out_dim, in_dim)
    module.register_parameter("w1",
        nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)])))  # Column-wise

    # w2 shape = (experts, in_dim, out_dim)
    module.register_parameter("w2",
        nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)])))  # Row-wise

    # w3 shape = (experts, out_dim, in_dim)
    module.register_parameter("w3",
        nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)])))  # Column-wise
```

The expert weights carry a leading expert dimension:

```
w1 : [E, d_ff^E, d]      w2 : [E, d, d_ff^E]      w3 : [E, d_ff^E, d]
```

so the shard dimensions are shifted by one from the dense case:

| | logical role | shape | shard dim | dense equivalent |
|---|---|---|---|---|
| `w1` | gate, `d → d_ff` | `[E, d_ff, d]` | `Shard(1)` = `d_ff` | `ColwiseParallel` |
| `w3` | up, `d → d_ff` | `[E, d_ff, d]` | `Shard(1)` = `d_ff` | `ColwiseParallel` |
| `w2` | down, `d_ff → d` | `[E, d, d_ff]` | `Shard(2)` = `d_ff` | `RowwiseParallel` |

All three shard the **same logical axis** — the FFN hidden dimension `d_ff^E` —
which is exactly topic 21 §2: column-parallel into row-parallel, with the
intermediate `d_ff`-wide activation never gathered. The expert dimension is
untouched, so every rank still has all `E` experts.

Note the `nn.Parameter(...)` wrapping and `register_parameter`: these are raw
`nn.Parameter` tensors, not `nn.Linear` modules, so the standard
`ColwiseParallel`/`RowwiseParallel` styles (which look for `.weight`) do not
apply. Hence a bespoke style.

### 1.1 And why the grouped matmul needs `to_local`

```python
def forward(self, routed_input, num_tokens_per_expert):
    if isinstance(self.w1, DTensor):
        # Grouped MM operates on rank-local tensors, so extract each parameter's
        # local shard before invoking the kernel.
        w1, w2, w3 = self.w1.to_local(), self.w2.to_local(), self.w3.to_local()
```

`torch._grouped_mm` has no DTensor sharding rule — it is a fairly new kernel with
data-dependent group offsets. So `GroupedExperts.forward` drops to local tensors
and does the matmul on raw shards.

That is legitimate (each rank's slice of `w1` against the full `routed_input`
gives exactly that rank's slice of `h`, per topic 08 §2.1) but it has a
consequence: **autograd loses the placement information.** Which is the subject
of §2.

## 2. Correction 1: the input gradient is `Partial`

```python
def _prepare_input_fn(self, _mod, inputs, device_mesh):
    routed_input, num_tokens_per_expert = inputs
    # This round-trip is a forward no-op. The expert computation uses local tensors,
    # so DTensor cannot infer that each TP rank produces only a partial input gradient.
    # Mark it Partial so backward reduces it to Replicate across the TP ranks.
    routed_input = DTensor.from_local(
        routed_input, device_mesh, (Replicate(),)
    ).to_local(grad_placements=(Partial(),))
    return routed_input, num_tokens_per_expert
```

Work out what the gradient *should* be. `routed_input` is `Replicate()` — every
TP rank has the same tokens. From topic 11 §2.1:

```
forward: Replicate()  ⟹  backward: Partial(sum)
```

Each rank computes `∂L/∂routed_input` using only *its slice* of `w1`, `w2`, `w3`,
so each rank's gradient is a **partial sum**. The true gradient is their sum.

Normally DTensor would know this and insert the reduction. But §1.1 dropped to
local tensors for the matmul, so the graph autograd sees is a plain local
computation and it has no idea a reduction is owed.

The fix is the two-step round-trip:

```
DTensor.from_local(x, mesh, (Replicate(),))     declare: forward is replicated
        .to_local(grad_placements=(Partial(),)) declare: backward is Partial
```

**Forward: a no-op** (wrap then unwrap, no collective). **Backward: an
all-reduce**, because `Partial → Replicate` is an all-reduce (topic 19 §3).

Without it, `∂L/∂routed_input` on each rank is `1/P` of the truth (it accounts
for only one slice of the FFN's hidden units), so the gradient flowing back to
the router — and to every earlier layer through the residual — is wrong by a
factor that depends on `P_tp`. No error, no crash. Just a model that trains
worse the more TP you use.

```
┌──────────────────────────────────────────────────────────────┐
│  When you drop to local tensors for a kernel, you take over  │
│  responsibility for declaring the gradient's placement.      │
└──────────────────────────────────────────────────────────────┘
```

Remember this pattern — it is the general escape hatch for kernels DTensor does
not model, and it appears again in `ExpertTensorParallel` (topic 30 §3).

## 3. Correction 2: the gate gradient, when scoring happens after

The subtler of the two, and the code comment states it exactly:

```python
# When score_before_experts=False, the router gradient depends on the expert
# output, which is a TP-partial sum (different on each TP rank).
# Without correction the gate weights silently diverge across TP ranks,
# degrading loss for every TP config.
# Marking the local gate-output gradient as Partial lets DTensor reduce the
# per-rank contributions when reconciling them with replicated tensors:
#   sum_k(grad * partial_k) = grad * full_output
gate_grad_placements = (Partial(),) if not moe.score_before_experts else None

moe_layer_plan = {
    ...
    "moe.router.gate": NoParallel(output_grad_placements=gate_grad_placements),
}
```

### 3.1 The mechanism

With `score_before_experts=False` (topic 26 §7.4) the gate multiplies the expert
*output*:

```
y = g · E(x)          ⟹        ∂L/∂g = ⟨∂L/∂y , E(x)⟩
```

So the gate's gradient contains `E(x)` — the expert output. And under TP, `E(x)`
is a `Partial(sum)`: rank `k` holds `partial_k`, and
`E(x) = Σ_k partial_k`.

Therefore rank `k` computes

```
(∂L/∂g)_k = ⟨∂L/∂y , partial_k⟩
```

which is **different on every rank**, and the true gradient is the sum:

```
Σ_k ⟨∂L/∂y, partial_k⟩ = ⟨∂L/∂y, Σ_k partial_k⟩ = ⟨∂L/∂y, E(x)⟩   ✓
```

(the identity in the comment). So the gate's gradient is `Partial` and must be
all-reduced.

### 3.2 Why "silently diverge" is the right description

`router.gate` is `NoParallel` — a **replicated** parameter. The invariant of a
replicated parameter is that every rank holds the identical value, which requires
identical gradients. Feed it different gradients and:

- each rank's optimizer applies a different update,
- the "replicated" weight is no longer replicated,
- the routing decision differs across TP ranks,
- so ranks that are supposed to be computing one logical forward pass route
  tokens to different experts.

Nothing checks this. Nothing crashes. The model just trains worse, by an amount
that grows with `P_tp` — and if you only ever test at `P_tp = 1` you will never
see it.

The fix declares the gradient's placement:

```python
NoParallel(output_grad_placements=(Partial(),))
```

which — since the parameter is `Replicate()` — makes DTensor all-reduce the
gradient before the optimizer sees it, restoring the invariant.

And note it is conditional: with `score_before_experts=True` the gate multiplies
the *input* instead, `∂L/∂g` involves `x` (which is `Replicate()`, not
`Partial`), and no correction is needed. `gate_grad_placements = None` in that
case. **The correct gradient placement depends on a modelling flag.**

> If you take one thing from this topic: **a replicated parameter with
> non-identical gradients is a silent correctness bug, and the assertion is
> cheap.** Add it to your test suite:
> ```python
> g = [torch.zeros_like(gate.weight.grad) for _ in range(tp_mesh.size())]
> dist.all_gather(g, gate.weight.grad, group=tp_mesh.get_group())
> assert all(torch.allclose(x, g[0]) for x in g)
> ```

## 4. The MoE module boundary

```python
"moe": PrepareModuleInputOutput(
    input_layouts=(Shard(1),),          # sharded because coming from SP region
    desired_input_layouts=(Replicate(),),  # we use full sequence in TP region
    use_local_input=True,               # module receives a plain local tensor
    output_layouts=(Partial(),),        # partial: still needs an all-reduce
    desired_output_layouts=(Shard(1),), # sharded because going to another SP region
),
```

Exactly the dense FFN's boundary (topic 22 §3.6): all-gather in, reduce-scatter
out, the sequence-parallel sandwich. `Shard(1) → Replicate` then
`Partial → Shard(1)`.

`use_local_input=True` is the notable one, and it pairs with the very first line
of `MoE.forward`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, DTensor):
        raise TypeError("A plain tensor is expected")
```

The MoE forward pass does `histc`, `argsort`, `gather`, `scatter_add`, and a
grouped matmul with data-dependent offsets. Several of those have no DTensor
sharding rule, and a DTensor arriving here would fail deep inside — or worse,
dispatch to something plausible. So the module **refuses** DTensors at its
boundary and the plan promises to hand it a local tensor.

An explicit contract, enforced by an assertion, at the boundary between the
DTensor world and hand-written index arithmetic. Worth copying whenever you write
a module that cannot participate in the type system.

## 5. Shared experts

```python
"moe.shared_experts.w1": ColwiseParallel(),
"moe.shared_experts.w2": RowwiseParallel(output_layouts=Partial()),
"moe.shared_experts.w3": ColwiseParallel(),
```

An ordinary `FeedForward`, so the ordinary dense plan applies — with one
difference from topic 22 §3.6: `output_layouts=Partial()` rather than
`Shard(1)`.

Why: the shared expert's output is added to the routed experts' output
(topic 26 §7.2's `scatter_add`), and *that* sum is what the `moe` module's
output hook reduce-scatters. Reducing the shared expert separately would be a
wasted collective. Leaving it `Partial` lets the two partial sums combine
locally, and **one** `Partial → Shard(1)` reduce-scatter finishes the job for
both.

Partial placements compose additively, so deferring the reduction until after all
the additions is always the right move. A small thing worth noticing: it is the
same reasoning that makes `Partial → Shard` better than "all-reduce then slice"
(topic 19 §3).

## 6. What TP does and does not solve for MoE

```
memory per rank for expert weights  =  E · 3 · d · d_ff^E / P_tp
                                       ↑
                                       still linear in E
```

TP divides expert memory by `P_tp`, and `P_tp ≤ min(n_heads, node size)` — call
it 8. But the whole point of MoE is to make `E` large: at `E = 256`,
`d = 7168`, `d_ff^E = 2048` the routed experts alone are

```
256 · 3 · 7168 · 2048 · 2 bytes = 22.5 GB per layer
```

`/8` is still 2.8 GB per layer, times 60 layers. TP cannot get there.

```
┌──────────────────────────────────────────────────────────────┐
│  TP shards each expert's weights but keeps all E experts     │
│  on every rank. Memory stays O(E).                           │
│  Expert parallelism shards the expert dimension itself,      │
│  making memory O(E / P_ep). That is topic 28.                │
└──────────────────────────────────────────────────────────────┘
```

The reference code selects between them explicitly:

```python
if ep_mesh is None:
    # TP only. Every rank holds all experts, but each expert's weights are
    # sliced (col/row-wise) across TP ranks.
    experts_mesh, experts_plan = tp_mesh, TensorParallel()
elif tp_mesh is None or not etp_enabled:
    # EP only or EP+TP, but no ETP. Experts are split across EP ranks;
    # each expert's weights stay whole.
    experts_mesh, experts_plan = ep_mesh, ExpertParallel()
else:
    # ETP. Experts split across EP ranks, and each expert's weights are
    # further sliced across the TP ranks in the same EP group.
    experts_mesh, experts_plan = ep_mesh, ExpertTensorParallel(ep_etp_mesh=ep_etp_mesh)
```

Three regimes, three styles, one dispatch. Topics 28 and 30 take the second and
third.

## 7. Check yourself

1. Give the shard dimension for `w1`, `w2`, `w3` and the dense-FFN style each
   corresponds to. Which logical axis do all three shard?
2. Why can't `ColwiseParallel` be used directly on the expert weights?
3. Why does `GroupedExperts.forward` call `to_local()` on its parameters, and
   what does autograd lose as a result?
4. Explain the `from_local(...).to_local(grad_placements=(Partial(),))`
   round-trip: what happens in the forward pass, and what in the backward?
5. What is `∂L/∂routed_input` off by without that correction, and why is the bug
   invisible at `P_tp = 1`?
6. Derive `∂L/∂g` for `y = g · E(x)` and show why it is `Partial` under TP.
   Verify the identity `Σ_k ⟨ḡ, partial_k⟩ = ⟨ḡ, E(x)⟩`.
7. Why does the code say the gate weights "silently diverge"? Trace the four
   steps from a wrong gradient to a wrong forward pass.
8. Why is `gate_grad_placements` `None` when `score_before_experts=True`?
9. Write the assertion that would catch a diverged replicated parameter.
10. Why does `MoE.forward` raise `TypeError` on a DTensor input, and which plan
    argument satisfies it?
11. Why do the shared experts output `Partial()` instead of `Shard(1)`?
12. Compute expert-weight memory for `E=256, d=7168, d_ff^E=2048, P_tp=8` and
    explain why TP is insufficient.

## Next

→ [28 — Expert parallelism](28-expert-parallelism.md)
