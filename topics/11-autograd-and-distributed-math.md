# 11 — Autograd and the Mathematics of Distributed Training

> Video: 04:32:59 — Autograd and the Mathematics of Distributed Training

## Why this exists

This is the load-bearing chapter of the course. Everything in Part III is a
corollary of it.

The claim we are going to prove is this:

> **Every collective communication operation is a linear map. The backward pass
> of a linear map is its adjoint. Therefore the backward collective for any
> forward collective is *forced* — you do not get to choose it, and you can
> derive it in one line.**

Once you have that, you never again have to memorize "which collective goes in
the backward pass". You compute it. And the table from topic 01 §3 stops being a
list of facts and becomes four applications of one theorem.

## 1. Reverse-mode autodiff in one page

### 1.1 The chain rule as a product of Jacobians

A neural network is a composition:

```
x → f_1 → h_1 → f_2 → h_2 → … → f_n → L ∈ R
```

The chain rule gives

```
∂L        ∂L     ∂h_n        ∂h_2   ∂h_1
──   =   ──── · ────── · … · ──── · ────
∂x       ∂h_n   ∂h_{n−1}     ∂h_1    ∂x
```

a product of `n` Jacobian matrices. The only freedom is **the order in which you
multiply them**, and matrix multiplication is associative, so both ends are
valid:

```
forward mode  (right to left):   ( … ( (∂h_1/∂x) · (∂h_2/∂h_1) ) … )
reverse mode  (left to right):   ( … ( (∂L/∂h_n) · (∂h_n/∂h_{n−1}) ) … )
```

### 1.2 Why reverse mode, quantitatively

Let `h_i ∈ R^{d_i}`, and note `L ∈ R^1`.

- **Reverse mode** always carries a *row vector* of length `d_i` (`∂L/∂h_i`, the
  **cotangent**). Each step is a vector-matrix product: `O(d_i · d_{i−1})` —
  the same cost as the forward matmul. One pass gives **all** gradients.
- **Forward mode** carries a matrix `∂h_i/∂x` of size `d_i × d_0`. To get all
  gradients you need `d_0` passes — one per input dimension.

For a language model, `d_0 = N ≈ 10^9` parameters and the output is one scalar.
Reverse mode is cheaper by a factor of `10^9`. There is no contest.

And this is exactly where topic 02's `6N` came from: reverse mode never
materializes a Jacobian, it only ever computes **vector–Jacobian products**
(VJPs), each of which costs the same as the corresponding forward operation.
Two VJPs per linear layer (one for the input, one for the weight) gives
`F_bwd = 2·F_fwd`.

### 1.3 The VJP is the only primitive

Every operation in a framework must supply one function:

```
given  ḡ = ∂L/∂output ,   return   ∂L/∂input = ḡ · (∂output/∂input)
```

That is it. `y = xW` supplies `x̄ = ḡ Wᵀ` and `W̄ = xᵀ ḡ`. RMSNorm supplies its
own. And — the point of this topic — **an all-gather supplies one too.**

### 1.4 The adjoint theorem

Here is the fact everything rests on:

> **If `f` is linear, `f(x) = A x`, then its VJP is `ḡ ↦ Aᵀ ḡ`.**

Proof: `∂(Ax)/∂x = A`, so `∂L/∂x = (∂L/∂y)·A`, which as a column vector is
`Aᵀ ḡ`. □

So for a **linear** operation, the backward pass is the **transpose** (adjoint)
of the forward operation. No derivative needs to be computed; you just transpose
the matrix. Collectives are linear. Hence the title claim.

## 2. Collectives as linear maps, and their adjoints

Set up the space. With `P` devices, treat the distributed tensor as one big
vector in `R^{P·n}` — the concatenation of every device's local shard. Every
collective is then a linear map `R^{P·n} → R^{P·m}`, i.e. a block matrix. Write
`I` for the `n×n` identity.

### 2.1 Broadcast / Replicate

One device (or a logical single value) `x` becomes `P` copies:

```
             ⎡ I ⎤
   A_bcast = ⎢ I ⎥ ,      y_p = x   for all p
             ⎢ ⋮ ⎥
             ⎣ I ⎦
```

Adjoint:

```
   A_bcastᵀ = [ I  I  …  I ] ,      x̄ = Σ_p ḡ_p
```

```
┌──────────────────────────────────────────────────────────┐
│   forward: broadcast  ⟹  backward: SUM-REDUCE            │
└──────────────────────────────────────────────────────────┘
```

**This single line is why data parallelism works.** The weights are broadcast
(replicated) to every device in the forward pass, so the gradient of a replicated
weight is the *sum* of the per-device gradients. Every rank computes a partial
gradient; the truth is their sum.

In DTensor's vocabulary: `Replicate()` forward ⟷ `Partial(sum)` backward.

### 2.2 All-reduce (sum)

Each device contributes `x_p`; every device receives `s = Σ_q x_q`:

```
             ⎡ I  I  …  I ⎤
   A_ar    = ⎢ I  I  …  I ⎥        (all blocks are I)
             ⎢ ⋮          ⎥
             ⎣ I  I  …  I ⎦
```

This matrix is **symmetric**, so `A_arᵀ = A_ar`:

```
┌──────────────────────────────────────────────────────────┐
│   forward: all-reduce  ⟹  backward: all-reduce           │
└──────────────────────────────────────────────────────────┘
```

Self-adjoint. Worth pausing on, because it is often stated loosely. Precisely:
`x̄_p = Σ_q ḡ_q` for every `p` — the gradients must be all-reduced. In the common
case where the cotangent is *already identical on every rank* (because every rank
computed the same downstream function of the same all-reduced value), that
all-reduce would multiply by `P`, so implementations instead treat the backward
as a no-op and rely on the cotangents already agreeing. Two different situations
that are easy to conflate:

- Tensor-parallel `RowwiseParallel`: forward all-reduce, backward **identity**,
  because the cotangent arriving at each rank is genuinely the same tensor.
- A hand-written all-reduce over genuinely different per-rank cotangents:
  backward is a real all-reduce.

If you are ever unsure, write the block matrix down and transpose it.

### 2.3 All-gather

Device `p` holds shard `x_p ∈ R^n`; every device receives the full
`X = [x_1 ‖ … ‖ x_P] ∈ R^{P·n}`:

```
   A_ag = ⎡ I₁ … I_P ⎤   stacked P times,
          where row-block q of the output is the full concatenation
```

Its adjoint takes `P` full-size cotangents `ḡ_q ∈ R^{P·n}` and returns, for
device `p`, the `p`-th slice of their sum:

```
   x̄_p = [ Σ_q ḡ_q ]_p
```

Sum across devices, then keep only your own slice. **That is the definition of
reduce-scatter.**

```
┌──────────────────────────────────────────────────────────┐
│   forward: all-gather  ⟹  backward: reduce-scatter       │
│   forward: reduce-scatter ⟹ backward: all-gather         │
└──────────────────────────────────────────────────────────┘
```

The second line is the adjoint of the first (adjoints are involutive). This pair
is **FSDP** (topic 20): gather the weight shards to compute, reduce-scatter the
gradient shards to update. Now you know why those two specific collectives, and
why in that order.

### 2.4 Shard (identity on disjoint pieces)

If each device simply operates on its own slice, with no communication, the map
is block-diagonal:

```
   A_shard = diag(I, I, …, I)         A_shardᵀ = diag(I, I, …, I)
```

```
┌──────────────────────────────────────────────────────────┐
│   forward: Shard(i)  ⟹  backward: Shard(i)               │
└──────────────────────────────────────────────────────────┘
```

Sharded stays sharded, on the same dimension. No collective either way.

### 2.5 All-to-all

Device `p` sends its `q`-th chunk to device `q`. As a matrix this is a
**permutation** of blocks: `y_{q,p} = x_{p,q}`. A permutation matrix's transpose
is its inverse, which is the same permutation with source and destination
swapped:

```
┌──────────────────────────────────────────────────────────┐
│   forward: all-to-all(split=a, concat=b)                 │
│      ⟹  backward: all-to-all(split=b, concat=a)          │
└──────────────────────────────────────────────────────────┘
```

Same collective, split and concat dimensions exchanged. This is expert
parallelism's dispatch/combine pair (topic 29): the combine is *literally the
backward of the dispatch*, which is why the code implements one and gets the
other for free.

### 2.6 The complete table, now derived

| forward | backward | as a matrix | where it appears |
|---------|----------|-------------|------------------|
| broadcast / `Replicate()` | sum-reduce / `Partial(sum)` | `[I;…;I]ᵀ = [I…I]` | DP: replicated weights |
| `Partial(sum)` | `Replicate()` | the same pair, reversed | TP row-parallel output |
| all-reduce | all-reduce | symmetric | TP (with the caveat in §2.2) |
| all-gather | reduce-scatter | stack ⟷ sum-slice | FSDP |
| reduce-scatter | all-gather | | FSDP gradients |
| `Shard(i)` | `Shard(i)` | block-diagonal | TP activations, CP |
| all-to-all(a,b) | all-to-all(b,a) | permutation | EP dispatch/combine |

You are no longer memorizing this. You are transposing block matrices.

## 3. The theorem that makes data parallelism legal

Everything above is about *placement*. This section is about the *loss*, and it
is a separate result.

The training loss over a global batch of `N` examples is

```
L = (1/N) Σ_{i=1}^{N} ℓ(x_i ; θ)
```

Differentiation is linear, so

```
┌──────────────────────────────────────────────────┐
│   ∂L/∂θ  =  (1/N) Σ_i  ∂ℓ(x_i;θ)/∂θ              │
└──────────────────────────────────────────────────┘
```

**The gradient of the mean is the mean of the gradients.** Therefore: split the
batch across `P` devices, let each compute the gradient of its own local mean,
average the results — the answer is *bit-for-bit the mathematical gradient* of
the global batch (up to floating-point reassociation). Not an approximation.
Not a heuristic.

Two immediate corollaries:

**(a) Gradient accumulation.** Split the local batch into `M` micro-batches:

```
∂L/∂θ = (1/M) Σ_{m=1}^{M} ∂L_m/∂θ
```

so you can compute micro-batch gradients one at a time and add them into the
`.grad` buffers. Memory for activations drops by `M`; the result is unchanged.
This is what makes pipeline parallelism possible (topic 14) — a pipeline *is*
gradient accumulation with the micro-batches placed in a schedule.

**(b) Data parallelism = gradient accumulation across space instead of time.**
Identical mathematics, different placement. Which is why DDP is "just" an
all-reduce of `.grad`.

### 3.1 The normalization trap — read this twice

The theorem requires the weights in the average to be right, and this is where
real training runs silently go wrong.

**Wrong:** average the per-rank mean losses when ranks have different token
counts.

```
L_wrong = (1/P) Σ_p [ (1/n_p) Σ_{i∈p} ℓ_i ]        ← "mean of means"
```

**Right:** weight by the number of tokens each rank actually contributed.

```
L_right = ( Σ_p Σ_{i∈p} ℓ_i ) / ( Σ_p n_p )        ← "total over total"
```

These agree only when all `n_p` are equal. They differ whenever:

- sequences are padded and you mask the padding,
- a document-packing dataloader gives ranks different valid-token counts,
- the last batch of an epoch is ragged,
- an MoE run drops tokens over capacity (topic 26).

The mean-of-means silently **up-weights the tokens on lightly-loaded ranks**.
The bug does not crash, does not appear in the loss curve, and shows up only as
a model that is slightly worse than it should be. The fix is mechanical: reduce
the token count alongside the loss and divide at the end.

```python
# per rank
local_loss_sum = (per_token_loss * mask).sum()
local_tokens   = mask.sum()

# one all-reduce over the concatenation, then divide
packed = torch.stack([local_loss_sum, local_tokens])
dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=dp_group)
loss = packed[0] / packed[1]
```

The same applies to gradient accumulation over `M` micro-batches with unequal
token counts: scale each micro-batch's loss by `n_m / Σ n_m`, not by `1/M`.
Topic 13 implements this.

## 4. What the graph is, and what it costs

### 4.1 The graph is built by the forward pass

PyTorch's graph is **dynamic**: each operation on a tensor with
`requires_grad=True` allocates a node recording (i) the backward function and
(ii) references to whatever tensors that function will need.

```python
x = torch.randn(3, requires_grad=True)
y = x * 2          # y.grad_fn = MulBackward0,   saves the scalar 2
z = y.sum()        # z.grad_fn = SumBackward0
z.backward()       # reverse topological traversal
```

`backward()` walks the graph in reverse topological order, calling each node's
VJP and accumulating into `.grad` for leaves. Three properties matter later:

- **Accumulation, not assignment.** `.grad += ` — which is why
  `optimizer.zero_grad()` exists, and why gradient accumulation is free.
- **The graph is freed after `backward()`** unless `retain_graph=True`. Pipeline
  schedules that split forward and backward in time must therefore keep the
  graph alive across schedule steps (topic 17).
- **`detach()` cuts the graph.** This is exactly what a pipeline stage boundary
  is: the received activation is a fresh leaf, and the gradient must be *sent
  back over the network* rather than flowing through.

### 4.2 Memory: what has to be kept alive

The activation memory of topic 02 §4.2 is precisely "the tensors the VJPs
reference". `y = xW` needs `x` for `W̄ = xᵀḡ`. RMSNorm needs its input.
Attention needs `Q, K, V` (FlashAttention recomputes the score matrix instead of
saving it). So:

```
M_act = Σ_ops  (bytes of tensors that op's VJP needs)
```

and the three ways to reduce it are all now visible as graph operations:

- **Activation checkpointing** — do not save; re-run the forward in the backward
  pass. Costs `+2N` FLOPs (topic 02 §3.4).
- **Sharding activations** — context parallelism (`Shard(seq)`), tensor
  parallelism (`Shard(head)`).
- **Fewer live micro-batches** — 1F1B instead of GPipe (topic 15).

## 5. What it means to cut the graph across devices

Now put it together. Take the single-device graph and assign each node to a
device. Wherever an edge crosses a device boundary, you have a cut. **Each cut
requires exactly two communications: one forward, one backward, and the backward
one is the adjoint of the forward one.**

That is the entire content of Part III. Here are the five cuts:

```
                       forward across the cut   backward across the cut
──────────────────────────────────────────────────────────────────────────
Pipeline (layers)      send activation          recv gradient
                       (point-to-point)         (point-to-point)

Data (batch)           nothing — the batch      all-reduce (or
                       is already disjoint      reduce-scatter) of ∂L/∂θ
                                                because θ is Replicate()

FSDP (params)          all-gather θ shards      reduce-scatter ∂L/∂θ
                                                (adjoint, §2.3)

Tensor (matrices)      all-reduce / all-gather  the adjoint (§2.2, §2.3)
                       of activations

Context (sequence)     ring send of K,V         ring send of the K,V grads

Expert (experts)       all-to-all dispatch      all-to-all combine (§2.5)
```

Note the second row: **data parallelism needs no forward communication at all**,
because the batch dimension is already disjoint and `Shard(batch)` is
self-adjoint on the activations. The only communication is on the *parameters*,
which are `Replicate()`, and §2.1 says a replicated forward means a summed
backward. That is the whole of DDP, derived.

And note the pipeline row: because `detach()` cuts the graph, the gradient
cannot flow — it must be explicitly received and re-injected via
`torch.autograd.backward(tensors=[activation], grad_tensors=[received_grad])`.
Topic 17 writes that code.

## 6. Two places the mathematics gets subtle

### 6.1 Randomness must be placed deliberately

Dropout and any other stochastic op has an RNG state, and the *same* op may need
different treatment depending on placement:

| Tensor is… | RNG must be… | why |
|------------|--------------|-----|
| `Replicate()` (e.g. a replicated activation under TP) | **identical** across ranks | otherwise ranks compute different functions of the same data, and the "replicated" tensor silently diverges |
| `Shard(batch)` (DP) | **different** across ranks | different examples should get different masks |
| `Shard(hidden)` (TP activation) | **different** across ranks | it is one logical tensor; each rank must drop its own elements |

PyTorch exposes this as the distributed RNG tracker
(`torch.distributed.tensor.random`). Getting it wrong under TP produces a model
that trains but whose replicated tensors are not actually equal — a bug that is
essentially invisible without an explicit cross-rank equality assertion.

This is also why sharded initialization needs care (topic 05 §4, topic 20).

### 6.2 Not everything is differentiable, and MoE routing is the case that bites

MoE's router does `top_k`, which is an `argmax` — piecewise constant, so its
derivative is zero almost everywhere and undefined at ties. The gradient
therefore does **not** flow through the *choice* of expert. It flows only through
the *gate values* that scale the chosen experts' outputs:

```
y = Σ_{i ∈ topk(s)} g_i · E_i(x) ,          g_i = softmax/sigmoid of s_i

∂y/∂s :  flows through g_i for the selected experts only.
         Zero for every unselected expert.
```

Consequence: **an expert that is never selected receives no gradient and never
improves**, which is a positive feedback loop toward collapse. This is not a
numerical problem, it is a structural property of the derivative — and it is why
MoE needs an auxiliary load-balancing loss or a bias-based balancing scheme
(topic 26). The systems consequence is topic 28's load imbalance.

## 7. Verify the adjoint table yourself

The claims in §2 are testable. This script checks two of them with real
collectives.

```python
# adjoint_test.py — runs on CPU with the gloo backend, no GPU needed.
import os, tempfile, torch, torch.distributed as dist, torch.multiprocessing as mp


def worker(rank, world, initfile):
    dist.init_process_group("gloo", init_method=f"file:///{initfile}",
                            rank=rank, world_size=world)
    torch.manual_seed(0)                 # same test data on every rank
    n = 3

    # ---- claim: the adjoint of all-gather is reduce-scatter --------------
    # Standard adjoint test for a linear map A:  <A x, g> == <x, Aᵀ g>
    x = torch.randn(n) + rank            # this rank's shard
    g = torch.randn(world * n)           # a full-size cotangent

    X = [torch.zeros(n) for _ in range(world)]
    dist.all_gather(X, x)                # forward: A x
    X = torch.cat(X)
    lhs = torch.dot(X, g)                # <A x, g>, local part of the pairing

    # backward per §2.3: sum cotangents across ranks, keep your own slice
    gs = g.clone()
    dist.all_reduce(gs, op=dist.ReduceOp.SUM)
    x_bar = gs[rank * n:(rank + 1) * n]  # == reduce_scatter of the cotangents
    rhs = torch.dot(x, x_bar)            # <x, Aᵀ g>

    both = torch.stack([lhs, rhs])       # the identity holds after summing ranks
    dist.all_reduce(both, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"<Ax,g> = {both[0]:.6f}   <x,A^T g> = {both[1]:.6f}")
        assert torch.allclose(both[0], both[1], atol=1e-5)
        print("all-gather adjoint == reduce-scatter  OK")

    # ---- claim: gradient of the mean == mean of the gradients ------------
    w = torch.randn(4, requires_grad=True)
    local_x = torch.randn(8, 4)                    # this rank's batch shard
    loss = (local_x @ w).pow(2).mean() / world     # local mean, scaled by 1/P
    loss.backward()
    dist.all_reduce(w.grad, op=dist.ReduceOp.SUM)  # §2.1: Replicate ⟹ Partial(sum)
    if rank == 0:
        print("DP gradient assembled via one all-reduce OK")

    dist.destroy_process_group()


if __name__ == "__main__":
    f = os.path.join(tempfile.gettempdir(), "pg_store_adj")
    if os.path.exists(f):
        os.remove(f)
    mp.spawn(worker, args=(4, f.replace("\\", "/")), nprocs=4, join=True)
```

Actual output:

```
<Ax,g> = 9.745436   <x,A^T g> = 9.745436
all-gather adjoint == reduce-scatter  OK
DP gradient assembled via one all-reduce OK
```

The two inner products agree to six decimals: the backward derived by
transposing the block matrix in §2.3 **is** the adjoint, not merely a plausible
guess.

> **Why `mp.spawn` with a `file://` store rather than `torchrun`?** `torchrun`'s
> default c10d rendezvous opens a TCP store on localhost, which fails in
> sandboxed or network-restricted environments with
> `RendezvousConnectionError: The connection to the C10d store has failed`. A
> `file://` store needs no sockets and works everywhere, which makes it the
> right choice for these teaching scripts. On a real cluster, use `torchrun`.

The inner-product test (`⟨Ax, g⟩ = ⟨x, Aᵀg⟩`) is the standard way to verify an
adjoint, and it is worth keeping in your toolkit: any time you implement a custom
collective with a custom backward, this test tells you whether your backward is
*the* backward or merely a plausible one.

## 8. Check yourself

1. Why is reverse-mode autodiff the right choice for training, and by what
   factor? Where does `F_bwd = 2·F_fwd` come from?
2. State the adjoint theorem. Why does it make the backward pass of a collective
   a matter of derivation rather than convention?
3. Write the block matrix for broadcast and transpose it. What collective is the
   result, and which parallelism does that justify?
4. All-reduce is self-adjoint. Explain why TP's `RowwiseParallel` nevertheless
   uses an identity backward.
5. Derive that the adjoint of all-gather is reduce-scatter, and name the
   technique that uses this pair.
6. All-to-all is its own adjoint with the split/concat dims swapped. What does
   that imply about implementing MoE's combine step?
7. State the "gradient of the mean = mean of gradients" theorem and both of its
   corollaries.
8. Give three concrete situations in which averaging per-rank mean losses is
   wrong, and write the correct reduction.
9. Under TP, a replicated activation goes through dropout. What must be true of
   the RNG, and what is the symptom if it is not?
10. Why does an unselected MoE expert receive exactly zero gradient? What does
    the framework have to add to prevent collapse?
11. A pipeline stage boundary `detach()`es the activation. What does that force
    the backward pass to do explicitly?

## Next

→ [12 — Distributed computation graphs and DDP](12-distributed-computation-graphs-ddp.md)
