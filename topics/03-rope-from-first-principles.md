# 03 — RoPE from First Principles

> Video: 00:51:57 — RoPE from First Principles
> Code: `torchfeather/model/rope.py`

## Why this exists

Self-attention is **permutation-equivariant**. Feed it a shuffled sequence and
you get the shuffled output — the mechanism literally cannot see order. Proof in
one line: attention computes

```
out_m = Σ_n softmax_n( ⟨q_m, k_n⟩ / √d ) · v_n
```

and nothing in that expression references `m` or `n` except as an index into the
sums. Permute the inputs, and every term is still present, just relabelled.

So position must be injected. This topic derives *the* way modern LLMs do it,
starting from a requirement rather than from the answer.

Position encoding also turns out to be a **distributed-training problem**, which
is the real reason it appears this early in the course: once you shard the
sequence across devices (context parallelism, topic 23), each device holds a
*slice* of positions and must apply the rotation for its *global* offset, not
its local index. Get that wrong and the model trains — badly, silently. You
cannot debug that without knowing exactly what RoPE computes.

## 1. What we want: state the requirement, not the solution

Before the additive/rotary debate, notice *where* position actually needs to
matter. The only place the tokens interact is the dot product `⟨q_m, k_n⟩`.
Everything else in a transformer is applied per-token. So the requirement should
be stated about the dot product.

We want a function `f(x, m)` — inject position `m` into a vector `x` — such that

```
┌──────────────────────────────────────────────────┐
│   ⟨ f(q, m), f(k, n) ⟩  =  g(q, k, n − m)        │   (R)
└──────────────────────────────────────────────────┘
```

for some function `g`. In words: **after injection, the attention score depends
on the two contents and on their relative distance — and on nothing else.**

Why that requirement is the right one:

- **Relative, not absolute.** "The token three words back" is a linguistically
  meaningful relationship; "the token at index 4,096" is not. Requiring
  dependence on `n − m` builds translation invariance into the score.
- **No extra memory or FLOPs.** `g` is realized by transforming `q` and `k`
  *individually*, i.e. `O(S·d)` work, not by building an `S × S` bias matrix
  (`O(S²)` memory, which is exactly what T5-style relative bias costs and what
  makes it awkward with FlashAttention).
- **Nothing else changes.** Attention kernels keep working; the softmax,
  masking, and the `V` path are untouched.

### 1.1 Why the obvious answer fails

The original transformer used *additive absolute* encodings:
`x_m ← x_m + p_m`. Expand the score:

```
⟨(x_m + p_m) W_Q , (x_n + p_n) W_K⟩
   = x_m W_Q W_Kᵀ x_nᵀ      content–content
   + x_m W_Q W_Kᵀ p_nᵀ      content–position   ← unwanted
   + p_m W_Q W_Kᵀ x_nᵀ      position–content   ← unwanted
   + p_m W_Q W_Kᵀ p_nᵀ      position–position
```

Even the last term does not reduce to a function of `n − m` in general, and the
two cross terms mean "what this token means" gets entangled with "where it is".
It works, but requirement (R) is violated, and empirically these models
extrapolate poorly. So: reject, and solve (R) properly.

## 2. Solving the requirement in 2 dimensions

Start with `d = 2`; the general case will be built from copies.

### 2.1 Restrict the search: `f` should be an isometry

Try `f(x, m) = R_m x` for some matrix `R_m` depending only on the position. Then

```
⟨R_m q, R_n k⟩ = qᵀ R_mᵀ R_n k
```

For this to be a function of `n − m` only, we need

```
R_mᵀ R_n = R_{n−m}                                    (R')
```

Setting `n = m` gives `R_mᵀ R_m = R_0 = I`: **`R_m` must be orthogonal.** That
is a pleasant consequence, not an assumption — it means `f` preserves norms, so
injecting position cannot change the magnitude of a query or key, hence cannot
change how "confident" a token is. Combined with `R_mᵀ = R_m^{-1}`, (R') says

```
R_m R_n = R_{m+n},     R_0 = I
```

i.e. `m ↦ R_m` is a **one-parameter group homomorphism into O(2)**.

### 2.2 Solve the group equation

The connected one-parameter subgroups of `O(2)` are exactly the rotations
(reflections have determinant `−1` and cannot form a connected subgroup
containing `I`). Writing the rotation by angle `θ`:

```
        ⎡ cos mθ   −sin mθ ⎤
R_m  =  ⎢                  ⎥
        ⎣ sin mθ    cos mθ ⎦
```

Check (R') directly:

```
R_mᵀ R_n = R_{−m} R_n = R_{n−m}   ✓
```

**The rotation is not a clever trick — it is the unique solution to
requirement (R) among linear isometries of the plane.** That is the whole idea
of RoPE.

### 2.3 The same thing in one line, with complex numbers

Identify `x = (x_0, x_1) ∈ R²` with `z = x_0 + i x_1 ∈ C`. Rotation by `mθ` is
multiplication by `e^{imθ}`, and the real inner product is
`⟨a, b⟩ = Re(ā · b)`. So with `f(z, m) = z e^{imθ}`:

```
⟨f(q,m), f(k,n)⟩ = Re( conj(q e^{imθ}) · k e^{inθ} )
                 = Re( q̄ k · e^{i(n−m)θ} )
```

Depends on `q`, `k`, and `n − m`. Requirement (R) satisfied, with

```
g(q, k, δ) = Re( q̄ k e^{iδθ} )
```

This complex form is not just elegant — it is **literally the implementation**.
`torchfeather` does exactly this:

```python
x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
y = torch.view_as_real(x * freqs_cis).flatten(3)
```

One complex multiply per pair of dimensions. Nothing else.

## 3. Generalizing to `d` dimensions

A single `θ` in `d` dimensions would be a poor encoder: one angular frequency
can only express one scale of distance, and it wraps around with period
`2π/θ`. Instead, split the `d`-dimensional vector into `d/2` independent
2-D planes and give each plane its own frequency `θ_i`:

```
        ⎡ R_m(θ_0)                        ⎤
        ⎢          R_m(θ_1)               ⎥
R_m  =  ⎢                    ⋱            ⎥      (block diagonal, d × d)
        ⎣                      R_m(θ_{d/2−1}) ⎦
```

Block-diagonal orthogonal matrices are orthogonal, and (R') holds
block-by-block, so requirement (R) still holds exactly:

```
⟨R_m q, R_n k⟩ = Σ_{i=0}^{d/2−1} Re( q̄_i k_i e^{i(n−m)θ_i} )
```

The score is a **sum of `d/2` sinusoids in the relative distance** — a Fourier
series in `n − m`, whose coefficients are set by the content. That is a
remarkably expressive relative-position model for zero extra parameters.

### 3.1 Choosing the frequencies

RoPE inherits the geometric schedule from the sinusoidal encodings:

```
┌───────────────────────────────────────────────┐
│   θ_i = base^(−2i/d),    i = 0 … d/2 − 1      │
│   base = 10000  (typically)                   │
└───────────────────────────────────────────────┘
```

In code, exactly:

```python
freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
```

(`torch.arange(0, dim, 2)/dim` gives `2i/d`, so `freqs[i] = base^(−2i/d)`. ✓)

Think in **wavelengths** — the number of token positions for a full rotation:

```
λ_i = 2π / θ_i = 2π · base^(2i/d)
```

For `d = 64`, `base = 10000`:

| `i` | `θ_i` | `λ_i` (tokens) | reads as |
|-----|-------|----------------|----------|
| 0 | 1.0 | 6.3 | "adjacent word" |
| 8 | 0.1 | 62.8 | "clause" |
| 16 | 0.01 | 628 | "paragraph" |
| 24 | 1.0e-3 | 6,283 | "chapter" |
| 31 | 1.33e-4 | 47,117 | "whole document" |

So RoPE is a **multi-scale positional clock**: fast hands resolve fine local
order, slow hands carry coarse long-range position. The slowest plane has
wavelength `2π·base^((d−2)/d)` — for `d = 64` that is 47,117 positions,
approaching `2π·base ≈ 62,832` as `d` grows. So at a 4k training length the
slowest hands have swept under 10% of a rotation: the model has never observed
the angles that positions beyond the training length would produce. That is the
mechanism behind RoPE's extrapolation failure, and the entire motivation for
YaRN (topic 04).

### 3.2 The pairing convention — a real-world footgun

Which coordinates form each 2-D plane? Two conventions are in use, and they are
**not interchangeable across checkpoints**:

**Interleaved** (original RoPE paper, DeepSeek, `torchfeather`):
pairs are `(x_0,x_1), (x_2,x_3), …`. This is what
`x.view(..., -1, 2)` + `view_as_complex` produces.

**Split-half / "rotate_half"** (HuggingFace Llama implementations):
pairs are `(x_i, x_{i + d/2})`, implemented as

```python
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)
q = q * cos + rotate_half(q) * sin
```

Both are the same operator up to a fixed permutation `P` of the coordinates
(`R_split = P R_interleaved Pᵀ`), so a model trained with either learns
equivalently — the permutation can be absorbed into `W_Q` and `W_K`. But if you
load weights trained with one convention into code using the other, the model
degrades in a way that looks like a subtle training bug rather than an obvious
crash. Whenever you port a checkpoint, check this first.

## 4. Properties worth knowing

**(a) Norm preservation.** `‖R_m x‖ = ‖x‖`. Position injection cannot rescale
logits. Contrast with additive PEs, which do.

**(b) It is relative bias, at absolute cost.** The score is a function of
`n − m` (an `S × S` worth of structure) but is produced by touching each token
once (`O(S·d)`). No `S²` bias tensor is materialized, which is what makes RoPE
compatible with FlashAttention and with the block-wise attention of ring
attention (topic 23).

**(c) Long-range decay.** Grouping the sum and applying an Abel-summation
argument (RoPE paper §3.4.3) shows that with the geometric frequency schedule,
the magnitude of `Σ_i Re(q̄_i k_i e^{iδθ_i})` tends to *decrease* with `|δ|`,
because the phases of the fast components decorrelate. This is a tendency
imposed by the frequency schedule, not a hard bound for arbitrary `q, k` — but
it is a sensible inductive bias: distant tokens interact less by default.

**(d) Applied to `Q` and `K` only, never `V`.** RoPE's whole justification is
requirement (R), a statement about the *score*. `V` never appears in a dot
product with a position-bearing vector; rotating it would just scramble the
values being averaged. Similarly it is applied **after** the `W_Q`/`W_K`
projections and **per head**.

**(e) Linearity.** `R_m` is linear, so it commutes with sums and can be folded
into other linear maps — the property topic 10 (MLA weight absorption) exploits,
and the property that *fails* for the RoPE'd part of MLA's keys, forcing the
"decoupled RoPE" design of topic 09. If you remember one forward-reference from
this chapter, make it this one.

**(f) Extrapolation failure.** At position `m > S_train`, the fast components
`mθ_i mod 2π` are in-distribution (they wrap constantly), but the slow
components reach angles never seen in training. Attention logits go
out-of-distribution and perplexity explodes. Fixes: position interpolation,
NTK-aware scaling, YaRN — topic 04.

## 5. Reference implementation, derived line by line

```python
# 1. Frequencies: θ_i = base^(−2i/d), one per 2-D plane.       shape [d/2]
freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

# 2. Positions m = 0 … S−1.                                    shape [S]
t = torch.arange(seqlen)

# 3. Angles m·θ_i for every (position, plane) pair.            shape [S, d/2]
freqs = torch.outer(t, freqs)

# 4. e^{i m θ_i} as a complex tensor of unit modulus.          shape [S, d/2]
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
```

`torch.polar(abs, angle) = abs · e^{i·angle}`, so step 4 is `1 · e^{i m θ_i}`.
This table is a **buffer, not a parameter** — precomputed once, shared by all
layers, never trained.

Application:

```python
def apply_rotary_emb(x, freqs_cis):
    dtype = x.dtype
    # [B, S, h, d]  →  [B, S, h, d/2] complex, pairing (x_0,x_1), (x_2,x_3), …
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    # broadcast over batch and heads: [1, S, 1, d/2]
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # one complex multiply == one 2-D rotation per plane
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)
```

Three details that matter:

- **`.float()`**: the angles and the rotation are computed in fp32 even when the
  model runs in bf16. bf16 has ~3 decimal digits of mantissa; accumulating
  positional phase error at position 16,000 in bf16 is a measurable quality
  loss. Cast back at the end.
- **`freqs_cis.view(1, S, 1, d/2)`**: broadcasts across batch and heads —
  position is a property of the sequence, identical for every head. Note this
  hard-codes the layout `[B, S, h, d]`. If your tensors are `[B, h, S, d]` this
  view is silently wrong; check the layout convention of your attention module.
- **`flatten(3)`**: `view_as_real` appends a size-2 dim, giving
  `[B, S, h, d/2, 2]`; flattening from dim 3 restores `[B, S, h, d]` with the
  interleaved pairing intact.

### 5.1 Verify the invariant numerically

Never trust a position encoding you have not tested. The property to test is
requirement (R) itself: the score must depend only on `n − m`.

```python
import torch

d, S = 64, 128
freqs = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
ang = torch.outer(torch.arange(S).float(), freqs)
cis = torch.polar(torch.ones_like(ang), ang)          # [S, d/2]

def rope(x, pos):                                     # x: [d]
    z = torch.view_as_complex(x.view(-1, 2))          # [d/2]
    return torch.view_as_real(z * cis[pos]).flatten()

q, k = torch.randn(d), torch.randn(d)

# same relative distance, different absolute positions → same score
s1 = rope(q, 5)  @ rope(k, 12)     # δ = 7
s2 = rope(q, 60) @ rope(k, 67)     # δ = 7
s3 = rope(q, 5)  @ rope(k, 13)     # δ = 8
print(f"{s1:.6f} {s2:.6f} {s3:.6f}")
assert torch.allclose(s1, s2, atol=1e-4)   # translation invariance ✓
assert not torch.allclose(s1, s3, atol=1e-4)

# norm preservation
assert torch.allclose(rope(q, 999).norm(), q.norm(), atol=1e-4)
```

If you later add context parallelism and this test still passes on rank 0 but
the model will not converge, the bug is that some rank is passing local instead
of global positions (topic 23, §4).

## 6. Cost

| | |
|---|---|
| Parameters | 0 |
| Precompute | `S × d/2` complex numbers, one buffer, shared across layers |
| FLOPs | ~6 per element (one complex multiply per 2 elements), i.e. `O(B·S·h·d)` — negligible vs `O(B·S·d²)` matmuls |
| Memory in backward | none needed: `R_m` is a fixed orthogonal map, so `∂/∂x (R_m x) = R_mᵀ` — the backward is just the inverse rotation, recomputed from the buffer |

Property (d) of the backward pass is worth pausing on: the gradient of RoPE is
"rotate by `−mθ`". No activation is saved. This is why RoPE is free even inside
an activation-checkpointed block.

## 7. Check yourself

1. Prove attention is permutation-equivariant without position encodings.
2. Requirement (R) plus linearity forced orthogonality — where exactly did that
   come from? (Hint: set `n = m`.)
3. Why are rotations the *only* linear isometries of `R²` satisfying (R')?
4. `base = 10000`, `d_h = 128`. What is the longest wavelength? What context
   length would you expect this model to handle before extrapolation problems
   appear, and why?
5. RoPE is a relative encoding but costs `O(S·d)` rather than `O(S²)`. Explain
   the mechanism that buys that.
6. You port a checkpoint from a `rotate_half` implementation into interleaved
   code. Perplexity is bad but not random. Explain.
7. Why is RoPE never applied to `V`? Why is it applied after `W_Q` rather than
   to `x` directly?

## Next

→ [04 — Implementing RoPE and YaRN](04-implementing-rope-and-yarn.md)
