# 08 — Block Matrix Multiplication and MLA Internals

> Video: 03:15:35 — Block Matrix Multiplication and MLA Internals
> Code: `minimal_examples/block_matrix_multiply.py`

## Why this exists

This is the most important piece of algebra in the course, and it looks like the
least. One identity — partition two matrices into blocks and multiply the blocks
— turns out to be the mechanism behind **three** apparently unrelated
techniques:

| Technique | What it is, in terms of this identity |
|-----------|---------------------------------------|
| MLA weight absorption (topic 10) | reassociating a product of blocks |
| Tensor parallelism (topics 21–22) | putting different blocks on different devices |
| Ring attention (topic 23) | computing the blocks of a product in a different order |

If you only take one thing from Part I into Part III, take this. Everything in
distributed training is "the same matmul, blocked differently".

## 1. The identity

Partition `A ∈ R^{m×n}` into a grid of blocks, and `B ∈ R^{n×p}` into a
compatible grid — compatible meaning **`A`'s column partition equals `B`'s row
partition**. Then

```
┌────────────────────────────────┐
│   C_ij  =  Σ_k  A_ik · B_kj    │
└────────────────────────────────┘
```

which is the scalar matmul rule with scalars replaced by blocks. The only
requirement is that the shared index `k` is partitioned the same way on both
sides; the `i` and `j` partitions are free.

`minimal_examples/block_matrix_multiply.py` verifies it concretely with
`Q ∈ R^{4×4}` split `2×2` and `K ∈ R^{4×8}` split `2×4`:

```python
q00, q01 = q[:2, :2], q[:2, 2:]
q10, q11 = q[2:, :2], q[2:, 2:]
k00, k01, k02, k03 = k[:2, :2], k[:2, 2:4], k[:2, 4:6], k[:2, 6:]
k10, k11, k12, k13 = k[2:, :2], k[2:, 2:4], k[2:, 4:6], k[2:, 6:]

c00 = q00 @ k00 + q01 @ k10          # ← Σ over the shared index k
c01 = q00 @ k01 + q01 @ k11
...
torch.testing.assert_close(block_output, q @ k)     # PASS
```

Run it. It takes a second and it makes the rest of the course concrete.

## 2. The two special cases that are tensor parallelism

Everything in Part III's tensor parallelism follows from taking the identity with
just **two** blocks, in the two possible orientations. It is worth stating them
now, because when you meet them in topic 21 they should feel inevitable.

### 2.1 Split the output dimension — "column parallel"

Partition only `B`'s columns. `A` stays whole:

```
A · [ B_1 | B_2 ]  =  [ A B_1 | A B_2 ]
```

Each output block needs `A` and *one* block of `B`. So if device 0 holds `B_1`
and device 1 holds `B_2`, and both have `A`:

- **No communication is needed to compute.** Each device produces a slice of the
  output, independently.
- The result is **sharded along the output dimension**.

```
        Replicate(A)  ×  Shard(1)(B)   →   Shard(1)(C)        no collective
```

### 2.2 Split the contraction dimension — "row parallel"

Partition `A`'s columns *and* `B`'s rows, together (they must match):

```
                     ⎡ B_1 ⎤
   [ A_1 | A_2 ]  ·  ⎢     ⎥  =  A_1 B_1  +  A_2 B_2
                     ⎣ B_2 ⎦
```

Now each device computes a **full-size** output that is only a *partial sum*.
The true result is the sum across devices:

- **Compute is local; a sum across devices is required to finish.**
- The intermediate result is `Partial(sum)` — every device holds a full-shaped
  tensor whose values are incomplete.

```
        Shard(1)(A)  ×  Shard(0)(B)   →   Partial(sum)(C)     needs all-reduce
```

### 2.3 Why this is the whole story

Chain them. Feed a column-parallel layer's output straight into a row-parallel
layer:

```
x · [W_1^{(1)} | W_1^{(2)}] → [h^{(1)} | h^{(2)}]        sharded, no comms
                                   │
                     ⎡ W_2^{(1)} ⎤ │
   [h^{(1)} | h^{(2)}] · ⎢       ⎥  = partial sums        one all-reduce
                     ⎣ W_2^{(2)} ⎦
```

The intermediate `h` is *never gathered*. Two matmuls, one collective, and the
`d_ff`-wide activation — the big one — stays sharded the whole time. That is
Megatron-LM's FFN, and it is the reason tensor parallelism is practical. You now
already know it; topic 21 will only add the nonlinearity argument (why SwiGLU
survives the split) and the backward pass.

## 3. Reassociation: the same product, different cost

Matrix multiplication is **associative**, so `(AB)C = A(BC)`. The results are
identical; the *costs* are not. For `A ∈ R^{m×n}`, `B ∈ R^{n×p}`,
`C ∈ R^{p×q}`, counting MACs:

```
(AB)C :   m·n·p  +  m·p·q
A(BC) :   n·p·q  +  m·n·q
```

Neither is universally better. This is the entire art of MLA absorption: find
the association that is cheapest **for the shape regime you are in**, and note
that decode (`m = 1`) and training (`m = S`) are different regimes.

### 3.1 The classic example: `Q Kᵀ`

```
Q Kᵀ = (x W_Q)(x W_K)ᵀ = x W_Q W_Kᵀ xᵀ = x (W_Q W_Kᵀ) xᵀ
```

Define `W_QK = W_Q W_Kᵀ ∈ R^{d×d}`. Two ways to compute the scores:

```
project-then-score:   x W_Q  → S·d·d_h ;  x W_K → S·d·d_h ;  Q Kᵀ → S²·d_h
                      total ≈ 2 S d d_h + S² d_h

fuse-then-score:      W_QK once (offline) ;  x W_QK → S·d·d ;  (·) xᵀ → S²·d
                      total ≈ S d² + S² d
```

With `d = 2048`, `d_h = 128`, `S = 16384`, the fused form is ~16× worse, because
it destroys the low-rank structure: `W_Q W_Kᵀ` has rank ≤ `d_h = 128` but is
stored and multiplied as a dense `2048×2048`. **The head dimension is a
bottleneck you want to keep.**

So for standard attention, projecting first always wins. Remember this, because
MLA is the case where the answer flips.

## 4. Applying it to MLA

Now do the same analysis on MLA's score. Per head, dropping the RoPE part for
the moment (topic 09 explains why it has to be dropped):

```
q_nope = x W_Q^C                (W_Q^C : d × d_h^C,   2048 × 128)
c      = x W_DKV                (the cached latent,   d_c = 512)
k_nope = c W_UK                 (W_UK  : d_c × d_h^C, 512 × 128)
```

The content part of the score between query `t` and key `s`:

```
⟨ q_nope_t , k_nope_s ⟩ = q_nope_t · k_nope_sᵀ
                        = (x_t W_Q^C) (c_s W_UK)ᵀ
                        = x_t W_Q^C W_UKᵀ c_sᵀ
                        = ( x_t W_Q^C W_UKᵀ ) · c_sᵀ
                          └──────┬───────┘
                                 │
                          define W_Q^abs = W_Q^C W_UKᵀ    [d × d_c] = 2048 × 512
```

```
┌────────────────────────────────────────────────────────────┐
│   ⟨q_nope_t, k_nope_s⟩  =  ⟨ x_t W_Q^abs , c_s ⟩           │
└────────────────────────────────────────────────────────────┘
```

**The query is projected into the latent space, and attends directly against the
cached latent.** `k_nope` is never formed. `W_UK` has vanished into the query
projection, where it can be pre-multiplied once, offline.

The same move works on the output side. Per head, with `A` the attention
weights:

```
out = (A V) W_O^{(i)} = (A c W_UV) W_O^{(i)} = A c ( W_UV W_O^{(i)} )
                                                    └──────┬──────┘
                                              define W_O^abs = W_UV W_O
```

so the attention output is computed **in latent space** (`d_c`-wide) and the
combined `W_O^abs` maps it back to `d`. `V` is never formed either.

Both absorptions are just associativity. Nothing deeper is happening. But notice
what they require:

> **Absorption is possible if and only if nothing non-linear, and nothing
> position-dependent, sits between the two matrices being fused.**

`W_Q^C W_UKᵀ` works because the two projections are adjacent linear maps. Put a
rotation `R_m` between them — which is exactly what RoPE does — and the product
becomes `W_Q R_m W_UKᵀ`, which depends on `m` and therefore cannot be
precomputed. That single observation is the reason MLA has a 128/64 head split,
and it is topic 09.

## 5. Which association wins, and when

Count MACs per head, per layer, for `S_q` query tokens against `S` keys.

**Naive (topic 07's forward): reconstruct, then attend over `d_h`.**

```
reconstruct k_nope :  S · d_c · d_h^C   =  S · 512 · 128   =  65,536 S
reconstruct v      :  S · d_c · d_h^V   =  S · 512 · 128   =  65,536 S
scores             :  S_q · S · d_qk    =  S_q · S · 192
A V                :  S_q · S · d_v     =  S_q · S · 128
```

**Absorbed: project queries into latent space, attend over `d_c`.**

```
q projection       :  S_q · d · d_c     =  S_q · 2048 · 512 =  1,048,576 S_q
scores             :  S_q · S · (d_c + d_h^R) = S_q · S · 576
"A V" in latent    :  S_q · S · d_c     =  S_q · S · 512
output projection  :  S_q · d_c · d     =  1,048,576 S_q
```

### Decode: `S_q = 1`, `S = 16384`

```
naive    :  131,072 · 16384  +  320 · 16384      ≈  2.15e9   MACs
absorbed :  2 · 1.05e6       +  1088 · 16384     ≈  1.96e7   MACs
                                                    ─────────
                                          absorbed is ~110× cheaper
```

The naive form is dominated by re-expanding the whole cached context at every
step — `O(S)` work that produces values used exactly once. Absorption deletes it.

### Training: `S_q = S = 16384`

```
naive    :  131,072·S + 320·S²  = 2.15e9  + 8.59e10  ≈  8.81e10
absorbed :  2.1e6·S   + 1088·S² = 3.44e10 + 2.92e11  ≈  3.26e11
                                                        ─────────
                                          naive is ~3.7× cheaper
```

The flip comes from the `S²` terms: absorbed attention contracts over
`d_c = 512` instead of `d_h = 128`, so its score and output matmuls are ~3.4×
larger — and at `S_q = S` those `S²` terms dominate everything.

```
┌──────────────────────────────────────────────────────────────┐
│  Training  (S_q = S) :  use the naive form.  Reconstruction  │
│                         is O(S); scores are O(S²).           │
│  Decode    (S_q = 1) :  use the absorbed form. Reconstruction│
│                         is O(S); scores are O(S).            │
└──────────────────────────────────────────────────────────────┘
```

This is why `torchfeather` ships **both** `forward` and `forward_absorbed`, and
why `absorb_mla_weights()` is a separate, `@torch.no_grad()`, inference-time
transformation rather than the training path.

And there is a second, larger reason absorption wins in decode which pure FLOP
counting cannot see — it changes *arithmetic intensity*, not just FLOPs. Topic 10
finishes that argument.

## 6. Do the algebra yourself

The associativity claim should not be taken on faith. This is the check:

```python
import torch
torch.manual_seed(0)

d, d_c, d_h, S = 64, 32, 16, 12

W_Q  = torch.randn(d, d_h)      # per-head query projection  (content only)
W_UK = torch.randn(d_c, d_h)    # per-head key up-projection
x    = torch.randn(S, d)
c    = torch.randn(S, d_c)      # the cached latents

# --- naive: reconstruct k, then score ---------------------------------------
q      = x @ W_Q                # [S, d_h]
k_nope = c @ W_UK               # [S, d_h]
scores_naive = q @ k_nope.T     # [S, S]

# --- absorbed: fold W_UK into W_Q, score against the latent ----------------
W_Q_abs = W_Q @ W_UK.T          # [d, d_c]   ← precomputed once, offline
q_abs   = x @ W_Q_abs           # [S, d_c]
scores_abs = q_abs @ c.T        # [S, S]

print((scores_naive - scores_abs).abs().max().item())     # 1.5e-4 on scores of ~500
torch.testing.assert_close(scores_naive, scores_abs, rtol=1e-4, atol=1e-4)
print("absorption is exact ✓")

# --- and now break it with RoPE, to preview topic 09 -----------------------
def rot(v, m, theta=0.5):                      # a single 2-D rotation plane
    z = torch.view_as_complex(v.reshape(*v.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(z * torch.polar(torch.ones(()), torch.tensor(m*theta))
                              ).flatten(-2)

m = 3
q_rot = rot(x @ W_Q, m)
scores_rope = q_rot @ k_nope.T                 # rotation applied between the two
scores_abs_rope = rot(x @ W_Q_abs, m) @ c.T    # "same" absorption, now wrong
print("absorption after RoPE differs by:",
      (scores_rope - scores_abs_rope).abs().max().item())   # 790.8 — not exact
```

The final block is the point of the exercise. Absorption is exact, until you put
a position-dependent rotation between the two matrices — then the "absorbed"
computation is a *different function*, not a cheaper one. Fixing that is the
subject of the next topic.

## 7. Check yourself

1. State the block matmul identity and the one compatibility condition it needs.
2. Derive both tensor-parallel primitives (column-split, row-split) from it, and
   say which one produces a `Partial(sum)` result.
3. In a column-parallel → row-parallel pair, which activation is never gathered,
   and how many collectives does the pair need in the forward pass?
4. For standard attention, why is `x(W_Q W_Kᵀ)xᵀ` a bad idea? Give the rank
   argument and the FLOP count.
5. Derive `W_Q^abs = W_Q^C W_UKᵀ` and state its shape.
6. Derive the output-side absorption `W_O^abs = W_UV W_O`.
7. Give the exact condition under which two adjacent linear maps can be absorbed.
   Name two things that violate it in a transformer.
8. Show the FLOP crossover: why absorbed wins at `S_q = 1` and loses at
   `S_q = S`. Which term flips the comparison?

## Next

→ [09 — Deriving MLA and decoupled RoPE](09-deriving-mla-and-decoupled-rope.md)
