# 09 — Deriving MLA and Decoupled RoPE

> Video: 03:34:58 — Deriving MLA and Decoupled RoPE
> Papers: DeepSeek-V2 (2405.04434), MQA (1911.02150), GQA (2305.13245)

## Why this exists

We have all the pieces: the bandwidth problem (topic 06), the code (topic 07),
and the algebra (topic 08). Now derive the architecture from the requirement,
and — more importantly — discover the obstacle that gives MLA its strange
128 + 64 head split.

The interesting claim of this topic is that **decoupled RoPE is not a design
choice. It is the unique repair for a conflict between two requirements**, and
once you see the conflict there is essentially nothing else you could have done.

## 1. The requirement

```
Minimize:    cached values per token per layer
Subject to:  (a) every head keeps its own key subspace and its own value channel
             (b) the score is computable without reconstructing per-head K, V
             (c) RoPE still works
```

Constraint (a) is what rules out MQA and GQA — they satisfy the cache goal by
*tying heads together*, which is exactly the expressiveness we want to keep.
Constraint (b) is arithmetic intensity (topic 06 §3.5): it is not enough to
store few bytes, those bytes must feed many FLOPs. Constraint (c) is
non-negotiable — a model without position encoding is not a language model.

We will satisfy (a) and (b), discover that (c) is *incompatible* with (b), and
then find the minimal repair.

## 2. Step 1 — low-rank compression

In MHA, each head `i` has `W_K^{(i)} ∈ R^{d×d_h}` and `W_V^{(i)} ∈ R^{d×d_h}`.
Stack them all:

```
W_KV  =  [ W_K^{(1)} | … | W_K^{(h)} | W_V^{(1)} | … | W_V^{(h)} ]
              ∈ R^{d × 2h·d_h}
```

and the cached quantity is `x_t W_KV ∈ R^{2h·d_h}` — 4096 values for our config.

**The idea:** constrain `W_KV` to be **low rank**. Factor it through a
`d_c`-dimensional bottleneck:

```
┌───────────────────────────────────────────────────────────┐
│   W_KV  =  W_DKV · W_UKV ,    W_DKV ∈ R^{d×d_c}           │
│                                W_UKV ∈ R^{d_c×2h·d_h}     │
└───────────────────────────────────────────────────────────┘
```

Then define the latent and note what caching it buys:

```
c_t = x_t W_DKV                          ∈ R^{d_c}          ← cache this (512)

K_t^{(i)} = c_t W_UK^{(i)}               ∈ R^{d_h}          reconstruct on demand
V_t^{(i)} = c_t W_UV^{(i)}               ∈ R^{d_h}
```

Constraint (a) is satisfied **exactly**: each head still has its own `W_UK^{(i)}`
and `W_UV^{(i)}`, so each head still gets a distinct `d_h`-dimensional key and a
distinct value. Nothing is tied.

What we gave up: `W_KV` is now rank ≤ `d_c = 512` instead of full rank 4096. That
is a genuine restriction on the function class. But it is a *soft* restriction
distributed across all heads, rather than GQA's *hard* restriction of forcing
groups of heads to be identical. Compare the two at equal cache:

```
GQA at 512 cached values  :  h_kv = 2 groups.  14 of 16 heads are copies.
MLA at 512 cached values  :  d_c = 512.        16 distinct heads, rank-512 K/V.
```

Empirically MLA at DeepSeek-V2's setting outperforms MHA (not merely matches it)
while using 1/57 of the cache. That is a little surprising, and the honest
explanation is that low-rank structure acts as a regularizer — the same reason
LoRA works. Follow-up work (e.g. TransMLA) further argues GQA is a *special
case* of MLA — a block-structured choice of `W_UK` — which if right means MLA is
never worse at equal cache. Treat that as a reasonable claim rather than settled
fact.

## 3. Step 2 — absorption, to satisfy constraint (b)

Compression alone fails constraint (b). If you cache `c_t` and then reconstruct
`K` and `V` at every decode step, you pay `O(S · d_c · d_h)` work per step to
produce values used once — worse arithmetic intensity than MHA, not better
(topic 08 §5).

Topic 08 gave the fix. Since `q_t = x_t W_Q^{(i)}` and `K_s = c_s W_UK^{(i)}` are
adjacent **linear** maps:

```
⟨ q_t , K_s ⟩ = x_t W_Q^{(i)} (c_s W_UK^{(i)})ᵀ
              = x_t ( W_Q^{(i)} W_UK^{(i)ᵀ} ) c_sᵀ
              = ⟨ x_t W_Q^abs,(i) , c_s ⟩
```

and symmetrically `W_O^abs,(i) = W_UV^{(i)} W_O^{(i)}`. The reconstruction
disappears; attention is computed directly against the `d_c`-wide cache.
Constraint (b) satisfied.

At this point we have an architecture that caches 512 values per token per layer,
keeps 16 distinct heads, and attends directly against the cache. Now add RoPE.

## 4. The collision

RoPE post-multiplies `q` and `k` by a position-dependent rotation (row-vector
convention, topic 03):

```
q'_t = q_t R_t ,      k'_s = k_s R_s ,      R_m orthogonal, block-diagonal
```

The score becomes

```
⟨ q'_t , k'_s ⟩ = q_t R_t (k_s R_s)ᵀ = q_t R_t R_sᵀ k_sᵀ = q_t R_{t−s} k_sᵀ
```

That relative-only structure is the entire point of RoPE. Now substitute MLA's
factorization:

```
⟨ q'_t , k'_s ⟩ = x_t W_Q · R_{t−s} · W_UKᵀ · c_sᵀ
                          └────────┘
                    a position-dependent matrix, sitting
                    exactly where the absorption must happen
```

The would-be absorbed matrix is

```
W_Q^abs(t, s) = W_Q · R_{t−s} · W_UKᵀ
```

and it **depends on `t − s`**. Three consequences, each fatal:

1. **It cannot be precomputed.** You would need one `[d × d_c]` matrix per
   distinct relative distance — up to `S` of them. At `d=2048, d_c=512,
   S=163840`, that is 172 billion values *per head per layer*. Absurd.
2. **The alternative — keep them separate — undoes the compression.** To apply
   `R_s` to the key you must *have* the key, i.e. compute `k_s = c_s W_UK`
   explicitly. That is precisely the reconstruction absorption was meant to
   avoid.
3. **You cannot move the rotation out of the way.** `R_{t−s}` does not commute
   with `W_UKᵀ` in general (rotations commute only with matrices that preserve
   the same 2-D planes), so no reassociation rescues it.

Stated as a theorem-shaped observation:

> **Compression + absorption + RoPE cannot all hold on the same dimensions.**
> Absorption needs a position-*independent* product of adjacent linear maps;
> RoPE inserts a position-*dependent* factor between them.

Note this is not a problem for MHA — there is nothing to absorb, so RoPE sits
harmlessly at the end of the `W_K` projection. The conflict is created by
compression.

## 5. The repair: decouple

If the three requirements cannot hold on the same dimensions, put them on
**different dimensions**. Split each head's query/key vector into two
concatenated parts and let the dot product split with it:

```
q_t = [ q_t^C ‖ q_t^R ]        k_s = [ k_s^C ‖ k_s^R ]

⟨ q_t , k_s ⟩ = ⟨ q_t^C , k_s^C ⟩  +  ⟨ q_t^R , k_s^R ⟩
                └──── content ────┘   └──── position ────┘
                 absorbable            RoPE lives here
                 (no RoPE)             (not absorbed)
```

The concatenated dot product decomposes into a sum — the one algebraic fact this
repair depends on. Now assign responsibilities:

### The content path (`d_h^C = 128`) — compressed, absorbed, no RoPE

```
q_t^C = x_t W_Q^C                            per head
k_s^C = c_s W_UK                             per head, from the cached latent
      ⟹ absorbable:  ⟨q^C, k^C⟩ = ⟨ x_t W_Q^C W_UKᵀ , c_s ⟩     ✓
```

These dimensions carry **no position information at all** ("NoPE").

### The position path (`d_h^R = 64`) — not compressed, RoPE'd, shared

```
q_t^R = ( x_t W_QR ) R_t                     per head
k_s^R = ( x_s W_KR ) R_s                     ONE per token, shared by all heads
      ⟹ cached directly (64 values); nothing to absorb, so RoPE is harmless   ✓
```

There is no up-projection on this path, so there is no product for the rotation
to sit inside. `k^R` is computed straight from `x_s`, rotated, and cached as-is.

### Why `k^R` is shared across heads

Two reasons, and the first is decisive:

1. **Cache cost.** Per-head `k^R` would cost `h · d_h^R = 16 · 64 = 1024`
   values per token — nearly tripling the 576-value cache and undoing the win.
   Shared, it costs 64.
2. **It is a natural place to share.** The rotation `R_s` is identical for every
   head (position is a property of the token, not the head), so heads differ only
   in the *content* of the positional key. Empirically that variety is not needed
   — the per-head query `q^R` already lets each head choose *which* positional
   pattern it looks for.

Note the asymmetry: **`k^R` is shared, `q^R` is per-head.** Correct, because
queries are not cached — a per-head query costs transient activation, not `O(S)`
memory. This is the same asymmetry as MQA, applied to only 64 dimensions.

This is exactly the `k_pe.expand(-1, -1, self.n_heads, -1)` from topic 07 §
step 4, and exactly why `precompute_freqs_cis` uses
`dim = args.qk_rope_head_dim = 64` rather than the full 192 (topic 04 §6).

## 6. The final accounting

```
cached per token per layer:

    c_t   (latent, content K and V for all heads)          d_c     = 512
    k^R_t (positional key, shared by all heads)            d_h^R   =  64
                                                           ─────────────
                                                                     576
```

Against MHA's `2·h·d_h = 4096` — a 7.1× reduction at `h = 16`, and 57× at
DeepSeek-V2's `h = 128`, because **the MLA cache does not depend on `h` at all.**
That head-independence is the property that makes the technique scale.

And the head dimension budget:

```
d_qk = 192  =  128 (content: compressed, absorbed, position-free)
                +  64 (position: uncompressed, RoPE'd, shared key)
d_v  = 128     (content only — values never enter a dot product,
                so they need no positional component)
```

Every number in `DeepSeekV3ModelArgs` is now derived rather than given.

### Does removing position from 128 of 192 dimensions hurt?

It is a fair worry, and the answer is the sum structure of §5: the score is
`content_term + position_term`, and the model can weight them freely by scaling
`W_Q^C` against `W_QR` during training. All the relative-position signal the
model needs flows through 64 dimensions — which, per topic 03 §3.1, is 32
independent frequency planes with wavelengths from 6.3 to 47,117 tokens. That is
a complete multi-scale positional clock. The content dimensions do not need a
second copy of it.

The trade is real but small, and it buys the 7–57× cache reduction. Note the
comparison is not "MLA vs. MHA at the same cache" — it is "MLA at 576 values vs.
GQA at 576 values", and against *that* baseline MLA keeps 16 distinct heads
where GQA-2 has 2.

## 7. The design as a decision tree

```
Cache is O(S·h·d_h) and dominates serving cost.
│
├─ Tie heads together?
│    MQA (h_kv=1)  → AI = h,  cache/h.     Quality cost: one shared subspace.
│    GQA (h_kv=g)  → AI = h/g, cache·g/h.  Quality cost: g distinct subspaces.
│
└─ Compress instead?   ← MLA
     c_t = x_t W_DKV, cache d_c.  Heads stay distinct.
     │
     ├─ Reconstruct K,V per step?     ✗ O(S) work/step, AI worse than MHA
     └─ Absorb W_UK into W_Q?         ✓ attend directly against the latent
          │
          └─ But RoPE sits between W_Q and W_UK.       ✗ CONFLICT
               │
               └─ Split the head dimension:            ✓ DECOUPLED RoPE
                    128 dims: content, absorbed, no position
                     64 dims: position, RoPE'd, shared key, not absorbed
```

## 8. Verify the collision and the repair

```python
import torch
torch.manual_seed(0)
d, d_c, dC, dR = 64, 32, 16, 8

W_Q_C = torch.randn(d, dC)     # content query   (absorbable path)
W_UK  = torch.randn(d_c, dC)   # content key up-projection
W_QR  = torch.randn(d, dR)     # positional query
W_KR  = torch.randn(d, dR)     # positional key (shared across heads)

def rot(v, m, theta=0.3):      # dR/2 planes, single frequency for simplicity
    z = torch.view_as_complex(v.reshape(*v.shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(z * torch.polar(torch.ones(()),
                                              torch.tensor(m * theta))).flatten(-2)

x = torch.randn(6, d)
c = x @ torch.randn(d, d_c)    # the latent

t, s = 4, 1

# --- reference score: content (no RoPE) + position (RoPE) -------------------
qC, kC = x[t] @ W_Q_C, c[s] @ W_UK
qR, kR = rot(x[t] @ W_QR, t), rot(x[s] @ W_KR, s)
ref = qC @ kC + qR @ kR

# --- absorbed: content path folded, position path untouched -----------------
W_Q_abs = W_Q_C @ W_UK.T                     # [d, d_c], precomputed offline
content = (x[t] @ W_Q_abs) @ c[s]            # never forms kC
position = rot(x[t] @ W_QR, t) @ rot(x[s] @ W_KR, s)
absorbed = content + position

print("decoupled absorption error:", (ref - absorbed).abs().item())   # 0.0

# --- what happens if you (wrongly) RoPE the content path too ---------------
bad_ref = rot(x[t] @ W_Q_C, t, 0.3) @ rot(c[s] @ W_UK, s, 0.3)
bad_abs = rot(x[t] @ W_Q_abs, t, 0.3) @ c[s]
print("naive absorption after RoPE:", (bad_ref - bad_abs).abs().item())  # 451.9
```

The two printed numbers are the whole topic. On scores of magnitude ~200:

```
decoupled absorption error: 0.0
naive absorption after RoPE: 451.9
```

Exact when RoPE is confined to the decoupled path; catastrophically wrong the
moment it touches the absorbed path. Not "slightly less accurate" — a different
function.

## 9. Check yourself

1. State the three constraints and say which existing method each of MQA and GQA
   violates.
2. `W_KV = W_DKV W_UKV` is a rank constraint. What is the rank, and what is the
   full-rank alternative's cache cost?
3. In what precise sense does MLA keep "distinct heads" while GQA does not, at
   equal cache size?
4. Write out `W_Q^abs(t,s)` with RoPE included and give the three reasons it
   cannot be used.
5. Rotations do not commute with `W_UKᵀ`. Why does that close off the obvious
   escape route?
6. Which algebraic identity makes the decoupled split work?
7. Why is `k^R` shared across heads but `q^R` per-head? Give the cache
   arithmetic for both choices.
8. `d_v = 128` has no positional component while `d_qk = 192` does. Justify from
   where `V` appears in the attention formula.
9. Compute the MLA cache for `h = 128, d_h = 128, d_c = 512, d_h^R = 64` and
   compare to MHA. Which term in the MHA formula is absent from MLA's?

## Next

→ [10 — MLA weight absorption](10-mla-weight-absorption.md)
