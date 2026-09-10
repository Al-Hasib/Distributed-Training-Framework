# 04 — Implementing RoPE and YaRN

> Video: 01:11:25 — Implementing RoPE and YaRN
> Code: `torchfeather/model/rope.py`, `torchfeather/model/model.py:49-53`

## Why this exists

Topic 03 ended on a defect. RoPE encodes position as an angle `m·θ_i`, and for
the slow planes the training run only ever exhibits a small arc of the circle:
at `S_train = 4096` with `d = 64`, the slowest plane sweeps
`4096 / 47117 ≈ 8.7%` of a rotation. Ask the model about position 40,000 and
those planes present angles from a region of the circle it has never seen. The
attention logits go out of distribution and perplexity explodes — not degrades,
explodes.

We want a model trained at 4k to work at 160k without retraining from scratch.
This topic derives the three successive fixes — **Position Interpolation**,
**NTK-aware scaling**, **NTK-by-parts (YaRN)** — plus YaRN's second, separate
correction (attention temperature), and then reads the reference implementation
line by line.

## 1. Framing: what can we even change?

The angle in plane `i` at position `m` is

```
φ_i(m) = m · θ_i ,      θ_i = base^(−2i/d)
```

There are exactly two knobs:

```
     rescale the position          rescale the frequency
        m → m / s                     θ_i → θ'_i
```

and they are the same knob, because `φ` only depends on the product. Every
method below is a different choice of **how much to shrink each `θ_i`**. Define
the **scale factor**:

```
s = S_new / S_train          (e.g. 163840 / 4096 = 40)
```

Our goal: at the new maximum position `S_new`, no plane should present an angle
outside the range it saw during training. The training range for plane `i` was
`[0, S_train · θ_i]`. So the safe condition is

```
S_new · θ'_i  ≤  S_train · θ_i      ⟺      θ'_i ≤ θ_i / s
```

Simple. But applying it blindly is exactly what goes wrong first.

## 2. Attempt 1 — Position Interpolation (PI)

Apply the bound with equality, to every plane:

```
θ'_i = θ_i / s        for all i          ⟺        m → m / s
```

Positions `0…S_new` are linearly squeezed into the trained window
`0…S_train`. In code it is a one-line change (`freqs = freqs / factor`).

**It works** — with a short finetune — and it was the first practical
long-context recipe (Chen et al., 2023).

**Why it is wasteful.** Consider plane 0: `θ_0 = 1`, wavelength 6.3 tokens. Over
4096 training positions it completed 650 full rotations, so *every* phase in
`[0, 2π)` was seen thousands of times. It was never out of distribution and
needed no help. Yet PI divides it by 40, so adjacent tokens — previously
separated by 1 radian — are now separated by 0.025 radians. The model's ability
to distinguish "the previous word" from "two words back" is crushed by a factor
of 40. That resolution is exactly what the fast planes exist for.

> **The insight PI misses:** the planes are in completely different regimes.
> Fast planes have seen everything and must be left alone. Slow planes have seen
> almost nothing and need the full correction.

## 3. Attempt 2 — NTK-aware scaling

Instead of scaling all `θ_i` equally, change the **base**, which scales them
*unequally* by construction:

```
θ'_i = (base')^(−2i/d)
```

At `i = 0` this is `1` regardless of `base'` — the fastest plane is
automatically untouched. Choose `base'` so the *slowest* plane
(`i = d/2 − 1`) receives the full factor `s`:

```
(base')^(−(d−2)/d) = base^(−(d−2)/d) / s
⟹ (base'/base)^((d−2)/d) = s
⟹ ┌────────────────────────────────┐
   │   base' = base · s^(d/(d−2))   │
   └────────────────────────────────┘
```

For `d = 128, s = 40`: `base' = 10000 · 40^(1.0159) ≈ 10000 · 42.8 ≈ 4.28e5`.
This is the "just raise rope_theta" trick, and it works better than PI with no
finetuning at all. The name comes from Neural Tangent Kernel theory: networks
learn high-frequency features poorly, so do not disturb the high frequencies you
already have.

**Why it is still not right.** The interpolation amount now varies *smoothly and
monotonically* with `i` — but the actual regime boundary is not smooth. What
matters is whether a plane completed many rotations or fewer than one, and
that threshold sits at a specific `i`. NTK-aware scaling under-corrects the
mid-slow planes (which genuinely need help) while still slightly perturbing the
fast ones (which need none). YaRN calls this "some dimensions are out of
bounds".

## 4. Attempt 3 — NTK-by-parts, i.e. YaRN

### 4.1 The right variable: rotations per training window

Stop indexing by `i`. Index by the quantity that actually determines the regime:

```
┌──────────────────────────────────────────────────────┐
│   r_i  =  S_train / λ_i  =  S_train · θ_i / (2π)     │
│         = number of full rotations plane i completed │
└──────────────────────────────────────────────────────┘
```

Now classify:

| regime | condition | what the model saw | correct action |
|--------|-----------|--------------------|----------------|
| **fast** | `r_i > β` (e.g. `β = 32`) | every phase, many times | **do not touch.** It is in-distribution at any position. Interpolating destroys local resolution. |
| **slow** | `r_i < α` (e.g. `α = 1`) | less than one rotation — a monotone arc used as an absolute position feature | **interpolate fully** (`θ_i/s`). Extrapolating runs off the end of the arc. |
| **middle** | `α ≤ r_i ≤ β` | partial coverage | **ramp linearly** between the two. |

The ramp: with `γ_i ∈ [0,1]` the "keep original" weight,

```
        ⎧ 1                      if r_i > β        (keep)
γ_i  =  ⎨ (r_i − α)/(β − α)      if α ≤ r_i ≤ β
        ⎩ 0                      if r_i < α        (interpolate)

┌────────────────────────────────────────────┐
│   θ'_i  =  (1 − γ_i) · θ_i/s  +  γ_i · θ_i │
└────────────────────────────────────────────┘
```

That is the whole of YaRN's frequency modification. `α` and `β` are the only new
hyperparameters, and `α = 1, β = 32` works across models.

### 4.2 Converting rotation thresholds to dimension indices

The implementation needs to build the ramp as a function of the plane index, so
invert `r_i = r` for `i`:

```
r = S_train · base^(−2i/d) / (2π)

base^(2i/d) = S_train / (2π r)

(2i/d) ln base = ln( S_train / (2π r) )

┌──────────────────────────────────────────────┐
│            d · ln( S_train / (2π r) )        │
│   i(r)  =  ───────────────────────────       │
│                   2 · ln base                │
└──────────────────────────────────────────────┘
```

Which is, verbatim, the reference code:

```python
def find_correction_dim(num_rotations, dim, base, max_seq_len):
    return (
        dim
        * math.log(max_seq_len / (num_rotations * 2 * math.pi))
        / (2 * math.log(base))
    )
```

Note `i(r)` is **decreasing** in `r` (more rotations ⇒ lower index ⇒ faster
plane). So the *high-rotation* threshold `β = beta_fast` maps to the *low*
index, and `α = beta_slow` maps to the high index — which is why the code reads:

```python
def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
    low  = math.floor(find_correction_dim(low_rot,  dim, base, max_seq_len))
    high = math.ceil (find_correction_dim(high_rot, dim, base, max_seq_len))
    return max(low, 0), min(high, dim - 1)

low, high = find_correction_range(beta_fast, beta_slow, dim, base,
                                  args.original_seq_len)
```

`beta_fast = 32` is passed as `low_rot` because it produces the **low index**.
The naming trips everyone up once; it is consistent.

### 4.3 The reference config, computed exactly

`dim = qk_rope_head_dim = 64`, `base = 10000`, `original_seq_len = 4096`,
`beta_fast = 32`, `beta_slow = 1`:

```
i(32) = 64·ln(4096/(32·2π)) / (2·ln 10000) = 64·ln(20.37)/18.42 = 10.47  → low  = 10
i(1)  = 64·ln(4096/(1 ·2π)) / (2·ln 10000) = 64·ln(651.9)/18.42 = 22.51  → high = 23
```

With `d/2 = 32` planes total:

```
plane index:  0 ────────── 10 ─────────── 23 ──────── 31
regime:       │  untouched  │  linear ramp │ θ/40      │
r_i:          650 …………… 32   ……………………  1   ……… 0.087
λ_i (tokens): 6.3 ……………… 128  ……………………  4096 …… 47117
```

11 fast planes keep full local resolution; 9 slow planes are fully interpolated;
13 blend. Compare to PI, which would have divided all 32 by 40.

### 4.4 The implementation, read against the derivation

```python
def linear_ramp_factor(min, max, dim):
    if min == max:
        max += 0.001                              # avoid divide-by-zero
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    return torch.clamp(linear_func, 0, 1)         # 0 at i≤low, 1 at i≥high

freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

if seqlen > args.original_seq_len:
    low, high = find_correction_range(beta_fast, beta_slow, dim, base,
                                      args.original_seq_len)
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
```

Map it to the formula:

| code | derivation |
|------|------------|
| `linear_ramp_factor(low, high, dim//2)` | `0` for fast planes, `1` for slow planes |
| `smooth = 1 - ramp` | `γ_i` — the **keep-original** weight (1 for fast, 0 for slow) |
| `freqs/factor * (1 - smooth)` | `(1 − γ_i) · θ_i / s` |
| `+ freqs * smooth` | `+ γ_i · θ_i` |

Exactly `θ'_i = (1 − γ_i)·θ_i/s + γ_i·θ_i`. ✓

Two things to notice about how the config is used:

- **`factor` (`rope_factor = 40`) is a config constant, not `S_new/S_train`.**
  It is the scale the model *was extended by* when YaRN was applied, and it
  belongs to the checkpoint. Do not recompute it from your current
  `max_seq_len`, or a model extended to 160k and then run at 16k will get the
  wrong frequencies. The `if seqlen > original_seq_len` guard only decides
  *whether* YaRN is active.
- **`original_seq_len`, not `seqlen`, is passed to `find_correction_range`.**
  The regime classification is about what the model *saw in training*, which is
  `original_seq_len` by definition. Passing `seqlen` here is a subtle and common
  bug: it moves the boundary as you change inference length.

## 5. YaRN's second correction: attention temperature

Interpolating frequencies is not sufficient. YaRN reports — and this is an
empirical finding, not a derivation — that extending context also requires
**sharpening the attention distribution**. The intuition: at `s×` the context
length the softmax runs over `s×` more keys, so probability mass spreads out and
average attention entropy rises; the model was tuned for the sharper
distribution it saw in training.

The fix is a temperature `t` on the logits:

```
attn = softmax( qᵀk / (t √d) )
```

which, because RoPE is applied to `q` and `k` individually, can be folded into a
uniform rescale of `q` and `k` by `√(1/t)` and therefore costs nothing. YaRN's
empirical fit (for Llama-family models):

```
√(1/t) = 0.1 · ln(s) + 1
```

DeepSeek-V3 uses the same form with a tunable coefficient `mscale`, and folds it
straight into the softmax scale — which is exactly what `torchfeather` does:

```python
# model.py:49-53
self.softmax_scale = self.qk_head_dim ** -0.5
if model_args.max_seq_len > model_args.original_seq_len:
    mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0
    self.softmax_scale = self.softmax_scale * mscale * mscale
```

The square appears because `√(1/t)` multiplies *both* `q` and `k`, giving
`1/t` on the logit. With `mscale = 0.70` and `rope_factor = 40`:

```
mscale_factor = 0.1 · 0.70 · ln(40) + 1 = 0.1·0.70·3.689 + 1 = 1.2582
softmax_scale = 192^(−1/2) · 1.2582²  = 0.07217 · 1.5831 = 0.11425
```

versus `0.07217` without YaRN — logits are sharpened by 58%.

Two consequences worth internalizing:

- It is a **constant**, computed once at construction. It does not vary per
  position or per plane. Once you enable long context, short sequences also get
  the sharpened scale — that is intentional and matches the released configs.
- It lives in **attention**, not in `rope.py`. If you implement YaRN and only
  touch the frequency table, you have implemented NTK-by-parts, not YaRN, and
  you will see a quality gap you cannot explain from the RoPE code.

## 6. Full annotated implementation

```python
def precompute_freqs_cis(args) -> torch.Tensor:
    dim        = args.qk_rope_head_dim   # 64 — only the decoupled RoPE dims!
    seqlen     = args.max_seq_len        # 16384
    base       = args.rope_theta         # 10000.0
    factor     = args.rope_factor        # 40  (the extension the ckpt was made with)
    beta_fast  = args.beta_fast          # 32  (rotations: "in-distribution" threshold)
    beta_slow  = args.beta_slow          # 1   (rotations: "never wrapped" threshold)

    # --- base RoPE frequencies: θ_i = base^(−2i/d) --------------------------
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    # --- YaRN NTK-by-parts interpolation ------------------------------------
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base,
                                          args.original_seq_len)   # (10, 23)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)        # γ_i
        freqs = freqs / factor * (1 - smooth) + freqs * smooth      # θ'_i

    # --- angles and unit complex exponentials -------------------------------
    t = torch.arange(seqlen)                       # [S]
    freqs = torch.outer(t, freqs)                  # [S, d/2]  = m·θ'_i
    return torch.polar(torch.ones_like(freqs), freqs)   # [S, d/2] complex
```

Note `dim = qk_rope_head_dim = 64`, **not** the full head dimension of 192. In
MLA only 64 of the query/key dimensions carry RoPE; the other 128 are
position-free ("NoPE"). Why that split exists is topic 09 — it is forced by the
weight-absorption trick, not chosen for modelling reasons.

### 6.1 Registration and materialization

```python
# model.py:394
self.register_buffer("freqs_cis", precompute_freqs_cis(model_args),
                     persistent=False)

# model.py:414-416  (inside init_weights)
buffer_device = buffer_device or self.freqs_cis.device
with torch.device(buffer_device):
    self.freqs_cis = precompute_freqs_cis(self.model_args)
```

Three deliberate decisions here, all of which matter later:

1. **`register_buffer`, not `nn.Parameter`.** No gradient, no optimizer state,
   and — critically — it is *not* a parameter that FSDP or TP will try to shard
   (topics 20, 22). Sharding the frequency table would be a category error.
2. **`persistent=False`.** It is excluded from `state_dict`, so checkpoints do
   not carry a `16384 × 32` complex tensor, and — more importantly — changing
   `max_seq_len` does not produce a checkpoint shape mismatch. The price is that
   it must be *recomputed* rather than loaded.
3. **Recomputed in `init_weights` on `buffer_device`.** This is what pays that
   price. Under FSDP/PP the model is often constructed on the `meta` device and
   materialized later; a `meta`-device buffer contains no data. `init_weights`
   is where every stage rebuilds it on real hardware.

> **Pipeline-parallel trap.** Under PP each stage holds a subset of layers but
> *every* stage with an attention layer needs `freqs_cis`. Because the buffer is
> non-persistent it will not arrive via `load_state_dict` — it must be
> regenerated per stage. If you build your pipeline split by hand and forget to
> call `init_weights` on a stage, you get a `meta` tensor error, or worse,
> zeros. See topic 17.

### 6.2 Passing it through the model

```python
# TransformerBlock.forward
x = x + self.attention(self.attention_norm(x), freqs_cis)

# Transformer.forward
for layer in self.layers.values():
    h = layer(h, self.freqs_cis)
```

It is threaded explicitly as an argument rather than read from a global. That
looks verbose but is what allows **position offsetting**, which is the hook
every advanced feature needs:

```python
h = layer(h, self.freqs_cis[start_pos : start_pos + seq_len])
```

- **KV-cache decode:** generating token `m` needs row `m`, not row `0`.
- **Context parallelism (topic 23):** rank `r` holds sequence positions
  `[r·S/P_cp, (r+1)·S/P_cp)` and must slice `freqs_cis` at its *global* offset.
  With the load-balanced zigzag sharding used for ring attention, the slice is
  not even contiguous — you gather the rows your rank actually owns.

This is the concrete form of the warning in topic 03: **RoPE is the one component
whose correctness depends on knowing the global position of a locally-held
token.** Every other transformer op is position-agnostic.

## 7. Test it

```python
import math, torch

class A:  # minimal stand-in for the model args
    qk_rope_head_dim, max_seq_len, original_seq_len = 64, 163840, 4096
    rope_theta, rope_factor, beta_fast, beta_slow = 10000.0, 40, 32, 1

cis = precompute_freqs_cis(A())
assert cis.shape == (163840, 32)
assert torch.allclose(cis.abs(), torch.ones(1))     # unit modulus everywhere

# 1. relative-position invariance survives YaRN (it must: it is still a rotation)
d = 64
def rope(x, pos):
    z = torch.view_as_complex(x.view(-1, 2).float())
    return torch.view_as_real(z * cis[pos]).flatten()

q, k = torch.randn(d), torch.randn(d)
assert torch.allclose(rope(q, 100) @ rope(k, 107),
                      rope(q, 9000) @ rope(k, 9007), atol=1e-3)

# 2. the fast planes are untouched, the slow planes are divided by 40
plain = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
yarn  = torch.angle(cis[1])                # angles at position m = 1  ==  θ'_i
assert torch.allclose(yarn[:10], plain[:10], atol=1e-6)          # i < low  = 10
assert torch.allclose(yarn[24:], plain[24:] / 40, atol=1e-9)     # i > high = 23
print("fast planes preserved, slow planes interpolated ✓")

# 3. the fully-interpolated planes now reach, at S_new, exactly the angle they
#    reached at S_train before extension — the safety condition from section 1
assert torch.allclose(163840 * yarn[24:], 4096 * plain[24:], rtol=1e-5)
print("slow planes exactly within the trained angular envelope ✓")
```

All three assertions pass against the reference implementation. Test 2 is the one
that catches real bugs: pass `seqlen` instead of `original_seq_len` to
`find_correction_range` and the boundary moves, so `yarn[:10]` stops matching
`plain[:10]`. The boundary itself is worth printing once —

```
i:      8        9        10       11       12
plain:  0.100000 0.074989 0.056234 0.042170 0.031623
yarn:   0.100000 0.074989 0.056234 0.039007 0.026879
                          └─ low=10: ramp is still 0, θ untouched
                                   └─ blending begins
```

— because seeing `yarn[10] == plain[10]` and `yarn[11] < plain[11]` confirms the
ramp is anchored where the derivation says it should be.

## 8. Summary table

| Method | `θ'_i` | Fast planes | Slow planes | Finetune needed |
|--------|--------|-------------|-------------|-----------------|
| none | `θ_i` | fine | **OOD** | — |
| PI | `θ_i / s` | **crushed** | fixed | yes |
| NTK-aware | `base'^(−2i/d)` | fine | under-corrected | no (works zero-shot) |
| NTK-by-parts | ramp on `r_i` | fine | fixed | short finetune |
| **YaRN** | ramp on `r_i` **+ temperature** | fine | fixed | shortest finetune, best ppl |

## 9. Check yourself

1. Why does PI need a finetune while NTK-aware scaling works zero-shot?
2. Derive `base' = base · s^(d/(d−2))` from the requirement that the slowest
   plane be interpolated by exactly `s`.
3. In `find_correction_range(beta_fast, beta_slow, ...)`, why is `beta_fast = 32`
   passed as the argument named `low_rot`?
4. What breaks if you pass `max_seq_len` instead of `original_seq_len` to
   `find_correction_range`? Would you notice from a loss curve?
5. `mscale` lives in the attention module, not `rope.py`. Why can it be folded
   into `softmax_scale` for free?
6. `freqs_cis` is registered with `persistent=False`. State one benefit and one
   obligation this creates. Which obligation bites under pipeline parallelism?
7. Under context parallelism with `P_cp = 4` and `S = 16384`, which rows of
   `freqs_cis` does rank 2 need? Now answer again for zigzag (load-balanced)
   sharding.

## Next

→ [05 — Building the transformer and weight initialization](05-transformer-and-weight-init.md)
