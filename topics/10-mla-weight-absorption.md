# 10 — MLA Weight Absorption

> Video: 04:10:54 — MLA Weight Absorption
> Code: `torchfeather/model/model.py` — `absorb_mla_weights`, `forward_absorbed`

## Why this exists

Topics 08 and 09 established that absorption is algebraically valid on the
content path. This topic implements it, verifies it numerically, and then — the
part that actually matters — shows what it does to **arithmetic intensity**,
which is the quantity MLA was invented to fix and the one FLOP counting cannot
see.

The headline result, derived in §5: absorbed MLA achieves **30 FLOP/byte** in
decode against MHA's **1**, while keeping all 16 heads distinct. GQA at the same
cache size gets 8.

## 1. What we are building

Two derived matrices, computed once, offline, from the trained weights:

```
W_Q^abs,(i)  =  W_Q^{C,(i)} · W_UK^{(i)ᵀ}        [d × d_c]     = 2048 × 512
W_O^abs,(i)  =  W_UV^{(i)}  · W_O^{(i)}          [d_c × d]     = 512 × 2048
```

and a forward pass that uses them to attend **directly against the 576-wide
cache**, never reconstructing per-head `K` or `V`.

## 2. The transpose bookkeeping

The single most error-prone part of implementing this is that PyTorch stores
`nn.Linear` weights **transposed** relative to the row-vector math convention:

```
math (row vectors):   y = x W ,        W ∈ R^{d_in × d_out}
PyTorch:              nn.Linear.weight ∈ R^{d_out × d_in}  =  Wᵀ
```

So every matrix in the code is the transpose of the one in topic 09's algebra.
Rather than fight it, work out the target once:

```
want:    W_Q^abs = W_Q^C W_UKᵀ                        [d × d_c]
store:  (W_Q^abs)ᵀ = W_UK (W_Q^C)ᵀ                    [d_c × d]
                     └─┬─┘ └───┬──┘
                       │       └─ wq_nope   (as stored: [d_h^C, d])
                       └───────── w_uk.transpose  (stored [d_h^C, d_c] → [d_c, d_h^C])
```

which is exactly the `bmm` in the source. Same on the output side:

```
want:    W_O^abs = W_UV W_O                           [d_c × d]
store:  (W_O^abs)ᵀ = W_Oᵀ W_UVᵀ                       [d × d_c]
                     └─┬┘ └──┬─┘
                       │      └─ w_uv     (as stored: [d_h^V, d_c])
                       └──────── w_o      (permuted to  [d, d_h^V])
```

## 3. The implementation, read against the algebra

### 3.1 Reshape the stored weights into per-head blocks

```python
# wq.weight : [h·(d_h^C + d_h^R), d]  →  [h, d_h^C + d_h^R, d]
wq = self.wq.weight.view(n_heads, qk_nope_head_dim + qk_rope_head_dim, dim)
wq_nope, wq_rope = torch.split(wq, [qk_nope_head_dim, qk_rope_head_dim], dim=1)
#   wq_nope : [h, 128, d]      ← absorbable (content)
#   wq_rope : [h,  64, d]      ← untouched  (position)

# wkv_b.weight : [h·(d_h^C + d_h^V), d_c]  →  [h, d_h^C + d_h^V, d_c]
wkv_b = self.wkv_b.weight.view(n_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank)
w_uk, w_uv = torch.split(wkv_b, [qk_nope_head_dim, v_head_dim], dim=1)
#   w_uk : [h, 128, d_c]       ← absorbs into the query
#   w_uv : [h, 128, d_c]       ← absorbs into the output
```

The two `split`s undo the two fusions from topic 07 §1. This is the moment the
"seven logical matrices" claim pays off — the code has to know the concatenation
layout to take the product apart.

### 3.2 Absorb into the query

```python
# [h, d_c, d_h^C] @ [h, d_h^C, d]  →  [h, d_c, d]
wq_abs_nope = torch.bmm(
    w_uk.float().transpose(1, 2),      # W_UK
    wq_nope.float(),                   # (W_Q^C)ᵀ
).to(dtype=dtype)

# each new query head is [ absorbed content | original RoPE ]
wq_abs = torch.cat([wq_abs_nope, wq_rope], dim=1) \
              .reshape(n_heads * (kv_lora_rank + qk_rope_head_dim), dim)

self.wq_abs = nn.Linear(dim, n_heads * (kv_lora_rank + qk_rope_head_dim),
                        bias=False, device=device, dtype=dtype)
self.wq_abs.weight.copy_(wq_abs)
self.wq_abs.requires_grad_(False)
```

Three things to note:

- **The per-head query width grows from 192 to 576** (`d_c + d_h^R = 512 + 64`).
  The absorbed query lives in latent space, which is wider than a head. This is
  the cost that makes absorption a *loss* during training (topic 08 §5).
- **`wq_rope` is concatenated unchanged.** The decoupled path is not absorbed —
  it cannot be (topic 09 §4). The layout `[absorbed_content(512) | rope(64)]`
  mirrors the cache layout `[latent(512) | k_pe(64)]` so the dot product lines
  up term by term.
- **`.float()` around the `bmm`, then cast back.** This is a product of two
  trained matrices; doing it in bf16 would compound two roundings into the
  derived weight, permanently. Compute in fp32, store in the model dtype.

### 3.3 Absorb into the output

```python
# wo.weight : [d, h·d_h^V]  →  [d, h, d_h^V]  →  [h, d, d_h^V]
w_o = self.wo.weight.view(dim, n_heads, v_head_dim).permute(1, 0, 2)

# [h, d, d_h^V] @ [h, d_h^V, d_c]  →  [h, d, d_c]
w_o_abs_per_head = torch.bmm(w_o.float(), w_uv.float()).to(dtype=dtype)

# [h, d, d_c] → [d, h, d_c] → [d, h·d_c]
w_o_abs = w_o_abs_per_head.permute(1, 0, 2).reshape(dim, n_heads * kv_lora_rank)

self.wo_abs = nn.Linear(n_heads * kv_lora_rank, dim, bias=False, ...)
self.wo_abs.weight.copy_(w_o_abs)
self.wo_abs.requires_grad_(False)
```

The double `permute` is pure index bookkeeping: `bmm` needs the head axis first,
the final `nn.Linear` needs `[d_out, d_in]` with heads interleaved into `d_in`
in the same order that `forward_absorbed` will concatenate them. Get the order
wrong and you get plausible-looking garbage — which is why §6 exists.

### 3.4 `requires_grad_(False)` — and why you cannot train in absorbed form

Both derived matrices are frozen. This is not an optimization; it is a
correctness requirement:

- `W_Q^abs` is `[2048 × 512]` = 1,048,576 entries, but it is a product of a
  `[2048 × 128]` and a `[128 × 512]`, so it has **rank ≤ 128**. Its true degrees
  of freedom are `2048·128 + 128·512 = 327,680`. Training it freely would move
  it off the low-rank manifold — you would be training a *different, larger*
  model, and there would be no way to project back to `(W_Q^C, W_UK)`.
- The absorbed form also **loses the ability to compute `V` explicitly**, which
  some training-time paths need.

So: `absorb_mla_weights` is a one-way, inference-only transformation, guarded by
`@torch.no_grad()`. Train in the factored form (topic 07); absorb for serving.
The `if self.q_lora_rank != 0: raise NotImplementedError()` guard at the top is
honest scaffolding — absorbing through a compressed query path is possible
(`W_QA W_QB W_UKᵀ`) but not implemented.

## 4. `forward_absorbed`

```python
q = self.wq_abs(x)                                   # [B, S, h·576]
q = q.view(B, S, self.n_heads, self.kv_lora_rank + self.qk_rope_head_dim)
q_nope, q_rope = torch.split(q, [self.kv_lora_rank, self.qk_rope_head_dim], -1)
q_rope = apply_rotary_emb(q_rope, freqs_cis)         # only the 64 RoPE dims
q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)   # [B, h, S, 576]

latent_raw, k_rope = torch.split(self.wkv_a(x),
                                 [self.kv_lora_rank, self.qk_rope_head_dim], -1)
latent = self.kv_norm(latent_raw)                    # ← this is what gets cached
k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis)

shared_cache = torch.cat([latent.unsqueeze(2), k_rope], dim=-1).transpose(1, 2)
#   [B, 1, S, 576]

k = shared_cache                                     # [B, 1, S, 576]
v = shared_cache[..., : self.kv_lora_rank]           # [B, 1, S, 512]  ← a VIEW

latent_output = self.inner_attention(q, k, v, scale=self.softmax_scale)
#   [B, h, S, 512]

latent_output = latent_output.transpose(1, 2).contiguous() \
                             .view(B, S, self.n_heads * self.kv_lora_rank)
return self.wo_abs(latent_output)                    # [B, S, d]
```

Four observations, each of which is the point of the exercise.

**(a) `K` and `V` are the same tensor.** `v` is a *slice* of `k` — the first 512
columns. No copy, no extra memory. The cache is one `[B, S, 576]` buffer; the
key uses all 576 columns, the value uses the first 512. This falls straight out
of the algebra: after absorption both the key and the value *are* the latent, and
the key merely has the 64 positional columns appended.

**(b) One KV head, `h` query heads.** `k` and `v` have head dimension 1 and are
broadcast across all 16 query heads. Structurally this is MQA — but each head
still has its own `W_Q^abs,(i)`, which encodes that head's own `W_UK^{(i)}`. **The
per-head key distinctness has moved from the key side to the query side.** That
sentence is the essence of weight absorption.

**(c) `d_v = 512 < d_qk = 576`, so the V-padding trick fires.** This is the
`if v_head_dim < q_head_dim: v = F.pad(...)` branch in the attention wrapper
(topic 06 §7.2) — now with a 64-column pad rather than the training path's
64-column pad on a different pair of widths. Same mechanism, different numbers.

**(d) `kv_norm` is applied *before* caching.** So the cached bytes are already
normalized and everything downstream of the cache is purely linear — which is
precisely the condition absorption needs. Had the norm been placed *after* the
up-projection, absorption would be impossible: a nonlinearity between `W_UK` and
the score.

## 5. The payoff: arithmetic intensity

Now the calculation that justifies the whole architecture. Decode, per token, per
layer, `h = 16`, bf16.

### Absorbed MLA

```
bytes read  =  (d_c + d_h^R) · 2                      =  576 · 2   =  1152 B
                (one cache, shared by all heads)

FLOPs       =  h · [ 2·(d_c + d_h^R)   ← q·k over 576 dims
                   + 2· d_c          ] ← "A·V" over 512 dims
            =  16 · (1152 + 1024)                     =  34,816 FLOPs

AI          =  34816 / 1152                           =  30.2 FLOP/byte
```

### The comparison table

| Scheme | cached values | bytes | FLOPs | **AI** | distinct heads |
|--------|---------------|-------|-------|--------|----------------|
| MHA | 4096 | 8192 | 8,192 | **1.0** | 16 |
| GQA-8 | 2048 | 4096 | 8,192 | **2.0** | 8 |
| GQA-2 | 512 | 1024 | 8,192 | **8.0** | 2 |
| MQA | 256 | 512 | 8,192 | **16.0** | 1 |
| **MLA absorbed** | **576** | **1152** | **34,816** | **30.2** | **16** |

Read the table carefully, because the interesting content is in the two right
columns.

**MLA does 4.25× MORE arithmetic than MHA.** It is not saving FLOPs — it is
spending them. And that is exactly right: at `AI = 1` against a ridge point of
295 (topic 06 §3.1), the GPU is idle 99.7% of the time waiting on memory. FLOPs
are *free* in that regime. The only currency that matters is bytes.

```
┌───────────────────────────────────────────────────────────────┐
│  In a memory-bound regime, the correct move is to spend       │
│  arithmetic to buy bandwidth. MLA is that trade, executed     │
│  precisely.                                                   │
└───────────────────────────────────────────────────────────────┘
```

**Against GQA at the same cache size** — the fair comparison, since both are
cache-reduction techniques — MLA is 3.8× higher in arithmetic intensity *and*
keeps 16 distinct heads where GQA-2 keeps 2. That is a win on both axes.

**Against MQA**, be honest: MQA moves fewer bytes (512 vs 1152), so pure decode
*latency* for the attention step favours MQA by ~2.25×. MLA's case is that it
achieves near-MQA bandwidth while retaining 16 genuinely distinct heads, and the
quality difference is what motivated the design. If you only cared about latency
and not about quality, MQA would be the answer.

### Decode speedup, since decode is bandwidth-bound

While `AI < 295`, wall-clock time is proportional to bytes moved:

```
MLA vs MHA attention step:   8192 / 1152  =  7.1×  faster
```

The extra FLOPs cost nothing, because they are hidden behind the memory traffic
that we were paying anyway. The 4.25× arithmetic increase is invisible until
`AI` approaches 295 — which would require roughly 10× more heads.

## 6. Verify it — and this is not optional

Weight absorption is a transpose-heavy, index-heavy transformation of trained
weights. There is no loss curve to warn you: a wrong `permute` produces a model
that runs, produces plausible-looking activations, and generates nonsense. The
only defence is an exact numerical equivalence test.

```python
import torch, torch.nn.functional as F
from torchfeather.model.model import Attention
from torchfeather.model.model_args import DeepSeekV3ModelArgs
from torchfeather.model.rope import precompute_freqs_cis

torch.manual_seed(0)

class MathAttn(torch.nn.Module):
    """CPU-runnable stand-in: the flash/efficient/cudnn kernels the real
    wrapper requests are unavailable on CPU ('RuntimeError: No available
    kernel'), and we also need to broadcast the single KV head by hand."""
    def forward(self, q, k, v, *, scale=None):
        if k.size(1) != q.size(1):
            k = k.expand(-1, q.size(1), -1, -1)
            v = v.expand(-1, q.size(1), -1, -1)
        return F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

S = 64
args = DeepSeekV3ModelArgs(dim=128, n_heads=4, kv_lora_rank=32,
                           qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
                           max_seq_len=S, original_seq_len=S, q_lora_rank=0)
attn = Attention(args)
attn.init_weights(init_std=0.02)
attn.inner_attention = MathAttn()

cis = precompute_freqs_cis(args)
x = torch.randn(2, S, 128)

with torch.no_grad():
    out_naive = attn(x, cis)                    # topic 07's forward
    attn.absorb_mla_weights()
    out_absorbed = attn.forward_absorbed(x, cis)

print("max abs diff ", (out_naive - out_absorbed).abs().max().item())
print("output scale ", out_naive.abs().max().item())
print("wq_abs       ", tuple(attn.wq_abs.weight.shape))
print("wo_abs       ", tuple(attn.wo_abs.weight.shape))
print("frozen       ", attn.wq_abs.weight.requires_grad,
                       attn.wo_abs.weight.requires_grad)
```

Actual output:

```
max abs diff  1.862645149230957e-08
output scale  0.04680074006319046
wq_abs        (160, 128)        = [h·(d_c + d_h^R), d] = [4·(32+8), 128]  ✓
wo_abs        (128, 128)        = [d, h·d_c]           = [128, 4·32]      ✓
frozen        False False
```

`1.9e-8` on outputs of scale `0.047` is fp32 rounding noise, nothing more.
**The two forward passes compute the same function**, which is what topics 08
and 09 promised. Both derived shapes match the formulas in §1, and both matrices
are frozen.

Two practical notes from having actually run this:

- Absorption on CPU needs the math attention backend. The real
  `ScaledDotProductAttentionWrapper` pins `[FLASH, EFFICIENT, CUDNN]` with
  `set_priority=True`, and on CPU that raises `No available kernel`.
- The single-KV-head broadcast is not automatic in every PyTorch version. On GPU
  either the GQA-capable kernel path or an explicit `expand` handles it; if you
  see a head-count mismatch error, that is what it is.

## 7. When to use which form

```
                    │ training (S_q = S)      │ decode (S_q = 1)
────────────────────┼─────────────────────────┼────────────────────────
form                │ forward (factored)      │ forward_absorbed
cached / held       │ nothing cached          │ 576 values/token/layer
FLOPs per step      │ 3.7× cheaper (t.08 §5)  │ ~110× cheaper (t.08 §5)
AI                  │ compute-bound anyway    │ 30 vs 1  → 7.1× faster
grads               │ required                │ frozen
```

Two forward passes, one function, chosen by regime. If you remember one sentence
from Part I, make it this: **the same mathematics has different optimal
implementations in different shape regimes, and a serious framework ships both.**

## 8. Check yourself

1. Write `(W_Q^abs)ᵀ` in terms of the stored `nn.Linear` weights and explain why
   the transpose appears.
2. Why is `.float()` used around both `bmm` calls?
3. `W_Q^abs` is `[2048 × 512]` but has rank ≤ 128. What are its true degrees of
   freedom, and what breaks if you train it directly?
4. In `forward_absorbed`, `v = shared_cache[..., :kv_lora_rank]`. Explain why `K`
   and `V` can be the same storage, from the algebra of topic 09.
5. Absorbed MLA has one KV head and `h` query heads — structurally MQA. Where did
   the per-head key distinctness go?
6. Compute `AI` for absorbed MLA and for GQA-2 at the same cache size. Which
   wins, and on which axis?
7. MLA does 4.25× more arithmetic than MHA in decode. Explain why that is the
   *correct* design decision, referring to the ridge point.
8. Why is `absorb_mla_weights` decorated `@torch.no_grad()` and why does it
   `raise NotImplementedError` for `q_lora_rank != 0`?
9. `kv_norm` is applied before the value is cached. What would break if it were
   applied after the up-projection instead?

## Part I complete

You can now count a model's parameters, FLOPs, and cached bytes; derive RoPE and
extend it with YaRN; build a pre-norm transformer whose initialization you can
justify line by line; and derive MLA from a bandwidth requirement including the
non-obvious RoPE decoupling.

More importantly, you have the two tools Part III runs on:

- **Block matrix multiplication** (topic 08) — which is already tensor
  parallelism, waiting to be distributed.
- **The roofline discipline** — count FLOPs, count bytes, take the ratio, compare
  to the hardware. We will apply exactly this to every collective operation.

## Next

→ [11 — Autograd and the mathematics of distributed training](11-autograd-and-distributed-math.md)
