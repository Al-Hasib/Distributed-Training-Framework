# 07 — Coding Multi-head Latent Attention (MLA)

> Video: 03:02:56 — Coding Multi-head Latent Attention (MLA)
> Code: `torchfeather/model/model.py` — `Attention.__init__`, `Attention.forward`

## Why this exists

Topic 06 gave the motivation. This topic writes the module, tracks every shape,
and explains each of the seven weight matrices. We deliberately code the
**training-time** forward pass here — the straightforward one that reconstructs
full `K` and `V`. The clever inference-time rearrangement is topics 08–10.

That order is not pedagogical convenience. The two forms compute *the same
function* but have different FLOP/memory profiles, and **training genuinely wants
the naive form**. Understanding why is the payoff of topic 10.

## 1. The seven matrices

```python
self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim      # 128 + 64 = 192

# query path (q_lora_rank == 0, so no compression)
self.wq    = nn.Linear(dim, n_heads * qk_head_dim, bias=False)          # 2048 → 3072

# key/value path: compress, normalize, decompress
self.wkv_a = nn.Linear(dim, kv_lora_rank + qk_rope_head_dim, bias=False) # 2048 → 576
self.kv_norm = nn.RMSNorm(kv_lora_rank, eps=norm_eps)                    # 512
self.wkv_b = nn.Linear(kv_lora_rank,
                       n_heads * (qk_nope_head_dim + v_head_dim),        # 512 → 4096
                       bias=False)

# output
self.wo    = nn.Linear(n_heads * v_head_dim, dim, bias=False)            # 2048 → 2048
```

Four `nn.Linear`s, but **seven logical matrices**, because two of them are
concatenations that get split in `forward`:

```
wkv_a  =  [ W_DKV | W_KR ]        512 + 64  columns
          ─┬────── ─┬───
           │        └─ produces k_pe: the decoupled RoPE key, SHARED by all heads
           └─ produces the latent c_t: the only thing worth caching

wkv_b  =  [ W_UK | W_UV ]         per head: 128 + 128 rows
           ─┬───   ─┬──
            │       └─ reconstructs this head's V
            └─ reconstructs this head's content ("nope") key
```

Fusing them into single `nn.Linear`s is a real optimization, not cosmetic: one
`[2048 × 576]` GEMM instead of two, better tensor-core occupancy, one kernel
launch. The same trick appears in every fused QKV projection. The price is that
`forward` is littered with `torch.split`, and that tensor parallelism must know
the concatenation structure to shard correctly (topic 22).

### 1.1 The dimension budget, per head

```
query / key:   d_qk = 192  =  128 (content, no position)  +  64 (RoPE)
value:         d_v  = 128
latent:        d_c  = 512   ← shared across all 16 heads
```

Three things to notice:

- **`d_qk ≠ d_v`** (192 vs 128). Legal because `Q`/`K` only enter a dot product
  while `V` only enters a weighted average (topic 06 §1). It is also what forces
  the V-padding trick in the attention wrapper (topic 06 §7.2).
- **`d_c = 512 > d_qk = 192`.** The latent is *wider* than any single head's key.
  It has to be: it must serve all 16 heads' keys *and* values. Compare
  `d_c = 512` against the `h·(d_h^C + d_h^V) = 4096` it reconstructs — an 8×
  low-rank bottleneck.
- **The 128/64 split is not a modelling decision.** It is forced by weight
  absorption's incompatibility with RoPE (topic 09). Remember this when you are
  tempted to tune it.

### 1.2 `q_lora_rank`, and why it is off

```python
# As stated in the DeepSeek V2 paper, this helps only reduce the activation
# memory, but doesn't influence the amount of cache. We won't be using it.
if self.q_lora_rank == 0:
    self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
else:
    self.wq_a   = nn.Linear(self.dim, self.q_lora_rank, bias=False)
    self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
    self.wq_b   = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim,
                            bias=False)
```

DeepSeek-V2/V3 optionally compress the query path too (`q_lora_rank = 1536`),
and the comment states exactly why it is a different kind of win: **queries are
never cached.** In decode you have exactly one query — the current token —
whereas keys and values accumulate over the whole context. So compressing `Q`
does nothing for the `O(S)` cache; it only reduces the transient activation
`[B, S, h·192]` during training. Real, but a second-order concern, and it adds
a matrix and a norm. Off here, and topic 10's absorption code refuses to run
with it on (`if self.q_lora_rank != 0: raise NotImplementedError()`).

## 2. The forward pass, shape by shape

`B = batch`, `S = seq_len`, `h = 16`.

### Step 1 — queries

```python
q = self.wq(x)                                    # [B, S, h*192] = [B, S, 3072]
q = q.view(batch_size, seq_len, -1, self.qk_head_dim)   # [B, S, 16, 192]
q_nope, q_pe = torch.split(q, [128, 64], dim=-1)  # [B,S,16,128], [B,S,16,64]
q_pe = apply_rotary_emb(q_pe, freqs_cis)          # rotate ONLY the 64 RoPE dims
q = torch.cat([q_nope, q_pe], dim=-1)             # [B, S, 16, 192]
```

The split/rotate/concat is the whole of "decoupled RoPE" on the query side.
128 dimensions carry pure content and no position at all; 64 carry position.
`view(..., -1, qk_head_dim)` uses `-1` for the head count rather than `n_heads`
— deliberate, because under tensor parallelism this rank only holds
`h/P_tp` heads and the shape must adapt (topic 22).

### Step 2 — the compressed latent

```python
kv = self.wkv_a(x)                                # [B, S, 576]
kv, k_pe = torch.split(kv, [512, 64], dim=-1)     # [B,S,512], [B,S,64]
```

`kv` (512) is the latent `c_t`. `k_pe` (64) is the RoPE key.

**`k_pe` has no head dimension.** One 64-dimensional RoPE key for the whole
token, shared by all 16 heads. That is MQA — but applied *only to the 64
positional dimensions*, while the 128 content dimensions stay fully per-head.
MLA is, in this precise sense, a hybrid: MQA where sharing is cheap
(position is the same for every head anyway), low-rank-per-head where sharing
would cost quality.

```python
k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)   # [B, S, 1, 64]
```

The `unsqueeze(2)` inserts the head axis of size 1 because `apply_rotary_emb`
expects `[B, S, h, d]` (topic 04 §6).

### Step 3 — reconstruct per-head K and V

```python
kv = self.wkv_b(self.kv_norm(kv))                 # [B, S, h*(128+128)] = [B,S,4096]
kv = kv.view(batch_size, seq_len, -1, 128 + 128)  # [B, S, 16, 256]
k_nope, v = torch.split(kv, [128, 128], dim=-1)   # [B,S,16,128], [B,S,16,128]
```

**`kv_norm` before the up-projection.** An RMSNorm sitting on the 512-dim
latent. Two reasons it belongs there:

- The latent is the model's information bottleneck *and* the cached quantity. Its
  scale is fed to 16 independent up-projections; without normalization the whole
  attention block inherits whatever scale `wkv_a` drifts to. Normalizing pins it.
- It mirrors standard LoRA practice (normalize between down- and
  up-projection) and, concretely, it makes the *cached* value scale-stable, which
  matters if you ever quantize the cache.

Note that `kv_norm` is applied to the latent **before** caching in the absorbed
path (topic 10 caches `self.kv_norm(latent_raw)`), so the norm is on the
cache-side of the boundary. That is a deliberate choice: it means the cached
bytes are already normalized and the up-projection is a pure linear map — which
is what makes absorption possible.

### Step 4 — assemble the full keys

```python
k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)
#              [B,S,16,128]      [B,S,1,64] → [B,S,16,64]        → [B,S,16,192]
```

`expand`, not `repeat`. `expand` creates a view with **stride 0** on the head
axis: no data is copied, no memory is allocated, and all 16 heads read the same
64 numbers. `repeat` would materialize `[B, S, 16, 64]` — at `S = 16384, B = 1`
that is 16 MB of pointless copy per layer, per forward. Getting this wrong is a
silent 27× memory regression across the model.

(The subsequent `torch.cat` does have to materialize, so the saving is on the
broadcast itself rather than the concatenation. `expand` is still strictly
better, and the pattern matters more in the absorbed path where no `cat`
follows.)

### Step 5 — attention and output

```python
q = q.transpose(1, 2)          # [B, 16, S, 192]
k = k.transpose(1, 2)          # [B, 16, S, 192]
v = v.transpose(1, 2)          # [B, 16, S, 128]

output = self.inner_attention(q, k, v, scale=self.softmax_scale)   # [B,16,S,128]

output = output.transpose(1, 2).contiguous()      # [B, S, 16, 128]
output = output.view(batch_size, seq_len, -1)     # [B, S, 2048]
return self.wo(output)                            # [B, S, 2048]
```

Standard from here. The `transpose(1,2)` puts heads before sequence because
that is what SDPA wants; the `.contiguous()` after the reverse transpose is
required before `view` can merge the head and head-dim axes.

`softmax_scale` carries YaRN's temperature correction and is `192^{-1/2} ·
mscale²` — see topic 04 §5. Note it uses `qk_head_dim = 192`, the *full*
query/key width including the RoPE dims, which is correct: the dot product runs
over all 192.

### The complete shape trace

```
x                                      [B, S, 2048]
├─ wq ──────────────────────────────►   [B, S, 16, 192]
│   ├─ q_nope                           [B, S, 16, 128]        content
│   └─ q_pe  ──RoPE──►                  [B, S, 16,  64]        position
│
└─ wkv_a ───────────────────────────►   [B, S, 576]
    ├─ latent ──kv_norm──► CACHE THIS   [B, S, 512]            ← 512 values/token
    │      └─ wkv_b ──►                 [B, S, 16, 256]
    │           ├─ k_nope               [B, S, 16, 128]
    │           └─ v                    [B, S, 16, 128]
    └─ k_pe ──RoPE──► CACHE THIS        [B, S,  1,  64]        ←  64 values/token
                                                                 ═══
                                                                 576 total
        k = [k_nope | k_pe.expand]      [B, S, 16, 192]

        SDPA(q, k, v)                   [B, 16, S, 128]
        └─ wo ──►                       [B, S, 2048]
```

The two lines marked CACHE THIS are the 576 values from topic 06's table.
Everything else is transient.

## 3. The catch: this forward pass does not realize the win

Look at the trace again. To compute attention we materialized

```
k :  [B, S, 16, 192]      =  3072 values per token
v :  [B, S, 16, 128]      =  2048 values per token
```

That is **5120 values per token**, not 576. The compression bought us nothing
yet — we compressed to 512 and then immediately decompressed to 5120.

For **training** this is fine and in fact optimal:

- Nothing is cached in training; `k` and `v` are transient activations, computed
  once per forward and consumed by the attention kernel.
- `S` is large (16384), so the attention matmuls are compute-bound and
  full-width `k`/`v` feed Flash Attention exactly the layout it wants.
- The up-projection `wkv_b` costs `2·S·d_c·h·256` FLOPs and is amortized over
  the whole sequence.

For **decode** it is a disaster: you would have to either cache the 5120-wide
reconstruction (defeating the purpose) or re-run `wkv_b` on the entire cached
context at every single step (`O(S)` extra work per token, and it *lowers*
arithmetic intensity because you now load the latent *and* write 5120 values).

So MLA needs a **second, algebraically equivalent forward pass** in which the
reconstruction never happens — where `W_UK` is folded into `W_Q` and `W_UV` into
`W_O`, and attention is computed directly against the 576-wide cache.

That rearrangement requires being fluent in reassociating blocked matrix
products, which is topic 08; the derivation that shows it is exactly equivalent
(and where RoPE breaks it) is topic 09; the implementation is topic 10.

## 4. Cost accounting

Per layer, per token, forward only, `h = 16`:

| Matrix | FLOPs (2 × params) |
|--------|--------------------|
| `wq` | `2 · 2048 · 3072` = 12.6 M |
| `wkv_a` | `2 · 2048 · 576` = 2.4 M |
| `wkv_b` | `2 · 512 · 4096` = 4.2 M |
| `wo` | `2 · 2048 · 2048` = 8.4 M |
| **projections** | **27.6 M** |
| attention scores (`S = 16384`) | `2 · 16 · 16384 · (192+128)` = 167.8 M |

At long context the score matmuls are 6× the projections — consistent with
topic 02 §3.3. **MLA does not reduce training FLOPs.** It reduces cached bytes
at inference, and (topic 06 §5) ring-attention traffic under context
parallelism.

## 5. Verify it

```python
import torch
from torchfeather.model.model import Attention
from torchfeather.model.model_args import DeepSeekV3ModelArgs
from torchfeather.model.rope import precompute_freqs_cis

torch.manual_seed(0)
args = DeepSeekV3ModelArgs(dim=256, n_heads=4, kv_lora_rank=64,
                           qk_nope_head_dim=32, qk_rope_head_dim=16,
                           v_head_dim=32, max_seq_len=128, original_seq_len=128)
attn = Attention(args)
attn.init_weights(init_std=0.02)
cis = precompute_freqs_cis(args)

x = torch.randn(2, 128, 256)
out = attn(x, cis)
print(out.shape)                       # torch.Size([2, 128, 256])

# 1. causality: changing a LATER token must not change an EARLIER output
x2 = x.clone(); x2[:, 100:] = torch.randn_like(x2[:, 100:])
o1, o2 = attn(x, cis), attn(x2, cis)
assert torch.allclose(o1[:, :100], o2[:, :100], atol=1e-5)
print("causal ✓")

# 2. the cache really is d_c + d_h^R wide
kv = attn.wkv_a(x)
print("cached per token:", kv.shape[-1],
      "=", args.kv_lora_rank, "+", args.qk_rope_head_dim)

# 3. k_pe is genuinely shared across heads (stride-0 expand)
_, k_pe = torch.split(kv, [args.kv_lora_rank, args.qk_rope_head_dim], dim=-1)
e = k_pe.unsqueeze(2).expand(-1, -1, args.n_heads, -1)
print("expand strides:", e.stride(), "← 0 on the head axis means no copy")
```

Test 1 is the one worth keeping. Causality bugs in attention are invisible in the
loss curve for a long time (the model just quietly cheats and reports
suspiciously good numbers) and this three-line check catches every one of them.

## 6. Check yourself

1. Name all seven logical weight matrices and say what each produces.
2. `wkv_a` and `wkv_b` are each one `nn.Linear` holding two logical matrices.
   Give one benefit and one cost of the fusion.
3. Why is `d_c = 512` larger than `d_qk = 192`? What would go wrong at
   `d_c = 128`?
4. `k_pe` has no head dimension. Explain in what sense MLA is "MQA on the
   positional dimensions and low-rank on the content dimensions".
5. Why `expand` rather than `repeat` in step 4? Quantify the difference at
   `S = 16384, h = 16, L = 27`.
6. Why does the query path have no compression by default, even though `Q` is
   the same size as `K`?
7. `q.view(batch_size, seq_len, -1, self.qk_head_dim)` uses `-1` for the head
   count. What parallelism makes this necessary?
8. This forward pass materializes 5120 values per token from a 512-value latent.
   Explain why that is correct for training and fatal for decode.

## Next

→ [08 — Block matrix multiplication and MLA internals](08-block-matmul-and-mla-internals.md)
