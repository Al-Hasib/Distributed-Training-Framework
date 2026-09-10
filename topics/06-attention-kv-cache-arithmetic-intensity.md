# 06 — Attention, the KV Cache and Arithmetic Intensity

> Video: 02:19:12 — Attention, the KV Cache and Arithmetic Intensity
> Code: `torchfeather/model/attention.py`

## Why this exists

This topic answers one question: **why does MLA exist?**

The answer is not "to save parameters" (topic 02 showed it barely does) and not
"to improve quality". It is a *memory-bandwidth* argument about a phase of the
model's life — autoregressive decoding — that does not even occur during
training. We are building a training framework, so it is fair to ask why we
care. Two reasons:

1. You cannot train an architecture you do not understand, and MLA's peculiar
   shape (`qk_nope_head_dim` + `qk_rope_head_dim`, a `kv_lora_rank` bottleneck,
   a key head shared across all heads) is unexplainable without this argument.
2. The same reasoning — count FLOPs, count bytes, take the ratio, compare to the
   hardware — is *exactly* the tool we will use for every collective
   communication in Part III. Learn it here on a small problem.

## 1. Attention, and its two matmuls

Per head, per layer:

```
Q = x W_Q     [S, d_qk]        K = x W_K   [S, d_qk]      V = x W_V   [S, d_v]

              ┌ Q Kᵀ ┐
A = softmax   │ ──── │  ⊙ causal_mask        [S, S]
              └  √d  ┘

out = A V                                    [S, d_v]
```

Note the asymmetry that MLA will exploit: `d_qk` and `d_v` need not be equal.
`Q` and `K` only ever appear inside a dot product, so `d_qk` sets the *rank of
the similarity function*. `V` only appears in a weighted average, so `d_v` sets
the *width of the information channel*. They are different jobs. Standard
implementations set them equal out of habit; MLA sets `d_qk = 192, d_v = 128`.

## 2. The KV cache

### 2.1 Why it exists

Autoregressive generation produces one token at a time. To generate token `t`,
attention needs `K` and `V` for **all** positions `1…t`. Naively you would
re-run the whole prefix through the model at every step — `O(t)` work per token,
`O(S²)` for a sequence, all of it recomputing values that cannot have changed
(`K_n` depends only on `x_n`, and `x_n` is fixed once generated).

So cache them. Per token, per layer, MHA must store:

```
K: h · d_h     values
V: h · d_h     values
   ─────────
   2 · h · d_h  values per token per layer
```

Total, for a batch of `B` sequences of length `S`:

```
┌──────────────────────────────────────────────────┐
│  M_kv = 2 · B · S · L · h · d_h · bytes_per_elem │
└──────────────────────────────────────────────────┘
```

### 2.2 It is enormous

Note what is *absent* from that formula: **the parameter count**. The KV cache
scales with `L·h·d_h`, `S`, and `B` — and it grows *per user*, while parameters
are shared across all users. That is why it dominates serving cost.

Llama-2-70B (`L=80, h=64, d_h=128`), bf16, one sequence of 32k:

```
2 · 32768 · 80 · 64 · 128 · 2 B = 85.9 GB
```

**More than the 70B of weights in bf16 (140 GB) requires per 64k of context** —
and that is for *one* user. A server holding 32 concurrent sessions would need
2.7 TB of cache. This is the wall that MQA, GQA and MLA were all invented to
break.

For our reference config (`L=27, h=16, d_h=128`), per token per layer:

| Scheme | values / token / layer | bytes (bf16) | vs MHA |
|--------|------------------------|--------------|--------|
| MHA | `2·h·d_h` = 4096 | 8192 | 1× |
| GQA, 8 groups | `2·8·d_h` = 2048 | 4096 | 2× |
| GQA, 2 groups | `2·2·d_h` = 512 | 1024 | 8× |
| MQA (1 group) | `2·1·d_h` = 256 | 512 | 16× |
| **MLA** | `d_c + d_h^R` = **576** | **1152** | **7.1×** |

At `h = 128` (DeepSeek-V2's real width) MHA would need `32768` values against
MLA's same 576 — a **57×** reduction. MLA's cache cost is *independent of the
number of heads*, which is what makes it scale.

That last row is the whole point, but it raises the obvious objection: MQA
already achieves 16× and is far simpler. Why not just use MQA? Because the
comparison above measures the wrong thing. To see the real trade we need
arithmetic intensity.

## 3. Arithmetic intensity and the roofline

### 3.1 The model

Any kernel does some arithmetic and moves some bytes. Define

```
                    FLOPs performed
   AI  =  ─────────────────────────────────      [FLOP / byte]
           bytes moved to/from HBM
```

The hardware has its own ratio — the **ridge point**:

```
              peak FLOP/s
   AI* =  ─────────────────
            peak bytes/s
```

For an H100 SXM in bf16: `989e12 / 3.35e12 = 295 FLOP/byte`.

```
   AI < AI*   →  memory-bound.   You are paying for bandwidth; the FLOPs are free.
   AI > AI*   →  compute-bound.  You are paying for arithmetic; the loads are free.
```

**295 is a shockingly high bar.** Modern accelerators have added FLOPs far faster
than bandwidth, so almost everything that is not a large matmul is memory-bound.
Internalize the number: any kernel below ~300 FLOP/byte is wasting the machine.

### 3.2 Training: comfortably compute-bound

One linear layer, `T` tokens through `W ∈ R^{d_in×d_out}`:

```
FLOPs  = 2 · T · d_in · d_out
bytes  = 2·(T·d_in + T·d_out + d_in·d_out)      # bf16: activations in/out + weights

AI ≈  T · d_in · d_out / (T·d_in + T·d_out + d_in·d_out)
```

With `T = 8192, d_in = d_out = 2048`: `AI ≈ 1365 FLOP/byte`. Comfortably above
295 — compute-bound, as it must be for training to be efficient at all. The
reason is simply that `T` is large: **every weight you load is reused `T`
times.**

### 3.3 Decode: catastrophically memory-bound

Now generate **one** token with a cache of `S_kv` entries. Per layer:

```
FLOPs:  Q Kᵀ  →  2 · h · d_h · S_kv
        A V   →  2 · h · d_h · S_kv
                 ────────────────────
                 4 · h · d_h · S_kv

bytes:  read K and V from HBM  →  2 · h · d_h · S_kv · 2 B  =  4 · h · d_h · S_kv
```

Therefore

```
┌──────────────────────────────────────────────────┐
│         4 · h · d_h · S_kv                       │
│  AI  =  ───────────────────  =  1 FLOP / byte    │
│         4 · h · d_h · S_kv                       │
└──────────────────────────────────────────────────┘
```

**Exactly 1**, independent of `S_kv`, `h`, `d_h`, and the model size. Against a
ridge point of 295, decode attention runs at roughly **0.3% of peak FLOPs**. The
GPU is doing nothing but streaming the KV cache through the memory bus.

The reason is structural and worth stating plainly: in decode, **every value
loaded from the cache is used exactly once**. There is no reuse to amortize the
load. Attention in decode is not a matmul; it is a `gemv` — a matrix-vector
product — and `gemv` is always memory-bound.

### 3.4 Batching does not save you

The instinctive fix is a bigger batch. It fails, because **each sequence has its
own cache**: increase `B` and the FLOPs *and* the bytes both scale by `B`. `AI`
stays 1.

This is the crucial difference from the FFN, where a bigger batch reuses the same
weights and `AI` grows linearly in `B`. It is why serving systems can batch their
way to good FFN utilization and still be attention-bound.

### 3.5 What actually raises AI: decoupling FLOPs from bytes

Look again at the ratio. To improve it we must make the numerator and
denominator scale differently. With GQA (`h` query heads, `h_kv` key/value
heads, each `K` head shared by `h/h_kv` queries):

```
FLOPs = 4 · h    · d_h · S_kv        ← unchanged: every query head still attends
bytes = 4 · h_kv · d_h · S_kv        ← reduced:   fewer distinct K/V to load

┌────────────────────┐
│  AI  =  h / h_kv   │
└────────────────────┘
```

**That** is the actual mechanism of MQA and GQA. The cache shrinks, but more
importantly the loaded bytes are *reused across query heads*, so arithmetic
intensity rises to the group size. MQA (`h_kv = 1`) gives `AI = h` — for
`h = 64`, a 64× improvement, from 1 to 64 FLOP/byte. Still below 295, but a
different universe from 1.

And the cost of MQA: `h_kv = 1` means all heads share one similarity subspace
and one value channel. This measurably hurts quality, which is why GQA exists as
the compromise (`h_kv = 8` is the common choice) and why Shazeer's original
paper is titled *"One Write-Head is All You Need"* somewhat optimistically.

## 4. The MLA idea, in one paragraph

MQA/GQA reduce the cache by making heads *share* K and V — paying in expressive
power, because the shared subspace is genuinely smaller.

MLA instead reduces the cache by **compressing**: store one low-rank latent
`c_t ∈ R^{d_c}` per token, and reconstruct each head's full, *distinct* `K` and
`V` from it with per-head up-projections `W_UK`, `W_UV`.

```
MHA:  cache K_t, V_t  per head           →  2 · h · d_h        values
GQA:  cache K_t, V_t  per group          →  2 · h_kv · d_h     values
MLA:  cache c_t = x_t W_DKV              →  d_c                values
      reconstruct  K_t^(i) = c_t W_UK^(i),  V_t^(i) = c_t W_UV^(i)
```

Every head still gets its own `d_h`-dimensional key and value — no sharing of
subspaces — but all `h` of them are generated from a single `d_c`-dimensional
latent. The information bottleneck is `d_c = 512` rather than `h·d_h = 2048`,
so it is a genuine restriction; but it is a *low-rank* restriction spread across
all heads rather than a *tied-parameter* restriction, and empirically it costs
far less quality than GQA at the same cache size.

And here is the part that makes it more than a storage trick: because
`W_UK` is a *linear* map, it can be **absorbed** into `W_Q` so that the
reconstruction never happens at all during decode. That raises arithmetic
intensity the same way GQA does, without tying any heads together. This is the
weight-absorption argument, and it is important enough to get its own topic
(topic 10). Two obstacles must be cleared first:

- The algebra of factored attention — topic 08 (block matrix multiplication).
- RoPE **breaks** the absorption, because `R_m` sits between `W_Q` and `W_UK`
  and rotations do not commute with arbitrary matrices. The fix — apply RoPE to
  only a few dimensions, kept on a separate path — is *decoupled RoPE*,
  topic 09. This is why the head dimension is split `128 + 64`, and why
  `precompute_freqs_cis` uses `dim = qk_rope_head_dim = 64` rather than the full
  192 (topic 04 §6).

So the strange shape of MLA is fully determined by two constraints: minimize
cached bytes, and stay compatible with RoPE.

## 5. Training-time reality: what actually matters for us

Be clear about what carries over to training, since that is what we are building.

**The KV cache does not exist in training.** All positions are processed at once,
so `K` and `V` for the whole sequence are computed, used and discarded (or kept
as activations for backward, which is a different budget). Nothing in section 3.3
applies.

What *does* apply in training:

| Inference concern | Training analogue |
|-------------------|-------------------|
| KV cache bytes | activation memory, `O(B·S·d·L)` — topic 02 §4.2 |
| decode `AI = 1` | not applicable; training attention is compute-bound |
| `S_kv` growth | the `S²` FLOP term, 67% of compute at `S=16384` — topic 02 §3.3 |
| cache too big for one GPU | activations too big for one GPU → context parallelism, topic 23 |

And one thing that carries over directly: **MLA changes what context parallelism
must communicate.** Ring attention (topic 23) passes `K` and `V` blocks around a
ring of devices. Under MHA that is `2·h·d_h = 4096` values per token; under MLA
you can instead pass the latent `c_t` plus `k_pe`, i.e. 576 values — a 7×
reduction in ring traffic. The inference-motivated design pays off again in
distributed training, for the same underlying reason: fewer bytes per token.

## 6. FlashAttention, briefly

One more `S²` problem, which the prerequisite video covers in full. The naive
implementation materializes `A = softmax(QKᵀ/√d)`, an `[S, S]` tensor. At
`S = 16384` in bf16 that is 537 MB **per head per layer** — and it must be kept
for the backward pass.

FlashAttention never materializes it: it tiles `Q`, `K`, `V`, and uses the
**online softmax** recurrence to keep a running maximum and running
denominator, rescaling the accumulated output as new blocks arrive. Memory drops
from `O(S²)` to `O(S)`; the `S²` FLOPs remain but are performed in on-chip SRAM,
so `AI` rises by roughly the tile size.

Two properties matter for later topics:

- **It is block-wise.** Attention over a block of keys can be computed
  independently and *merged* afterwards using the running statistics. That
  mergeability is exactly what ring attention needs to split `K`/`V` across
  devices (topic 23) — ring attention is FlashAttention with the block loop
  distributed instead of local.
- **It does not materialize a bias matrix.** Which is why RoPE (a per-token
  transform) is compatible and T5-style additive relative bias is not (topic 03
  §1).

## 7. The implementation, and its two subtleties

```python
class ScaledDotProductAttentionWrapper(torch.nn.Module):
    def forward(self, q, k, v, *, scale=None):
        v_head_dim, q_head_dim = v.shape[-1], q.shape[-1]

        tp_spec = None
        if isinstance(q, DTensor):
            tp_spec = (q.device_mesh, q.placements)
            q, k, v = q.to_local(), k.to_local(), v.to_local()

        if v_head_dim < q_head_dim:
            v = F.pad(v, (0, q_head_dim - v_head_dim))

        with sdpa_kernel([SDPBackend.FLASH_ATTENTION,
                          SDPBackend.EFFICIENT_ATTENTION,
                          SDPBackend.CUDNN_ATTENTION], set_priority=True):
            out = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

        if out.shape[-1] != v_head_dim:
            out = out[..., :v_head_dim]

        if tp_spec is not None:
            out = DTensor.from_local(out, tp_spec[0], tp_spec[1], run_check=False)
        return out
```

### 7.1 Why a `Module` wrapping a function

`F.scaled_dot_product_attention` is a function, and PyTorch's context-parallel
machinery (`_ContextParallel`) applies to **modules** — it installs hooks that
intercept `q, k, v` and replace local attention with ring attention. A function
has nothing to hook. So the wrapper exists purely to give CP an attachment
point, which is why the docstring insists that `q, k, v` be the first three
positional arguments: the hook matches on position.

This is a good early example of a recurring theme: **parallelism is applied by
substituting modules**, so your model's module boundaries determine which
parallelizations are even expressible. Design them deliberately.

### 7.2 The V-padding trick

Flash Attention kernels require `d_qk == d_v`. MLA has `192 ≠ 128`. Rather than
fall back to the slow math backend, pad `V` with 64 zero columns:

```
softmax(QKᵀ/s) · [V | 0]  =  [ softmax(QKᵀ/s)·V | 0 ]
```

Zero columns of the right operand produce zero columns of the product, so
trimming `out[..., :v_head_dim]` recovers the exact result. **Mathematically
lossless, and it buys the fast kernel.** The cost is `4/12 = 33%` wasted FLOPs
in the `AV` matmul and a slightly larger `V` tensor — a good trade against
losing Flash Attention entirely, which would be a several-fold slowdown *and*
would reintroduce the `O(S²)` memory.

### 7.3 The DTensor round-trip

The comment in the source is the clearest explanation of a genuinely subtle bug,
so read it as written:

> When Tensor Parallel (TP) and Context Parallel (CP) are both active, q, k, v
> arrive as DTensors on the TP mesh. We must convert them to local tensors so
> that CP's input hook can properly wrap them on the CP mesh for ring attention.
> Without this, ring attention would incorrectly communicate on the TP process
> group instead of the CP process group.

The tensor is sharded on **two different mesh dimensions at once**: by head
across `tp`, by sequence position across `cp`. The DTensor arriving here is
labelled with its `tp` placement, and if CP's hook wraps that, the ring rotates
`K`/`V` blocks among the wrong set of ranks — producing wrong numbers, not an
error. So: strip to local, let CP do its thing on the CP mesh, then re-wrap with
the saved `tp_spec` so the downstream `wo` (a `RowwiseParallel` module) still
receives a correctly-labelled DTensor.

A second, smaller reason is also in the comment: `F.pad` has no registered
DTensor sharding rule, so calling it on a DTensor changes `V`'s placements and
the CP handler then fails with *"inputs need to be redistributed"*.

You are not expected to fully follow this yet — it needs topics 20–23. It is
here because it is the honest answer to "what does composing two parallelisms
actually cost?": **the compositions interact at the tensor-layout level, and the
interaction is not automatic.** Section 7.3 is a nine-line function containing
the entire theme of Part III.

## 8. Check yourself

1. Derive `M_kv = 2·B·S·L·h·d_h·bytes` and compute it for Llama-2-70B at
   `S = 32768`, `B = 1`. Compare to the weights.
2. Show `AI = 1 FLOP/byte` for decode attention. Why is it independent of `S_kv`?
3. Why does increasing the batch size raise `AI` for the FFN but not for
   attention?
4. Derive `AI = h/h_kv` for GQA. What does MQA give for `h = 64`, and is that
   above or below the H100 ridge point?
5. MLA reduces our config's cache 7.1× and DeepSeek-V2's 57×. Explain the
   difference from the formulas.
6. Both MQA and MLA shrink the cache. State precisely what each *gives up*, and
   why MLA's sacrifice is considered cheaper.
7. Padding `V` with zeros wastes 33% of the `AV` FLOPs. Justify the trade.
8. Under TP+CP, why must `q, k, v` be converted to local tensors before ring
   attention runs? What is the observable symptom if you skip it?
9. The KV cache does not exist during training. Give two ways MLA nevertheless
   changes the *training* system.

## Next

→ [07 — Coding Multi-head Latent Attention (MLA)](07-coding-mla.md)
