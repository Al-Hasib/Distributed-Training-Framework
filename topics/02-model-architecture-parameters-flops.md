# 02 — Model Architecture, Parameters and Training FLOPs

> Video: 00:16:50 — Model Architecture, Parameters and Training FLOPs
> Code: `torchfeather/model/model_args.py`

## Why this exists

Every decision in distributed training is a trade between three budgets:
**FLOPs**, **bytes held** (memory) and **bytes moved** (communication). You
cannot make that trade without being able to count all three exactly. This topic
builds the counting machinery. It is the least glamorous chapter and the one you
will return to most.

Three questions we must be able to answer for any config, on paper, in a minute:

1. How many parameters does this model have — total, and *active* per token?
2. How many FLOPs does one training step cost?
3. How many bytes does the KV cache / activation stack cost?

## 1. The architecture we are counting

We target a DeepSeek-V3-shaped model: a pre-norm transformer with **MLA**
attention and a **MoE** FFN in all but the first few layers.

```
tokens [B, S]
   │
   ├─ embedding                                     V·d params
   │
   ├─ × L blocks:
   │     x = x + MLA(RMSNorm(x))
   │     x = x + FFN(RMSNorm(x))       FFN = dense SwiGLU  (first n_dense layers)
   │                                       or MoE          (all remaining layers)
   │
   ├─ RMSNorm                                       d params
   └─ output projection (untied)                    V·d params
                                                  → logits [B, S, V]
```

The reference config (`DeepSeekV3ModelArgs`) is:

| Field | Value | Symbol |
|-------|-------|--------|
| `dim` | 2048 | `d` |
| `n_layers` | 27 | `L` |
| `n_dense_layers` | 1 | — |
| `n_heads` | 16 | `h` |
| `vocab_size` | 102400 | `V` |
| `inter_dim` | 10944 | `d_ff` (dense) |
| `moe_inter_dim` | 1408 | `d_ff^E` (per expert) |
| `q_lora_rank` | 0 | no query compression |
| `kv_lora_rank` | 512 | `d_c` |
| `qk_nope_head_dim` | 128 | `d_h^C` |
| `qk_rope_head_dim` | 64 | `d_h^R` |
| `v_head_dim` | 128 | `d_h^V` |
| `max_seq_len` | 16384 | `S` |
| `num_experts` | 8 | `E` |
| `num_shared_experts` | 1 | `E_s` |
| `top_k` | 1 | `k` |

## 2. Counting parameters

### 2.1 The rule

A parameter is an entry of a weight matrix. Count matrices, multiply their two
dimensions, and add. Biases are omitted throughout (`bias=False` everywhere,
standard for LLMs — the following RMSNorm makes them redundant).

### 2.2 A dense (MHA + SwiGLU) block, for calibration

Before MLA, establish the classic result. With `h·d_h = d`:

**Attention** — four square matrices `W_Q, W_K, W_V, W_O`, each `d × d`:

```
N_attn = 4 d²
```

**SwiGLU FFN** — three matrices. `W_1` (gate) and `W_3` (up) are `d × d_ff`,
`W_2` (down) is `d_ff × d`:

```
FFN(x) = W_2 ( silu(x W_1) ⊙ (x W_3) )
N_ffn  = 3 d · d_ff
```

The conventional choice `d_ff = (8/3)d` makes `N_ffn = 8d²`, matching the
parameter count of the older 2-matrix `d_ff = 4d` GELU MLP. That is *why* the
`8/3` factor exists: it is the value that keeps SwiGLU parameter-neutral against
the MLP it replaced.

**Per block, therefore:**

```
N_block = 4d² + 8d² = 12 d²         (+ 2d for the two RMSNorms)
```

This `12d²` is the number worth memorizing. It gives the useful approximation
for a whole dense model:

```
N ≈ 12 L d² + 2 V d
```

**Sanity check on Llama-2-7B** (`L=32, d=4096, V=32000, d_ff=11008`):

```
12 · 32 · 4096²  = 6.44e9
3·d·d_ff vs 8d²  : 3·4096·11008 = 1.352e8 per layer vs 1.342e8  ✓ (d_ff ≈ 8d/3)
2 V d            = 2 · 32000 · 4096 = 2.62e8
total            ≈ 6.71e9  →  "7B"  ✓
```

### 2.3 An MLA block

MLA replaces `W_Q, W_K, W_V` with a low-rank/compressed factorization
(derived properly in topic 09; here we only count). With `q_lora_rank = 0` the
query path is uncompressed:

| Matrix | Shape | Count (this config) |
|--------|-------|---------------------|
| `wq` | `d × h(d_h^C + d_h^R)` = 2048 × 3072 | 6,291,456 |
| `wkv_a` (`W_DKV` ∥ `W_KR`) | `d × (d_c + d_h^R)` = 2048 × 576 | 1,179,648 |
| `wkv_b` (`W_UK` ∥ `W_UV`) | `d_c × h(d_h^C + d_h^V)` = 512 × 4096 | 2,097,152 |
| `wo` | `h·d_h^V × d` = 2048 × 2048 | 4,194,304 |
| `kv_norm` | `d_c` | 512 |
| **total** | | **13,763,072** |

Compare against MHA at the same `d`: `4d² = 16,777,216`. MLA is *cheaper in
parameters* here (0.82×) and — the actual point — **7.1× cheaper in KV cache**:
MHA would store `2·h·d_h = 4096` values per token per layer against MLA's 576
(topic 06 does this properly). At DeepSeek-V2's `h = 128` the same comparison
gives 57×, which is why the technique was invented at that scale.

Note `wkv_a` produces `d_c + d_h^R = 576` numbers per token. That 576 is the
entire per-layer KV cache for MLA. Hold onto it.

### 2.4 A MoE block

```
router:          d × E                      = 2048 × 8      =     16,384
shared expert:   3 · d · (E_s · d_ff^E)     = 3·2048·1408    =  8,650,752
routed experts:  E · 3 · d · d_ff^E         = 8 · 8,650,752  = 69,206,016
                                                       total = 77,873,152
```

**Active** parameters per token are what the FLOPs depend on. With `k = 1`:

```
active = router + shared + (k/E) · routed
       = 16,384 + 8,650,752 + 8,650,752  =  17,317,888     (22% of total)
```

This gap — total parameters vs active parameters — is the entire economic
argument for MoE, and the reason expert parallelism exists (topic 28). Memory
scales with **total**; compute scales with **active**.

### 2.5 The full count

| Component | Multiplicity | Params |
|-----------|--------------|--------|
| Embedding | 1 | 209,715,200 |
| Output projection | 1 | 209,715,200 |
| MLA | 27 | 371,602,944 |
| Dense FFN | 1 | 67,239,936 |
| MoE FFN | 26 | 2,024,701,952 |
| RMSNorms | 27·2 + 1 | 112,640 |
| **Total** | | **≈ 2.88 B** |
| **Active per token** | | **≈ 1.31 B** |

Sparsity ratio `2.88 / 1.31 ≈ 2.2×`. (Production DeepSeek-V3 pushes this to
`671B / 37B ≈ 18×` with `E=256, k=8`.)

## 3. Counting training FLOPs

### 3.1 The `6N` rule, derived

Take one linear layer `y = x W`, with `x ∈ R^{T×d_in}`, `W ∈ R^{d_in×d_out}`,
where `T = B·S` is the token count.

**Forward.** A matmul `[T,d_in] × [d_in,d_out]` performs `T·d_in·d_out`
multiply-accumulates. One MAC = 2 FLOPs:

```
F_fwd = 2 · T · d_in · d_out = 2 · T · (#params of W)
```

**Backward.** Autograd must produce *two* things:

```
∂L/∂x = (∂L/∂y) Wᵀ      [T,d_out] × [d_out,d_in]  →  2 · T · d_in · d_out
∂L/∂W = xᵀ (∂L/∂y)      [d_in,T] × [T,d_out]      →  2 · T · d_in · d_out
```

Two matmuls of exactly the same size as the forward one. Hence:

```
F_bwd = 2 · F_fwd
```

**Total.**

```
F_step = F_fwd + F_bwd = 6 · T · (#params)
```

Sum over all weight matrices in the model:

```
┌──────────────────────────────────┐
│   C ≈ 6 · N · D    FLOPs         │
└──────────────────────────────────┘
```

where `N` counts the *matmul* parameters and `D` the tokens seen. Three caveats
that the reference code handles explicitly:

- **The embedding is excluded.** It is a gather, not a matmul: zero FLOPs
  forward, and its backward is a scatter-add. The output projection *is*
  included — it is a real `d × V` matmul, and at `V = 102400` it is not small.
- **For MoE, use *active* parameters.** A token only touches `k` experts.
- **Norms, activations, residuals are ignored.** They are `O(T·d)`, smaller by a
  factor of `d` than the `O(T·d²)` matmuls. For `d = 2048` this is a 0.1% error.

### 3.2 The attention-score term

The `6N` rule covers parameter matmuls. Attention contains two matmuls with *no
parameters at all* — their operands are both activations — and these scale with
`S`, not with `N`.

Per head, per layer, forward:

```
scores = Q Kᵀ :  [S, d_qk] × [d_qk, S]  →  2 S² d_qk   FLOPs
out    = A V  :  [S, S]     × [S, d_v]  →  2 S² d_v    FLOPs
```

Divide by `S` to get per-token, multiply by 3 for forward+backward, and by
`h` heads and `L` layers:

```
┌────────────────────────────────────────────────┐
│  F_attn/token = 6 · L · h · S · (d_qk + d_v)   │
└────────────────────────────────────────────────┘
```

with `d_qk = d_h^C + d_h^R` and `d_v = d_h^V`. This is *exactly* the second
term in `model_args.py`:

```python
head_dims = self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim
num_flops_per_token = (
    6 * (nparams_dense - nparams_embedding + nparams_sparse_active)
    + 6 * n_layers * n_heads * head_dims * seq_len
)
```

`head_dims = 128 + 64 + 128 = 320 = d_qk + d_v`. Now you can read that line and
know where every symbol came from.

> **Causal masking.** With a causal mask only half the score matrix is needed,
> so a good kernel (FlashAttention) does `≈ S²/2` work. The formula above is the
> conventional *unmasked* count; frameworks keep it for comparability across
> papers. Halve the attention term if you want the true achieved FLOPs. Be
> consistent, and say which convention you used when you report MFU.

### 3.3 When does attention dominate?

```
F_attn        6 L h S (d_qk + d_v)        L h S (d_qk + d_v)
────────  =  ───────────────────────  =  ────────────────────
F_param            6 · N_body                   N_body
```

For a classic dense model where `N_body = 12 L d²` and `h·d_h = d`,
`d_qk + d_v = 2d/h`, this collapses to the famous:

```
F_attn / F_param  =  S / (6 d)
```

So attention is negligible while `S ≪ 6d`, and takes over beyond it. For our
config, evaluated exactly:

| `S` | `F_param`/token | `F_attn`/token | attention share |
|-----|-----------------|----------------|-----------------|
| 4,096 | 6.59 GFLOP | 3.40 GFLOP | 34% |
| 16,384 | 6.59 GFLOP | 13.59 GFLOP | **67%** |
| 65,536 | 6.59 GFLOP | 54.36 GFLOP | 89% |

At the config's own `max_seq_len = 16384`, **two thirds of the compute is
attention**, none of it parameter-bound. This is the quantitative reason context
parallelism (topic 23) exists as a separate technique: at long context the thing
you must parallelize is not the weights, it is the `S²` score matrix.

### 3.4 Activation checkpointing changes the constant

If you recompute activations in the backward pass instead of storing them, you
pay one extra forward:

```
no recompute:            2N (fwd) + 4N (bwd)            = 6N
full recompute:          2N (fwd) + 2N (recompute) + 4N = 8N
selective recompute:     between the two
```

Papers quoting `8N` (including the Megatron-LM scaling paper's throughput model)
are assuming full recomputation. When you compare MFU numbers between two
sources, check which constant each used — a 33% discrepancy hides here.
`torchfeather` implements this in `distributed/activation_checkpoint.py`
(topic 20).

### 3.5 MFU — the number you actually report

```
                achieved FLOPs per second        F_step / t_step
MFU  =  ───────────────────────────────────  =  ─────────────────────
        aggregate peak FLOPs of the cluster      W · peak_flops
```

Worked example: our 2.88B model, `S = 16384`, global batch of 64 sequences,
step time 1.9 s, on 32 H100s (bf16 peak `989e12`, dense, no sparsity):

```
tokens/step = 64 · 16384                        = 1,048,576
FLOPs/token = 6.59e9 + 13.59e9                  = 2.018e10
F_step      = 1.048576e6 · 2.018e10             = 2.116e16
MFU         = 2.116e16 / (1.9 · 32 · 989e12)    = 0.352  →  35%
```

35% is a respectable number for a MoE model at long context. Interpreting MFU
requires knowing your own convention (masked or not, `6N` or `8N`) — always
state it.

### 3.6 Where the compute budget should go: scaling laws

Two results tell you how to *choose* `N` and `D` rather than count them.

**Chinchilla** (Hoffmann et al., 2022). Minimizing loss subject to
`C = 6ND` gives `N ∝ C^0.5`, `D ∝ C^0.5`, i.e. scale both equally, with the
optimum near:

```
D ≈ 20 N
```

This replaced the earlier Kaplan et al. (2020) prescription, which
under-weighted data (`N ∝ C^0.73`) because it used a fixed LR schedule across
model sizes. Note that inference cost pushes production models well past
`20N` ("over-training") — Chinchilla optimizes *training* loss per training
FLOP only.

**Fine-grained MoE scaling** (Krajewski et al., 2024). Introduces *granularity*
`G` = how finely the FFN is split into experts (small `d_ff^E`, many `E`, larger
`k`). The finding: at fixed active parameters and fixed training budget, higher
granularity strictly improves loss, until the routing/all-to-all overhead
dominates. This is why modern MoEs moved from `E=8, k=2` (Mixtral) to
`E=256, k=8` (DeepSeek-V3) with a shared always-on expert. Topic 26 derives what
granularity does to the router; topic 29 shows what it does to the all-to-all.

## 4. Counting bytes

FLOPs are one budget; bytes are the other two.

### 4.1 Model state (per replica, before sharding)

| Item | dtype | bytes/param |
|------|-------|-------------|
| params | bf16 | 2 |
| grads | fp32 | 4 |
| Adam `m` | fp32 | 4 |
| Adam `v` | fp32 | 4 |
| master params | fp32 | 4 |
| | | **18** |

Our 2.88B model: `2.88e9 × 18 ≈ 51.8 GB`. On an 80 GB card that leaves 28 GB
for activations — which is why we shard (topics 20, 21).

Note the asymmetry MoE creates: **optimizer state scales with total (2.88B),
compute scales with active (1.31B)**. MoE buys compute efficiency by spending
memory. Expert parallelism is the tool that pays that bill.

### 4.2 Activations

Per layer, the tensors that must be kept for backward are `O(B·S·d)` each, plus
the attention working set. Roughly:

```
M_act ≈ L · B · S · d · c · bytes_per_elem
```

where `c` is a small constant (≈ 10–20 for a transformer block with SwiGLU,
depending on what the kernels fuse). For `L=27, B=1, S=16384, d=2048`, bf16,
`c=14`:

```
27 · 1 · 16384 · 2048 · 14 · 2 B ≈ 25.4 GB
```

Per sequence. This is why long-context training OOMs before it slows down, and
why activation checkpointing, FSDP and context parallelism are not optional at
scale.

### 4.3 KV cache (inference, but it drives the architecture)

Deferred to topic 06, where we derive it alongside arithmetic intensity — the
two together are the whole motivation for MLA.

## 5. Implementation

`get_nparams_and_flops` classifies parameters by name, which is the practical
way to separate dense from sparse:

```python
for name, p in model.named_parameters():
    if "embedding" in name:
        nparams_embedding += p.numel()
        nparams_dense += p.numel()
    elif "moe.shared_experts" in name:
        nparams_shared_experts += p.numel()
    elif "moe.router" in name:
        nparams_moe_router += p.numel()
    elif "moe.experts" in name:
        nparams_experts += p.numel()
    else:
        nparams_dense += p.numel()

nparams_sparse_active = (
    nparams_moe_router
    + nparams_shared_experts
    + nparams_experts * self.moe_args.top_k // self.moe_args.num_experts
)
```

Read the three design choices in that snippet:

1. Embedding is added to `dense` **and** tracked separately, so it can be
   subtracted from the FLOP count while remaining in the parameter count.
2. Shared experts and the router are *always active* — they enter
   `sparse_active` at full weight.
3. Routed experts are scaled by `k/E`. This is an expectation: it assumes
   perfect load balance. Under imbalance the *achieved* FLOPs are the same
   (each token still visits `k` experts) but the *wall-clock* is worse because
   one device does more work than the others. FLOP counting cannot see load
   imbalance — topic 26's load-balancing loss exists precisely because of this
   blind spot.

### A standalone counter you can run

```python
def count_params(d, L, n_dense, V, d_ff, d_ff_e, h,
                 d_c, d_nope, d_rope, d_v, E, E_s, k):
    d_qk = d_nope + d_rope
    mla = (d * h * d_qk            # wq (q_lora_rank = 0)
           + d * (d_c + d_rope)    # wkv_a
           + d_c * h * (d_nope + d_v)  # wkv_b
           + h * d_v * d           # wo
           + d_c)                  # kv_norm
    dense_ffn = 3 * d * d_ff
    moe = d * E + 3 * d * (E_s * d_ff_e) + E * 3 * d * d_ff_e
    moe_active = d * E + 3 * d * (E_s * d_ff_e) + k * 3 * d * d_ff_e
    embed = V * d
    total = (2 * embed + L * mla + n_dense * dense_ffn
             + (L - n_dense) * moe + L * 2 * d + d)
    active = (2 * embed + L * mla + n_dense * dense_ffn
              + (L - n_dense) * moe_active + L * 2 * d + d)
    return total, active


def flops_per_token(total, active, embed, L, h, d_qk, d_v, S, recompute=False):
    c = 8 if recompute else 6
    return c * (active - embed) + 6 * L * h * (d_qk + d_v) * S


total, active = count_params(
    d=2048, L=27, n_dense=1, V=102400, d_ff=10944, d_ff_e=1408, h=16,
    d_c=512, d_nope=128, d_rope=64, d_v=128, E=8, E_s=1, k=1)
print(f"total  {total/1e9:.2f}B")   # 2.88B
print(f"active {active/1e9:.2f}B")  # 1.31B
print(f"{flops_per_token(total, active, 102400*2048, 27, 16, 192, 128, 16384)/1e9:.2f} GFLOP/token")
```

## 6. Check yourself

1. Why is `d_ff = (8/3)d` the standard choice for SwiGLU rather than `4d`?
2. Derive `F_bwd = 2 · F_fwd` for `y = xW`. Which of the two backward matmuls
   can be skipped for the *first* layer of the network, and why does it not
   change the `6N` rule in practice?
3. Our config at `S = 16384` spends 67% of its FLOPs on attention scores. Which
   parallelism strategy reduces that per-device cost, and which ones do not
   touch it at all?
4. An MoE with `E=256, k=8` has 20× more total than active parameters. Which of
   the following scale with total, and which with active: step FLOPs, optimizer
   memory, all-reduce volume for data parallelism, all-to-all volume for expert
   parallelism?
5. You measure 45% MFU using `8N`, and a colleague measures 38% on the same run
   using `6N` and causal-halved attention. Who is faster?

## Next

→ [03 — RoPE from first principles](03-rope-from-first-principles.md)
