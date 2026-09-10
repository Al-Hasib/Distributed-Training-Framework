# Appendix — Notation

Used consistently in every topic. When a paper uses a different symbol, the
paper's symbol is given in the last column.

## Model dimensions

| Symbol | Meaning | Typical | Paper aliases |
|--------|---------|---------|---------------|
| `L` | number of transformer layers/blocks | 32 | `l`, `n_layers` |
| `d` | model / residual-stream dimension | 4096 | `h`, `d_model`, `n_embd` |
| `h` | number of attention heads | 32 | `n_h`, `a` |
| `d_h` | per-head dimension (`d/h` in MHA) | 128 | `d_head` |
| `d_ff` | FFN intermediate dimension | `(8/3)d` | `d_mlp`, `ffn_dim` |
| `V` | vocabulary size | 128256 | `v` |
| `S` | sequence length | 4096 | `s`, `T`, `n_ctx` |
| `B` | micro-batch size (per forward pass) | 1–8 | `b` |
| `N` | total parameter count | — | — |
| `D` | total training tokens | — | — |

## MLA-specific

| Symbol | Meaning | DeepSeek-V2 value |
|--------|---------|-------------------|
| `d_c` | KV compression (latent) dimension | 512 |
| `d_c'` | query compression dimension | 1536 |
| `d_h^R` | decoupled-RoPE head dimension | 64 |
| `d_h^C` | content (NoPE) head dimension | 128 |

## MoE-specific

| Symbol | Meaning |
|--------|---------|
| `E` | number of routed experts |
| `k` | experts activated per token (top-k) |
| `E_s` | number of shared (always-on) experts |
| `C` | expert capacity (tokens an expert accepts) |
| `f` | capacity factor, `C = f · k · T / E` |
| `T` | tokens in the local batch, `T = B · S` |

## Parallelism degrees

| Symbol | Meaning | Mesh dim name |
|--------|---------|---------------|
| `P_dp` | data-parallel degree (replicate) | `dp_replicate` |
| `P_sh` | data-parallel shard degree (FSDP) | `dp_shard` |
| `P_pp` | pipeline-parallel degree | `pp` |
| `P_tp` | tensor-parallel degree | `tp` |
| `P_cp` | context-parallel degree | `cp` |
| `P_ep` | expert-parallel degree | `ep` |
| `W` | world size, `W = P_pp · P_dp · P_sh · P_cp · P_tp` |

## Distributed primitives

| Symbol | Meaning |
|--------|---------|
| `r` | rank (global), `0 ≤ r < W` |
| `M` | number of micro-batches in a pipeline step |
| `AG` | all-gather |
| `RS` | reduce-scatter |
| `AR` | all-reduce (= `RS` then `AG`) |
| `A2A` | all-to-all |

## DTensor placements

| Placement | Meaning |
|-----------|---------|
| `Shard(i)` | tensor is split along dim `i` across the mesh dim |
| `Replicate()` | every rank holds the full tensor |
| `Partial(op)` | every rank holds a partial value; the true value is `op`-reduced across ranks (usually `Partial(sum)`) |

## Conventions

- Row-vector convention: activations are `x ∈ R^{B×S×d}`, and a linear layer is
  `y = x W` with `W ∈ R^{d_in×d_out}`. PyTorch's `nn.Linear` stores `W^T`
  (shape `[d_out, d_in]`); this matters for tensor parallelism and is called out
  explicitly wherever it does.
- `⟨a, b⟩` is the dot product. `⊙` is elementwise product. `∥` is concatenation.
- A **MAC** (multiply-accumulate) counts as 2 FLOPs.
- All logarithms in scaling-law contexts are natural logs unless subscripted.
