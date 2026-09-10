# Appendix — Papers and References

Every paper cited in the course, with **what to read it for** and **which topic
uses it**. Read the middle column before opening the paper — most of these are
worth reading for one specific section.

## Architecture

| Paper | Read it for | Topic |
|-------|-------------|-------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (1706.03762) | The baseline. Note it is *post*-norm and uses additive sinusoidal position encodings — both of which we reject and explain why. | 03, 05 |
| [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) (2104.09864) | §3.2 for the 2-D derivation, §3.4.3 for the long-term-decay argument. Skip the experiments. | 03 |
| [YaRN](https://arxiv.org/abs/2309.00071) (2309.00071) | §3.2 (NTK-by-parts) and §3.4 (attention temperature). The `α = 1, β = 32` defaults come from here. | 04 |
| [Fast Transformer Decoding (MQA)](https://arxiv.org/abs/1911.02150) (1911.02150) | The original arithmetic-intensity argument for sharing key heads. Short and worth reading in full. | 06, 09 |
| [GQA](https://arxiv.org/abs/2305.13245) (2305.13245) | The interpolation between MHA and MQA, and the uptraining recipe. | 06, 09 |
| [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (2405.04434) | §2.1 — MLA, including the decoupled-RoPE derivation. The single most important paper for Part I. | 06–10 |
| [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (2412.19437) | The full architecture: MLA + fine-grained MoE + shared experts + auxiliary-loss-free balancing, at 671B/37B. | 02, 26, 30 |
| [FlashAttention](https://arxiv.org/abs/2205.14314) | The online-softmax recurrence and the IO-complexity argument. Prerequisite video covers it. | 06, 23 |

## Mixture of Experts

| Paper | Read it for | Topic |
|-------|-------------|-------|
| [GShard](https://arxiv.org/abs/2006.16668) (2006.16668) | The original top-2 token-choice router, expert capacity, and the auxiliary balancing loss. The design this course's dropless implementation departs from. | 26 |
| [Scaling Laws for Fine-Grained MoE](https://arxiv.org/abs/2402.07871) (2402.07871) | Granularity `G` as a scaling variable. Why modern MoEs use many small experts rather than few large ones. | 26 |
| [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664) (2408.15664) | The bias-based balancing rule: bias for *selection*, original score for *gating*. The two-line crux of topic 26 §4.2. | 26, 24 |

## Parallelism and systems

| Paper | Read it for | Topic |
|-------|-------------|-------|
| [GPipe](https://arxiv.org/abs/1811.06965) (1811.06965) | Micro-batching and the bubble formula `(P−1)/(M+P−1)`. Also the `O(M)` memory problem that 1F1B fixes. | 14, 15 |
| [Megatron-LM: Efficient Large-Scale Training on GPU Clusters](https://arxiv.org/abs/2104.04473) (2104.04473) | The definitive treatment of composing TP + PP + DP. Interleaved 1F1B, and the throughput model (note it assumes activation recomputation, hence `8N` not `6N`). | 15, 18, 21 |
| Megatron-LM (tensor parallelism, 1909.08053) | The original column/row-parallel FFN and attention decomposition. | 21, 22 |
| Reducing Activation Recomputation (2205.05198) | Sequence parallelism: the free `P×` on activation memory in the norm regions. | 21 |
| [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241) (2401.10241) | Splitting the backward pass into `B` (input gradient) and `W` (weight gradient), and the ZBV V-shaped placement. | 15 |
| ZeRO (1910.02054) | The stage-1/2/3 memory decomposition that FSDP implements. | 20 |
| Ring Attention (2310.01889) | Distributing the online-softmax block loop. Note the load-imbalance problem under causal masking, which zigzag sharding solves. | 23 |

## Scaling laws

| Paper | Read it for | Topic |
|-------|-------------|-------|
| [Scaling Laws for Neural LMs](https://arxiv.org/abs/2001.08361) (2001.08361) | The first power-law fits. Its `N ∝ C^0.73` prescription is now known to under-weight data (fixed LR schedule across model sizes). | 02 |
| [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) (2203.15556) | `D ≈ 20N`. Read §3 for the three estimation methods. Remember it optimizes *training* loss per training FLOP only — inference cost pushes production models well past `20N`. | 02 |

## Code

| Repository | What it is |
|------------|------------|
| [torchfeather](https://github.com/hkproj/torchfeather) | The reference implementation for this course. Cloned locally at `../Distributed-Training-Framework/`. |
| [torchtitan](https://github.com/pytorch/torchtitan) | PyTorch's reference distributed-training codebase. `torchfeather` follows its structure closely; several comments in the source cite it. Read it for production-grade versions of everything here. |
| PyTorch `torch.distributed` | `tensor/` (DTensor), `fsdp/` (`fully_shard`), `pipelining/` (`PipelineStage`, schedules), `tensor/parallel/` (`ParallelStyle`), `tensor/experimental/_attention.py` (context parallelism / ring attention). |

## Prerequisite videos

- **Flash Attention derived and coded from first principles** — the online-softmax
  derivation that topic 23 distributes.
- **Coding a Transformer from scratch on PyTorch** — the baseline architecture
  topic 05 modifies.

## Suggested reading order, if you are starting from these papers

```
1. MQA (1911.02150)              — short; establishes arithmetic intensity
2. DeepSeek-V2 §2.1              — MLA and decoupled RoPE
3. GPipe                          — the bubble
4. Megatron-LM 2104.04473        — composition; read §2-4
5. ZeRO                           — the memory decomposition
6. Zero Bubble (2401.10241)      — the B/W split
7. Ring Attention                 — then re-read FlashAttention
8. Aux-loss-free (2408.15664)    — short; the balancing rule
9. DeepSeek-V3                    — everything at once, at scale
```

Papers 1, 6 and 8 are each readable in under an hour and each contain exactly one
idea. Start there.
