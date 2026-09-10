# Distributed Training from First Principles

A complete, self-contained course on building a distributed training framework in
PyTorch — derived from first principles, with the mathematics of every component
explained before it is coded.

Reference implementation: [`torchfeather`](https://github.com/hkproj/torchfeather)
(cloned locally at `../Distributed-Training-Framework/`).

**Prerequisites:** none for distributed training. Helpful background:
- Flash Attention derived and coded from first principles
- Coding a Transformer from scratch in PyTorch

---

## How to use this course

Each topic file is standalone but ordered. Every file follows the same shape:

1. **Why this exists** — the problem that forces the technique into existence
2. **First principles** — the derivation, with all algebra shown
3. **Implementation** — runnable PyTorch, annotated
4. **Where it lives in `torchfeather`** — real file/line pointers
5. **Check yourself** — questions that fail if you only skimmed

Work top to bottom. The parallelism chapters (14–30) assume the autograd
mathematics from topic 11; do not skip it.

---

## Part I — The Model (topics 01–10)

You cannot parallelize what you cannot count. Part I builds a modern
transformer and establishes exactly how many parameters, FLOPs and bytes each
piece costs, because every parallelism decision later is a trade between
compute, memory and communication.

| # | Topic | Video timestamp |
|---|-------|-----------------|
| 01 | [Introduction and the shape of the problem](01-introduction.md) | 00:00:00 |
| 02 | [Model architecture, parameters and training FLOPs](02-model-architecture-parameters-flops.md) | 00:16:50 |
| 03 | [RoPE from first principles](03-rope-from-first-principles.md) | 00:51:57 |
| 04 | [Implementing RoPE and YaRN](04-implementing-rope-and-yarn.md) | 01:11:25 |
| 05 | [Building the transformer and weight initialization](05-transformer-and-weight-init.md) | 01:50:16 |
| 06 | [Attention, the KV cache and arithmetic intensity](06-attention-kv-cache-arithmetic-intensity.md) | 02:19:12 |
| 07 | [Coding Multi-head Latent Attention (MLA)](07-coding-mla.md) | 03:02:56 |
| 08 | [Block matrix multiplication and MLA internals](08-block-matmul-and-mla-internals.md) | 03:15:35 |
| 09 | [Deriving MLA and decoupled RoPE](09-deriving-mla-and-decoupled-rope.md) | 03:34:58 |
| 10 | [MLA weight absorption](10-mla-weight-absorption.md) | 04:10:54 |

## Part II — The Mathematics of Distributed Training (topics 11–13)

The single most important part of the course. Every parallelism strategy is a
statement about *where the chain rule is allowed to be cut*, and what has to be
communicated to sew it back together.

| # | Topic | Video timestamp |
|---|-------|-----------------|
| 11 | [Autograd and the mathematics of distributed training](11-autograd-and-distributed-math.md) | 04:32:59 |
| 12 | [Distributed computation graphs and DDP](12-distributed-computation-graphs-ddp.md) | 05:04:19 |
| 13 | [Building the training loop](13-building-the-training-loop.md) | 05:10:47 |

## Part III — Parallelism (topics 14–25)

| # | Topic | Video timestamp |
|---|-------|-----------------|
| 14 | [Pipeline parallelism from first principles](14-pipeline-parallelism-first-principles.md) | 06:06:17 |
| 15 | [Pipeline schedules: GPipe, 1F1B, zero bubble](15-pipeline-schedules.md) | 06:26:52 |
| 16 | [Datasets, tokenization and data parallelism](16-datasets-tokenization-data-parallelism.md) | 07:10:29 |
| 17 | [Coding pipeline parallelism](17-coding-pipeline-parallelism.md) | 07:42:10 |
| 18 | [Device meshes and combining PP with DP](18-device-meshes-pp-dp.md) | 08:37:34 |
| 19 | [Distributed communication collectives](19-communication-collectives.md) | 09:48:52 |
| 20 | [Implementing device meshes, DDP and FSDP](20-implementing-meshes-ddp-fsdp.md) | 10:49:41 |
| 21 | [Tensor parallelism from first principles](21-tensor-parallelism-first-principles.md) | 12:13:08 |
| 22 | [Coding tensor parallelism](22-coding-tensor-parallelism.md) | 13:44:57 |
| 23 | [Context parallelism and ring attention](23-context-parallelism-ring-attention.md) | 14:47:43 |
| 24 | [Metrics, optimizers, schedulers, checkpointing](24-metrics-optimizers-schedulers-checkpointing.md) | 15:41:22 |
| 25 | [Combining parallelism in the training loop](25-combining-parallelism.md) | 16:00:33 |

## Part IV — Mixture of Experts and Expert Parallelism (topics 26–30)

| # | Topic | Video timestamp |
|---|-------|-----------------|
| 26 | [Mixture of Experts from first principles](26-moe-from-first-principles.md) | 16:49:58 |
| 27 | [Tensor parallelism for MoE](27-tensor-parallelism-for-moe.md) | 18:02:52 |
| 28 | [Expert parallelism](28-expert-parallelism.md) | 18:36:24 |
| 29 | [All-to-all token dispatch and combine](29-all-to-all-dispatch-combine.md) | 19:01:56 |
| 30 | [Expert tensor parallelism](30-expert-tensor-parallelism.md) | 19:27:38 |

## Appendices

- [Notation and symbols](appendix-notation.md) — one table, used everywhere
- [Papers and references](appendix-references.md) — every paper cited, with what to read it for
- [Glossary](appendix-glossary.md)

---

## The one-paragraph summary of the whole course

A transformer's training step is a composition of differentiable functions. On
one device, autograd walks that composition backwards and you get gradients. To
train a model too large or too slow for one device, you cut the composition into
pieces and put the pieces on different devices. There are only four things you
can cut: the **layers** (pipeline parallelism), the **batch** (data
parallelism), the **weight matrices** (tensor parallelism), and the **sequence**
(context parallelism) — plus a fifth for sparse models, the **experts** (expert
parallelism). Each cut severs the chain rule somewhere, and each severed point
must be repaired with exactly one collective communication operation in the
forward pass and exactly one in the backward pass. Learn which collective
repairs which cut, and you have understood distributed training.
