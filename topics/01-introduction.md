# 01 — Introduction and the Shape of the Problem

> Video: 00:00:00 — Introduction

## 1. What we are building

A working distributed training framework, from scratch, that can train a modern
dense or mixture-of-experts language model across many GPUs using **five**
composable forms of parallelism simultaneously:

```
                      ┌─────────────────────────────────────────┐
                      │        one training step                │
                      └─────────────────────────────────────────┘
   cut the layers    →  pipeline parallelism        (PP)
   cut the batch     →  data parallelism / FSDP     (DP / FSDP / HSDP)
   cut the matrices  →  tensor parallelism          (TP)
   cut the sequence  →  context parallelism         (CP)
   cut the experts   →  expert parallelism          (EP)
```

That list is exhaustive for the transformer. A training step consumes a tensor
of shape `[B, S]` and a stack of `L` layers whose weights are matrices. There
are exactly four axes to cut (batch, sequence, layer index, matrix dimension),
plus one more that only exists once the FFN is sparse (expert index). Everything
in this course is a consequence of that observation.

## 2. Why one GPU is not enough — the three walls

Three separate limits force distribution. They are not the same limit and they
are not fixed by the same technique.

### Wall 1: memory (the model does not fit)

For a model with `N` parameters trained in mixed precision with Adam, the
*static* memory per replica is:

| Item | Bytes per parameter | Why |
|------|---------------------|-----|
| Parameters (bf16) | 2 | the forward pass |
| Gradients (bf16 or fp32) | 2–4 | accumulated in backward |
| Adam first moment `m` (fp32) | 4 | momentum |
| Adam second moment `v` (fp32) | 4 | variance |
| Master weights (fp32) | 4 | optimizer updates in fp32 |
| **Total** | **16–18** | |

A 70B model therefore needs about **1.1–1.3 TB** of state before a single
activation is stored. An H100 has 80 GB. The gap is a factor of ~15.

On top of that, activations for the backward pass scale as `O(B·S·d·L)`, which
for long context dominates everything else.

> **Fixes:** FSDP/ZeRO shards the 16 bytes/param across `P_sh` ranks. Tensor
> parallelism shards it across `P_tp`. Pipeline parallelism shards it across
> `P_pp` by giving each rank only `L/P_pp` layers. Activation checkpointing and
> context parallelism attack the activation term.

### Wall 2: time (the model does not train fast enough)

Total training compute is approximately

```
C ≈ 6 · N · D        FLOPs
```

(derived in topic 02). For a Chinchilla-optimal 70B model, `D ≈ 20N = 1.4T`
tokens:

```
C ≈ 6 × 70e9 × 1.4e12 ≈ 5.9e23 FLOPs
```

One H100 sustains roughly `4e14` bf16 FLOP/s at good utilization. So:

```
5.9e23 / 4e14 ≈ 1.5e9 seconds ≈ 47 years
```

on a single GPU. To finish in one month you need about **560 GPUs** running at
high efficiency. Nothing about memory is involved in this argument — even if the
model fit on one card, you would still need the cluster.

> **Fix:** data parallelism is the only technique that adds throughput by adding
> devices. Everything else exists to make the model *fit* so that data
> parallelism can be applied on top.

### Wall 3: bandwidth (the devices cannot talk fast enough)

Every cut you make must be repaired by communication. The repair cost is set by
the interconnect, and interconnects are *hierarchical*:

| Link | Bandwidth (order of magnitude) |
|------|-------------------------------|
| HBM (on-device memory) | 3000 GB/s |
| NVLink (intra-node, 8 GPUs) | 400–900 GB/s |
| InfiniBand / RoCE (inter-node) | 25–50 GB/s |

The ratio between intra-node and inter-node is ~20×. This single fact
determines the entire design of a parallelism strategy, and gives the rule that
recurs throughout the course:

> **The bandwidth rule.** Put the chattiest parallelism on the fastest link.
> Tensor parallelism communicates *twice per layer* → keep `P_tp` inside a node.
> Pipeline parallelism communicates *twice per stage boundary* and the messages
> are small → it survives slow inter-node links. Data parallelism communicates
> *once per step* and can be overlapped → it also survives slow links.

## 3. The mental model: follow the tensor

The mistake most people make when learning this material is studying each
parallelism technique in isolation. Real frameworks compose all of them, and the
only way to keep it straight is to track three things for every tensor:

1. **Its logical shape** — what it would be on one device.
2. **Its layout** — which mesh dimensions it is sharded, replicated or partial
   along.
3. **Its gradient's layout** — which is *not* always the same as the forward
   layout, and is the source of nearly every bug.

This is exactly what PyTorch's `DTensor` abstraction encodes, and why we build
up to it rather than starting from it. The central duality, proved in topic 11
and used from topic 19 onwards:

```
   forward placement          backward placement of the gradient
   ─────────────────          ──────────────────────────────────
   Replicate()          <-->  Partial(sum)
   Partial(sum)         <-->  Replicate()
   Shard(i)             <-->  Shard(i)
```

and therefore, for the collectives:

```
   forward op          backward op
   ──────────          ───────────
   all-gather     <-->  reduce-scatter
   reduce-scatter <-->  all-gather
   all-reduce     <-->  all-reduce     (see topic 11 for the exact statement)
   all-to-all     <-->  all-to-all     (with split/concat dims swapped)
```

Memorize the table now; the rest of the course earns it.

## 4. Roadmap, and why it is in this order

```
Part I   (02–10)  The model.
                  Count parameters, FLOPs and KV-cache bytes exactly.
                  Build RoPE, YaRN, and MLA — because you cannot reason
                  about context parallelism without knowing precisely which
                  tensors attention needs, and you cannot reason about
                  expert parallelism without an MoE FFN.

Part II  (11–13)  The mathematics.
                  Autograd as a linear operator; what it means to cut a
                  computation graph across devices; the training loop that
                  every later chapter modifies.

Part III (14–25)  The four dense parallelisms, each derived then coded,
                  then composed on a device mesh.

Part IV  (26–30)  Sparsity: MoE, and the expert / expert-tensor parallelism
                  that only sparse models need.
```

Part I looks like a detour. It is not. Three concrete examples of why the
architecture and the parallelism are inseparable:

- **MLA compresses the KV cache into a latent of size `d_c`.** That changes what
  context parallelism has to ship around the ring (topic 23).
- **RoPE is position-dependent.** Shard the sequence across devices and each
  device must know its *global* position offset, or the embedding is silently
  wrong (topic 23). This is a real and common bug.
- **MoE routes tokens to experts.** If experts live on different devices, the
  routing *is* an all-to-all (topic 29), and the load balance of the router
  becomes a systems problem, not just a modelling one.

## 5. What "from first principles" means here

For every component we will:

1. State the problem it solves in one sentence.
2. Write down the requirement as an equation.
3. Solve the equation (or show the standard solution satisfies it).
4. Translate the solution to code, matching the reference implementation.
5. Count its cost in FLOPs, bytes moved, and memory held.

Step 5 is the one usually skipped, and it is the one that makes the difference
between knowing the names of techniques and being able to design new ones.

## 6. Environment and setup

The reference framework is `torchfeather`, cloned at
`../Distributed-Training-Framework/`. Its layout maps onto this course:

```
torchfeather/
├── model/
│   ├── rope.py              → topics 03, 04
│   ├── attention.py         → topics 06, 07, 09, 10
│   ├── model.py             → topic 05
│   ├── model_args.py        → topic 02
│   ├── moe/                 → topics 26–30
│   └── parallelize.py       → topics 20, 22, 25  (the composition point)
├── distributed/
│   ├── parallel_dims.py     → topic 18  (device mesh construction)
│   ├── pipeline_parallel.py → topics 15, 17
│   ├── model_parallel.py    → topics 20, 22
│   └── expert_parallel.py   → topics 28–30
├── components/              → topic 24  (optimizer, scheduler, metrics, ckpt)
└── train.py                 → topics 13, 25  (the training loop)
```

Minimal standalone examples worth running early:

```
minimal_examples/block_matrix_multiply.py   → topic 08
minimal_examples/pp_gpipe.py                → topic 15
minimal_examples/tp_swiglu.py               → topic 21
minimal_examples/tp_attention.py            → topic 21
minimal_examples/ring_attention.py          → topic 23
minimal_examples/ring_attention_intro.ipynb → topic 23
```

You do not need a GPU cluster to follow along. Almost every collective can be
exercised with the `gloo` backend on CPU:

```bash
torchrun --nproc_per_node=4 my_script.py
```

If `torchrun` is unavailable or its rendezvous fails (`RendezvousConnectionError:
The connection to the C10d store has failed` — common in sandboxes and
network-restricted environments, since the default c10d store opens a localhost
TCP socket), use a **file-based store** instead, which needs no networking:

```python
import os, tempfile, torch.distributed as dist, torch.multiprocessing as mp

def worker(rank, world, initfile):
    dist.init_process_group("gloo", init_method=f"file:///{initfile}",
                            rank=rank, world_size=world)
    ...
    dist.destroy_process_group()

if __name__ == "__main__":
    f = os.path.join(tempfile.gettempdir(), "pg_store")
    if os.path.exists(f):
        os.remove(f)
    mp.spawn(worker, args=(4, f.replace("\\", "/")), nprocs=4, join=True)
```

Every multi-process script in this course uses that harness. Remove the store
file between runs — a stale one causes a hang, not an error.

Single-GPU and CPU-only machines can run every derivation and most of the code
in this course; only the throughput numbers need real hardware. Two exceptions
worth knowing up front: the MoE forward pass requires CUDA (`torch.histc` on
int64 and `torch._grouped_mm`), and the fast attention backends the reference
wrapper requests are GPU-only, so CPU experiments need the math backend.

## 7. Check yourself

1. A 13B model fits in 80 GB for inference in bf16 but OOMs immediately in
   training. Give the arithmetic that explains the difference.
2. Your 70B training run has `P_tp = 16` on a cluster of 8-GPU nodes. Why is
   this configuration almost certainly slower than `P_tp = 8`?
3. Data parallelism is the only parallelism that increases throughput per device
   added. Explain why pipeline parallelism does not, and why we use it anyway.
4. Without looking: what is the backward-pass collective for a forward
   all-gather? For a forward all-to-all?

## Next

→ [02 — Model architecture, parameters and training FLOPs](02-model-architecture-parameters-flops.md)
