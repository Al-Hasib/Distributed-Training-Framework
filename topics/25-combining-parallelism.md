# 25 — Combining Parallelism in the Training Loop

> Video: 16:00:33 — Combining Parallelism in the Training Loop
> Code: `torchfeather/train.py`, `torchfeather/model/parallelize.py`

## Why this exists

Topic 01 promised this chapter: *"we won't study each parallelism technique in
isolation."* Every individual technique is now derived and coded. This topic does
the thing that actually distinguishes a framework from a demo — **it runs all of
them at once and tracks what happens.**

The claim to test: composing five parallelisms is not five independent
transformations. There are roughly **fourteen** places where two of them
interact and the correct behaviour is not what either would do alone. Every one
of those has already appeared in this course; here they are collected, which is
what makes them usable.

## 1. The nesting order, complete

```
PP          cuts the module tree           (topic 17 §3)
 └─ FSDP    intercepts parameter access    (topic 20 §11)
     └─ torch.compile
         └─ activation checkpointing
             └─ TP / EP   rewrite parameter layout   (topics 22, 28)
                 └─ CP    wraps the attention module (topic 23)
                     └─ the model
```

Expressed in code as PP outside, then `parallelize_fn` per part:

```python
stages, model_parts = pipeline_module_split(model, ...)      # PP first
for i, m in enumerate(model_parts):
    m = parallelize_fn(m, parallel_dims, job_config)          # then everything else
    model_parts[i] = m
    stages[i].submod = m
```

```python
def parallelize_deepseekv3(model, parallel_dims, job_config):
    assert job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    if parallel_dims.tp_enabled:            apply_non_moe_tp(...)   # 1
    if tp_enabled or ep_enabled:            apply_moe_ep_tp(...)    # 2
    if ac_mode != "none":                   apply_ac(...)           # 3
    if model_compile_enabled:               apply_compile(...)      # 4
    if fsdp_enabled or ep_enabled:          apply_fsdp(...)         # 5
    elif dp_replicate_enabled:              apply_ddp(...)
```

Each adjacency is forced:

| Boundary | Why this order |
|----------|----------------|
| PP outside FSDP | PP cuts modules; FSDP wraps whatever module remains |
| FSDP outside compile | FSDP must intercept parameter access on the final callable |
| compile outside AC | compile must see the recomputation wrapper |
| AC outside TP | recomputation must replay the *parallelized* forward |
| TP innermost | it rewrites the parameters themselves |

CP is the exception: it is applied at *runtime* via a context manager around the
forward/backward, not at construction time, because it shards *activations and
buffers* rather than parameters.

## 2. Follow one tensor through everything

Configuration: `P_pp=2, P_dp=2, P_cp=2, P_tp=4` on 32 GPUs, `S=8192`, MoE with
`P_ep=4`. One token's journey.

```
[dataloader]  the batch is sharded over the "batch" mesh (dp_replicate × dp_shard)
              → replicated across tp, cp, pp                          (topic 16 §3)
              inputs [B, 8192]

[CP context]  inputs, labels and every model part's freqs_cis are sharded
              on the sequence dim with zigzag load balancing          (topic 23 §5)
              → each rank holds 4096 positions, in two non-contiguous chunks
              local inputs [B, 4096]

[PP stage 0]  holds tok_embeddings + layers 0-13
              (stage 1 receives a float activation instead of token ids)

  [FSDP]      all-gather this block's parameter shards from 4 ranks
              (fsdp = dp_shard × cp = 2 × 2)                          (topic 18 §3.2)

  [embedding] RowwiseParallel: vocab-sharded over tp
              Partial(sum) → reduce-scatter → Shard(1)                (topic 22 §2.1)
              h  Shard(seq) over tp   [B, 1024, d]

  [norm]      SequenceParallel — local                                (topic 22 §2.2)

  [attention] PrepareModuleInput: Shard(1) → Replicate = ALL-GATHER
              h  [B, 4096, d] replicated over tp
     wq, wkv_b  ColwiseParallel  → per-head shards                    (topic 22 §3.2)
     wkv_a, kv_norm  NoParallel  → replicated (head-independent)      (topic 21 §3.1)
     inner_attention:
         DTensor detected → to_local(), saving the tp spec            (topic 06 §7.3)
         CP hook: ring-rotate K,V across 2 cp ranks, online softmax   (topic 23 §3)
         DTensor.from_local() with the saved tp spec
     wo         RowwiseParallel  Partial → Shard(1) = REDUCE-SCATTER  (topic 22 §3.5)
              h  Shard(seq) over tp

  [MoE]       router → top-k → all-to-all dispatch over ep            (topic 29)
              grouped matmul on local experts
              all-to-all combine (the adjoint)                        (topic 11 §2.5)

  [FSDP]      reshard_after_forward = FALSE, because PP is on:
              hold the parameters for all M micro-batches             (topic 20 §3)

[PP boundary] send h.detach() to stage 1                              (topic 17 §1)

[PP stage 1]  layers 14-26 + norm + output
  [output]    ColwiseParallel, Shard(-1) on the vocabulary
              → loss_parallel: never gather [B,S,102400]              (topic 21 §6)
  [loss]      cross-entropy, reduction="sum"                          (topic 13 §4)

[backward]    the PP schedule runs it; each collective is replaced by
              its adjoint, automatically                              (topic 11 §2)

[normalize]   PP already ran backward, so divide gradients by
              global_valid_tokens after the fact                      (topic 13 §4.1)

[clip]        global norm across DTensor shards + pp_mesh + ep         (topic 13 §5)

[step]        parameters grouped by mesh (dense vs expert)             (topic 24 §1.2)
              expert_bias updated in a post-step hook                  (topic 24 §1.3)
```

Every annotation is a place where the naive composition would be wrong.

## 3. The interaction table

The reference for when you are debugging a composed configuration.

| # | Pair | What changes | Where |
|---|------|--------------|-------|
| 1 | **PP × FSDP** | `reshard_after_forward = False`, else `M×` redundant all-gathers | t20 §3 |
| 2 | **PP × CP** | each model part has its own `freqs_cis` to shard | t17 §4.4, t23 §5.1 |
| 3 | **PP × loss** | only the last stage has a loss; others return `-1.0` | t13 §9 |
| 4 | **PP × normalization** | schedule owns backward → divide grads post-hoc; `scale_grads=False` | t13 §4.1, t15 §7 |
| 5 | **PP × clip** | global norm needs an explicit all-reduce over `pp_mesh` | t13 §5 |
| 6 | **PP × RNG** | *distinct* seeds across `pp` (different layers ⇒ independent masks) | t13 §7 |
| 7 | **TP × CP** | `to_local()` round-trip, or the ring uses the wrong process group | t06 §7.3, t23 §7 |
| 8 | **TP × dataloader** | shard by the `batch` mesh, never by `global_rank` | t16 §3.2 |
| 9 | **TP × CP × data** | `S % (tp · 2 · cp) == 0` | t18 §5 |
| 10 | **CP × FSDP** | `fsdp = dp_shard × cp` — a free extra `P_cp` on parameter memory | t18 §3.2 |
| 11 | **CP × loss** | divide `local_valid_tokens` by `cp` (all CP ranks load the full sequence) | t13 §3 |
| 12 | **EP × FSDP** | experts on `efsdp`; explicit prefetch; `Shard(1)` when experts run out | t20 §7–8 |
| 13 | **EP × optimizer** | parameter groups by mesh, or fused DTensor dispatch fails | t24 §1.2 |
| 14 | **EP × clip** | the `ep` dimension must be counted exactly once | t13 §5 |

And the invariant that spans four files:

```
15. NORMALIZATION.  Exactly one division, by global_valid_tokens.
    loss.py             reduction="sum"                  — do not average
    pipeline_parallel.py scale_grads=False               — do not divide by M
    model_parallel.py   set_gradient_divide_factor(1.0)  — do not divide by P
    train.py            loss_sum / global_valid_tokens   — THE division
```

Three components explicitly switched off so one can be right. If you write a
framework, this is the pattern to copy: when several layers *could* perform a
normalization, name the one that *does* and disable the others with a comment
saying why.

## 4. What each parallelism costs, side by side

Per micro-batch, `[B, S, d]` activations, `V_act = B·S·d·2` bytes:

| | memory saved | communication per step | link | overlappable |
|---|---|---|---|---|
| **DP replicate** | activations ÷ `P` | 1 all-reduce of `∂L/∂θ` | slow ok | yes |
| **FSDP** | 18 B/param ÷ `P` | 2 AG + 1 RS per module (`1.5×` DDP) | medium | yes (prefetch) |
| **PP** | params ÷ `P_pp` | `2·M·(P_pp−1)` p2p messages of `V_act` | **slowest ok** | yes |
| **TP** | params **and** activations ÷ `P_tp` | `4·L` collectives, `2V_act(P−1)/P` each | **fastest only** | **barely** |
| **CP** | activations ÷ `P_cp` | `(P_cp−1)·L` ring rotations of K,V | medium/fast | yes (at long `S`) |
| **EP** | expert params ÷ `P_ep` | 2 all-to-all per MoE layer | medium/fast | partially |

Read the two extremes. **TP** is the only one that shards a single layer's
activations *and* is essentially unoverlappable — hence "fastest link only, and
only as much as you need". **PP** is the only one whose message size is
independent of the model — hence "put it on the slowest link".

## 5. Choosing a configuration

```
1. Compute the memory requirement:  18 bytes/param × N  +  activations
2. TP  = smallest degree that makes one layer fit, ≤ node size, ≤ n_heads.
         Always with sequence parallelism and loss parallelism.
3. CP  = only if S is large (≳ 32k). Also buys parameter memory via
         fsdp = dp_shard × cp.
4. EP  = for MoE: enough to fit the expert weights; must satisfy topic 18 §4.4.
5. PP  = smallest degree that closes the remaining gap. Keep it small (bubble),
         and ensure M = local_batch / micro_batch ≥ 4·P_pp.
6. FSDP (dp_shard) = as many of the remaining ranks as still overlap.
7. DP replicate = the rest (HSDP shape).

Then check, in this order:
   dp_replicate · dp_shard · cp · tp · pp == world_size
   S % (tp · 2 · cp) == 0
   n_heads % tp == 0
   M ≥ 4 · P_pp
   the EP borrowing constraints (topic 18 §4.4)
```

Two rules of thumb that follow from §4:

> **Prefer FSDP.** It is the cheapest per byte saved and it overlaps well. Reach
> for TP/PP/CP only when FSDP cannot fix the specific constraint you have.
>
> **Add parallelism from the inside out.** TP within a node, then CP, then FSDP
> across nodes, then PP across the slowest boundary. Which is exactly the mesh
> dimension order (topic 18 §2).

## 6. Debugging a composed configuration

Composed parallelism fails in ways that no single-technique intuition covers.
The method that works:

**Ablate, one dimension at a time.** Set every degree to 1, verify against a
single-GPU run, then enable one at a time. The order matters — enable the ones
with the fewest interactions first:

```
baseline: all degrees 1        → must match single-GPU exactly
+ dp_shard                     → gradients must match to fp noise
+ tp                           → gradients must match; check n_heads % tp
+ cp                           → gradients must match; check the RoPE positions
+ pp                           → gradients must match; check normalization
+ ep                           → check the router's token counts balance
```

At each step the test is the same, and it is the only test that works:

> **Compare gradients against the previous configuration, not the loss curve.**
> Every parallelism bug in this course — wrong data shard, local instead of
> global RoPE positions, double normalization, CP on the TP group — produces
> *plausible* losses. Gradients do not lie.

**Symptom → cause table**, drawn from the whole of Part III:

| Symptom | First suspects |
|---|---|
| hangs at step 1 | mismatched collective order (t19 §5.1); PP peer/shape mismatch (t17 §5); rank-dependent conditional |
| OOM only with PP | `reshard_after_forward=True` under PP, or `M` too large (GPipe memory, t14 §5) |
| loss is `-1` | logging a non-last PP stage's sentinel (t17 §5) |
| loss ~5% worse than baseline | double normalization (§3 item 15); or TP ranks getting different data (t16 §3.2) |
| loss much worse at long context | local instead of global RoPE positions under CP (t23 §5) |
| wrong numbers with TP+CP, no error | missing `to_local()` round-trip (t06 §7.3) |
| clipping never engages | `pp_mesh`/`ep` missing from the norm reduction (t13 §5) |
| throughput collapses inter-node | `P_tp` crossing the node boundary (t18 §2) |
| throughput degrades after each restart | `expert_bias` missing from the checkpoint (t24 §4.5) |
| slow, uneven step times | GC stragglers (t13 §8.1); CP without load balancing (t23 §4.2) |
| fused-optimizer dispatch error with EP | parameters not grouped by mesh (t24 §1.2) |

## 7. Part III complete

You can now derive, implement, and compose four dense parallelisms. The structure
that unifies them, stated once more:

```
Every parallelism is a CUT of the computation graph.
Every cut costs exactly one forward collective and one backward collective.
The backward one is the ADJOINT of the forward one — derivable, not memorized.
Which cut to use is set by WHICH RESOURCE is binding: parameters, activations,
a single layer, the sequence, or throughput.
Where to place it is set by the INTERCONNECT hierarchy.
```

Part IV adds the fifth cut, which exists only for sparse models — and which
introduces the one collective we have not yet needed in anger: the all-to-all.

## 8. Check yourself

1. State the full nesting order and justify each adjacency.
2. Why is CP applied as a runtime context manager rather than at construction?
3. Follow `[B, S]` inputs to the loss under `P_pp=2, P_cp=2, P_tp=4`, naming
   every collective in order.
4. From memory, give six entries of the interaction table with the reason.
5. Name the four places that could normalize gradients and the one that does.
   What is the symptom if two do?
6. Which parallelism is the only one whose message size is independent of model
   size? Which is the only one that shards a single layer's activations?
7. Why is TP essentially unoverlappable while FSDP is not?
8. Give the configuration procedure and all five constraint checks.
9. Why ablate one dimension at a time, and why compare gradients rather than
   losses?
10. Your run OOMs only when PP is enabled. Two candidate causes?
11. Loss is 5% worse than the single-GPU baseline, no errors. Two candidates, and
    how would you distinguish them?
12. Throughput degrades for thousands of steps after every restart. Cause?

## Next

→ [26 — Mixture of Experts from first principles](26-moe-from-first-principles.md)
