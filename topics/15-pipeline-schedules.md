# 15 — Pipeline Schedules: GPipe, 1F1B and Zero Bubble

> Video: 06:26:52 — Pipeline Schedules: GPipe, 1F1B and Zero Bubble
> Code: `torchfeather/distributed/pipeline_parallel.py`, `minimal_examples/pp_gpipe.py`
> Papers: GPipe (1811.06965), Megatron-LM (2104.04473), Zero Bubble (2401.10241)

## Why this exists

Topic 14 left us with a contradiction:

```
raise M  →  bubble ↓ ,  activation memory ↑
lower M  →  bubble ↑ ,  activation memory ↓
```

A **schedule** is a choice of *when* each rank does each piece of work. Both
quantities above are properties of the schedule, not of pipeline parallelism
itself — so the contradiction is not fundamental, and this topic dissolves it
in three steps:

1. **1F1B** — same bubble as GPipe, memory `O(P)` instead of `O(M)`. Free.
2. **Interleaved 1F1B** — bubble divided by `V`, at the cost of `V×` more
   messages.
3. **Zero bubble** — split the backward pass into its two independent halves and
   use one of them as filler. Bubble → ~0.

Step 3 is a direct application of topic 11 §1.3, and it is one of the most
elegant results in the field.

## 1. A schedule is a list of actions per rank

Strip away the details and a pipeline schedule is a table: for each rank, an
ordered list of `(action, micro-batch)` pairs. `minimal_examples/pp_gpipe.py`
makes this literal:

```python
class ActionKind(Enum):
    ...            # FORWARD, BACKWARD, SEND, RECV, ...

@dataclass
class Action:
    ...

def build_stage_zero_schedule(num_microbatches): ...
def build_stage_one_schedule(num_microbatches): ...
```

Actions available at each slot:

```
F(m)   forward micro-batch m on this stage
B(m)   backward micro-batch m: compute ∂L/∂x  (needed by the previous stage)
W(m)   backward micro-batch m: compute ∂L/∂W  (needed only by the optimizer)
S/R    send / recv an activation or a gradient
```

The `B`/`W` distinction is the one that unlocks zero bubble. Every schedule
below is a different ordering of the same action multiset — the total work is
identical, only the idle time differs.

## 2. GPipe: all forwards, then all backwards

```
M = 4, P = 4         (F = forward, B = backward, · = idle)

rank 0   F1 F2 F3 F4 ·  ·  ·  B4 B3 B2 B1
rank 1   ·  F1 F2 F3 F4 ·  B4 B3 B2 B1 ·
rank 2   ·  ·  F1 F2 F3 F4 B3 B2 B1 ·  ·
rank 3   ·  ·  ·  F1 F2 F3 B2 B1 ·  ·  ·
```

```
bubble  =  (P − 1) / (M + P − 1)
memory  =  O(M) micro-batches of activations on stage 0
```

Simple, and the memory term makes it unusable at the `M` values the bubble
formula wants. Its value is as the baseline.

## 3. 1F1B: the same bubble, `O(P)` memory

**The observation.** Stage 0 does not need to run all `M` forwards before its
first backward. It only needs to stay far enough ahead that the pipe stays full
— and "far enough" is `P − 1` micro-batches, not `M`.

So: run a **warmup** of `P − 1 − i` forwards on stage `i`, then alternate
**one forward, one backward** forever, then drain.

```
M = 8, P = 4

rank 0   F1 F2 F3 F4 B1 F5 B2 F6 B3 F7 B4 F8 B5 B6 B7 B8
rank 1   ·  F1 F2 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 F8 B6 B7 B8
rank 2   ·  ·  F1 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 F8 B7 B8
rank 3   ·  ·  ·  F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 F7 B7 F8 B8
         └warmup┘ └────────── steady state: 1F 1B ──────────┘└drain┘
```

**Memory.** On stage `i`, the number of micro-batches whose activations are
alive is at most `P − i`:

```
stage 0: P live      stage 1: P−1      …      stage P−1: 1
```

Independent of `M`. That is the whole win, and it is why 1F1B is the default
single-stage schedule everywhere.

**Bubble — unchanged.** This surprises people, so it is worth arguing. The
critical path is set by dependencies, not by ordering:

- micro-batch 1's forward must traverse `P` stages: `(P−1)·t_f` before the last
  stage can start.
- micro-batch `M`'s backward must traverse `P` stages coming back:
  `(P−1)·t_b` after the last stage finishes it.

Rearranging *when* the middle work happens cannot shorten that path. So:

```
bubble(1F1B) = bubble(GPipe) = (P − 1)/(M + P − 1)
memory(1F1B) = O(P)  ≪  memory(GPipe) = O(M)
```

**1F1B strictly dominates GPipe.** There is no reason to use GPipe in production;
it survives as the pedagogical starting point (and as
`minimal_examples/pp_gpipe.py`, which is easier to read).

## 4. Interleaved 1F1B: divide the bubble by `V`

**The observation.** The bubble is `(P−1)` *stage-times* of idleness. Make each
stage-time smaller and the absolute bubble shrinks.

So cut the model into `V·P` **virtual stages** instead of `P`, and give each rank
`V` of them, spread out:

```
V = 2, P = 4, so 8 virtual stages

rank 0:  stages 0, 4
rank 1:  stages 1, 5
rank 2:  stages 2, 6
rank 3:  stages 3, 7
```

which is exactly the "loop" placement in the reference code:

```python
# Rank i owns stages i, i + pp_degree, i + 2*pp_degree, ...
# With 8 stages and 4 ranks: rank 0 owns (0, 4), rank 1 owns (1, 5), ...
stage_indices = tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
```

A micro-batch now travels `0 → 1 → 2 → 3 → 0 → 1 → 2 → 3` — around the ranks
twice.

**The bubble.** Each virtual stage does `t_f/V` of forward work. The fill phase
still ends when all `P` ranks are busy, which takes `P − 1` *chunk*-times, not
`P − 1` stage-times:

```
T_fwd = (P − 1)·(t_f/V)  +  M·t_f
```

so

```
┌──────────────────────────────────────────────┐
│                      P − 1                   │
│   bubble  =  ─────────────────────           │
│                 V·M + P − 1                  │
└──────────────────────────────────────────────┘
```

The bubble is reduced by roughly `V×`. For `P = 8, M = 16`:

```
V = 1:  (8−1)/(16+7)     = 30%
V = 2:  (8−1)/(32+7)     = 18%
V = 4:  (8−1)/(64+7)     = 10%
```

**The costs.** Two, and both are real:

- **`V×` more messages.** A micro-batch crosses `V·P − 1` boundaries instead of
  `P − 1`. The *volume per boundary* is unchanged, but the message *count*
  multiplies, so latency and launch overhead multiply. On a slow link this can
  eat the gain outright — measure, do not assume.
- **More in-flight activations.** The warmup depth per rank grows with `V`
  (each rank must keep the earlier chunks of a micro-batch alive while it works
  on later ones), so peak activation memory grows roughly linearly in `V`.
  Interleaving trades memory back for bubble — the same currency GPipe was
  spending, just at a better exchange rate.

The reference code enforces the structural requirements:

```python
assert num_virtual_stages % parallel_dims.pp == 0
stages_per_rank = num_virtual_stages // parallel_dims.pp
assert not is_single_stage_schedule or stages_per_rank == 1
assert is_single_stage_schedule or stages_per_rank >= 2
```

Single-stage schedules (GPipe, 1F1B) require exactly one chunk per rank;
multi-stage schedules require at least two. And `model_parts` is a **list**
precisely because of this — the reason for topic 13 §1's plural.

## 5. Zero bubble: split the backward pass

Now the good part. Recall topic 11 §1.3: a linear layer's backward pass produces
**two** independent quantities from two separate matmuls of equal cost:

```
B :  ∂L/∂x = (∂L/∂y) Wᵀ      ← the PREVIOUS STAGE is blocked on this
W :  ∂L/∂W = xᵀ (∂L/∂y)      ← only the OPTIMIZER needs this, at step end
```

```
t_b = t_B + t_W  ,     t_B = t_W = t_f
```

**The dependency structures are completely different:**

```
B(m) on stage i   →   B(m) on stage i−1        a chain: latency-critical
W(m) on stage i   →   optimizer.step()          no successor until the step ends
```

`W` has **no downstream dependency within the step**. It is a pile of work that
must happen sometime before `optimizer.step()` and is otherwise unconstrained.

That is exactly what you need to fill a bubble.

### 5.1 The schedule

Run `B` as eagerly as possible (to unblock upstream stages) and use `W` as
filler whenever a rank would otherwise idle:

```
1F1B      (t_b = t_B + t_W as one indivisible block):
rank 3   F1 B1  F2 B2  F3 B3  ...
rank 0   F1 F2 F3 F4  ·  ·  ·  B1 ...
                      └─ idle: nothing to do ─┘

Zero-bubble (B and W separated):
rank 0   F1 F2 F3 F4 W· W· W· B1 W1 ...
                     └─ deferred W fills the fill-phase bubble ─┘
```

The three variants in the paper:

| Schedule | Bubble | Memory | Idea |
|----------|--------|--------|------|
| **ZB-H1** (`ZBH1`) | ~1/3 of 1F1B | = 1F1B | reorder `W` into 1F1B's bubbles at unchanged memory |
| **ZB-H2** | ≈ 0 | ~2× 1F1B | allow more in-flight micro-batches so `W` covers everything |
| **ZBV** (`ZBVZeroBubble`) | ≈ 0 | = 1F1B | V-shaped placement: zero bubble *at 1F1B memory* |

ZBV is the notable result: near-zero bubble with no memory penalty, which makes
it the best-known point on the trade-off curve.

### 5.2 The V-shaped placement, and why it works

```python
# V-shaped placement used by ZBVZeroBubble. Each rank owns exactly two stages:
# rank i gets stage i from the first half and stage num_stages - 1 - i from the
# second half. With 8 stages and 4 ranks, the rank path is 0,1,2,3,3,2,1,0;
# rank 0 owns stages (0, 7), rank 1 owns (1, 6), and so on.
assert stages_per_rank == 2
stage_v_pairs = list(zip(range(pp_degree), range(num_stages - 1, pp_degree - 1, -1)))
stage_indices = stage_v_pairs[pp_rank]
```

For `P = 4`, `num_stages = 8`: `[(0,7), (1,6), (2,5), (3,4)]`.

```
   stage:   0    1    2    3    4    5    6    7
   rank:    0    1    2    3    3    2    1    0
            └────────── down ──────────┘
                                  └──── back up ────┘
```

Compare with the loop placement (`0,1,2,3,0,1,2,3`). The V-shape has one crucial
property:

> **The first stage and the last stage live on the same rank.**

So the forward wave and the backward wave meet where they started. A micro-batch's
last forward chunk (stage 7, rank 0) is immediately followed by its first
backward chunk (also stage 7, also rank 0) — no network hop, no wait. The
pipeline closes into a loop, and the drain phase of one micro-batch overlaps the
fill of another with no dead time.

The loop placement cannot do this: stage 7 is on rank 3 and stage 0 is on rank 0,
so the wave must travel back across all ranks before rank 0 has anything to do.

Note the code asserts `stages_per_rank == 2` for ZBV — the V has exactly two
arms.

## 6. Choosing a schedule

```
Does the model fit without PP?              →  don't use PP
Is M ≥ 4·P_pp achievable?                   →  1F1B is fine (bubble < 20%)
Bubble still hurting, memory available?     →  Interleaved 1F1B (V = 2)
Bubble hurting, memory tight?               →  ZBV
Learning the material?                      →  GPipe, then read pp_gpipe.py
```

Selected by name in the reference code:

```python
schedule_class = get_schedule_class(job_config.parallelism.pipeline_parallel_schedule)
is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
looped_schedule = issubclass(schedule_class, PipelineScheduleMulti)
```

and `pipeline_module_split` picks the placement from the schedule class:

```python
style = "v" if schedule_class is ScheduleZBVZeroBubble else "loop"
```

## 7. `scale_grads=False`

```python
schedule = schedule_class(stages, n_microbatches=n_microbatches,
                          loss_fn=loss_fn, scale_grads=False)
```

By default a PyTorch pipeline schedule divides gradients by `n_microbatches`,
implementing the `1/M` in `∂L/∂θ = (1/M) Σ_m ∂L_m/∂θ`.

It is disabled here because that normalization is **wrong** for this trainer.
Topic 13 §4 divides by the *global valid token count* instead — which differs
from `1/M` whenever micro-batches contain different numbers of real tokens
(padding, packing, ragged batches). Letting the schedule apply `1/M` and then
dividing by tokens as well would double-normalize.

So the trainer takes over, with the post-hoc correction from topic 13 §4.1:

```python
if parallel_dims.pp_enabled:
    for m in self.model_parts:
        for p in m.parameters():
            if p.grad is not None:
                p.grad.div_(global_valid_tokens)
```

Two components, one invariant, one flag connecting them. Worth tracing once — it
is the kind of interaction that produces a "model trains 5% worse and nobody
knows why" if either side is changed in isolation.

## 8. Summary

| | GPipe | 1F1B | Interleaved | ZBV |
|---|---|---|---|---|
| bubble | `(P−1)/(M+P−1)` | same | `(P−1)/(VM+P−1)` | ≈ 0 |
| activation memory | `O(M)` | `O(P)` | `O(P·V)` | `O(P)` |
| stages per rank | 1 | 1 | `V ≥ 2` | 2 (V-shaped) |
| messages per micro-batch | `P−1` | `P−1` | `V·P−1` | `V·P−1` |
| complexity | low | low | medium | high |

```
1F1B dominates GPipe outright.
Interleaved buys bubble with memory and message count.
ZBV buys bubble by exploiting that ∂L/∂W has no downstream dependency.
```

## 9. Check yourself

1. Why does 1F1B have the same bubble as GPipe? Give the critical-path argument.
2. Why is 1F1B's activation memory `O(P)` rather than `O(M)`? What is the peak on
   stage `i`?
3. Derive `bubble = (P−1)/(V·M + P−1)` for interleaved 1F1B. Where exactly does
   the `V` enter?
4. Interleaving reduces the bubble `V×`. Name its two costs and say which one
   dominates on a slow inter-node link.
5. State the two quantities the backward pass computes and the cost of each. Why
   does `∂L/∂W` have no downstream dependency within a step?
6. Explain how deferring `W` fills the pipeline bubble. Why can't `B` be deferred
   the same way?
7. For `P = 4, num_stages = 8`, list the stage assignments under loop placement
   and under V placement. Which property of the V-shape enables zero bubble?
8. Why does ZBV assert `stages_per_rank == 2`?
9. Why is `scale_grads=False`? What would go wrong with the default, and in which
   specific data situations?
10. You have `P_pp = 8`, `M = 8`, and 12 GB of spare memory per GPU. Which
    schedule, and what bubble do you expect?

## Next

→ [16 — Datasets, tokenization and data parallelism](16-datasets-tokenization-data-parallelism.md)
