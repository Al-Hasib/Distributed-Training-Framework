# 13 — Building the Training Loop

> Video: 05:10:47 — Building the Training Loop
> Code: `torchfeather/train.py`

## Why this exists

The training loop is where every abstraction in the framework has to actually
agree with every other one. It is also where the *order* of operations is
load-bearing in ways that are invisible until they fail.

This topic builds the loop that topics 14–30 will keep modifying. Read it once
now for the structure, and return to it after each parallelism chapter to see
what that chapter added.

Two things in here are worth the price of admission on their own:

- **§4 — token-weighted loss normalization**, the concrete implementation of
  topic 11 §3.1. The reference code does this correctly and most tutorials do
  not.
- **§2 — the construction order**, which is not negotiable and which nothing
  will warn you about.

## 1. The step, in the abstract

```python
def train_step(batch):
    optimizer.zero_grad()
    loss = loss_fn(model(batch.inputs), batch.labels)
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
    lr_scheduler.step()
```

Six lines. Every one of them changes under parallelism:

| line | what parallelism does to it |
|------|------------------------------|
| `zero_grad` | must cover *all* model parts (PP gives you a list, not one module) |
| `model(...)` | becomes a pipeline schedule step, or runs inside a CP context |
| `loss_fn` | must be reduced with the correct global token count |
| `backward` | already done by the PP schedule; or triggers FSDP reduce-scatters |
| `clip_grad_norm_` | the norm is a *global* quantity → needs collectives across meshes |
| `optimizer.step` | operates on DTensor shards; MoE needs a load-balancing hook |

Note in particular that `model_parts` is a **list**. Under pipeline parallelism
with interleaved stages, one rank owns several non-contiguous chunks of the model
(topic 15), so "the model" is plural from here on.

## 2. Construction order — and why it cannot be permuted

From `Trainer.__init__`:

```python
# 1. device and process group
self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
device_module.set_device(self.device)
torch.distributed.init_process_group(backend="nccl", timeout=...)

# 2. the device mesh (topic 18)
self.parallel_dims = self._create_parallel_dims(parallelism_config, world_size)
_ = parallel_dims.world_mesh

# 3. determinism, seeded per mesh dimension (§7)
dist_utils.set_determinism(...)

# 4. tokenizer and dataloader — need dp_degree and dp_rank from the mesh
# 5. model on the meta device — structure only, zero bytes
# 6. parallelize: PP split, TP plans, FSDP wrapping   (topics 17-25)
# 7. materialize: to_empty(device)
# 8. init_weights(buffer_device)                       (topic 05 §4)
# 9. optimizer, lr scheduler                           (topic 24)
# 10. checkpointer, then load
```

Four of these orderings are traps:

**(a) `set_device` before `init_process_group`.** NCCL binds communicators to the
current device. Initialize the group first and every rank may pick device 0,
producing either a hang or a silent single-GPU run.

**(b) The mesh before the dataloader.** The dataloader must know `dp_degree` and
`dp_rank` to shard the dataset, and those come from the mesh — not from
`WORLD_SIZE`. On a mesh with `P_tp = 8`, eight ranks share the *same* data shard
(they are the same data-parallel replica, split by matrices). Using
`global_rank`/`world_size` here is a real and common bug: it silently gives
every TP rank different data, which corrupts the tensor-parallel invariant that
they are computing one logical forward pass.

```python
if parallel_dims.dp_enabled:
    batch_mesh = parallel_dims.get_mesh("batch")
    dp_degree, dp_rank = batch_mesh.size(), batch_mesh.get_local_rank()
else:
    dp_degree, dp_rank = 1, 0
```

Note it asks the mesh for the `"batch"` dimension, not for `dp`. Topic 18
explains why that flattened view exists.

**(c) Parallelize before materialize before initialize.** Covered in topic 05
§4. Sharding a meta-device model is free; sharding a materialized one requires
holding the whole thing first. And initializing before sharding gives *different
values*, because each rank would draw from its own RNG over a different slice.

**(d) Checkpoint load last.** It must come after the optimizer exists, because
optimizer state is part of the checkpoint, and after parallelization, because
the loaded shards must match the sharding plan.

### 2.1 `Trainer(Stateful)`

```python
class Trainer(Stateful):
    step: int
    ntokens_seen: int
```

Implementing `torch.distributed.checkpoint.stateful.Stateful` means the trainer
itself participates in checkpointing — `step` and `ntokens_seen` are saved
alongside the weights. Sounds like bookkeeping; it is correctness. Resume with
the wrong `step` and your learning-rate schedule jumps, which is a genuine
mid-run loss spike that is very hard to diagnose after the fact.

## 3. Gradient accumulation, done in two phases

The step's most instructive feature is that it iterates the data **twice**:

```python
# Phase 1: collect all micro-batches and count valid tokens
microbatches = []
local_valid_tokens = torch.tensor(0, dtype=torch.int64, device=self.device)
for _ in range(self.gradient_accumulation_steps):
    input_dict, labels = next(data_iterator)
    local_valid_tokens += (labels != IGNORE_INDEX).sum()
    microbatches.append((input_dict, labels))

local_valid_tokens //= self.parallel_dims.cp

if parallel_dims.dp_cp_enabled:
    global_valid_tokens = dist_utils.dist_sum(
        local_valid_tokens, parallel_dims.get_mesh("loss"))
else:
    global_valid_tokens = local_valid_tokens.float()

# Phase 2: forward/backward each micro-batch, normalized by the global count
accumulated_losses = []
for input_dict, labels in microbatches:
    loss = self.forward_backward_step(input_dict, labels, global_valid_tokens)
    accumulated_losses.append(loss.detach())
```

Why two phases, when one would be simpler and use less memory? Because the
normalizer must be known *before* the first backward pass. You cannot divide by
a total you have not finished counting, and you cannot rescale gradients after
the fact without an extra pass over all of them.

Three details in those ten lines:

**`local_valid_tokens //= parallel_dims.cp`.** Under context parallelism every
CP rank loads the **full** sequence and CP splits it internally (topic 23). So
all `P_cp` ranks count the same tokens. Summing over a mesh that includes `cp`
would multiply the count by `P_cp`, so it is divided out first. This is the sort
of correction that is obvious once stated and invisible otherwise.

**The `"loss"` mesh.** Not `dp`, not `world` — the flattened `dp × cp` mesh,
because those are exactly the dimensions across which the batch's tokens are
distributed. `tp` ranks hold the *same* tokens (they split matrices), so
including `tp` would over-count; `pp` ranks hold different *layers*, and only the
last stage computes a loss at all.

**"If data runs out during gradient accumulation, that entire step will not be
executed."** Phase 1 raises `StopIteration` before any gradient exists, so you
never commit a partial step. Cheap, and it removes a whole class of
end-of-epoch bugs.

## 4. Token-weighted loss normalization

This is topic 11 §3.1 in production form. Start with the loss function:

```python
def cross_entropy_loss(pred, labels):
    """Cross-entropy loss with sum reduction for token-based normalization."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",              # ← SUM, not mean
        ignore_index=IGNORE_INDEX,
    )
```

`reduction="sum"`. The loss function deliberately does **not** average, because
it does not know the right denominator — that is a global quantity spanning
every rank and every micro-batch. Then:

```python
loss_sum = self.loss_fn(pred, labels)
loss = loss_sum / global_valid_tokens        # normalize BEFORE backward
del pred                                     # free before bwd to cut peak memory
loss.backward()
```

Trace what this achieves. Each rank computes

```
loss_local = (Σ_{i ∈ this rank, this µbatch} ℓ_i) / (Σ_all ranks, all µbatches n)
```

Sum over micro-batches (gradient accumulation) and over ranks (the DP
all-reduce) and you get

```
Σ_p Σ_m loss_local  =  (Σ_all ℓ_i) / (Σ_all n)   =  the true token-mean loss  ✓
```

**Total over total, never mean of means.** Correct under padding, packing, ragged
batches, and unequal micro-batch sizes — all the cases where the naive version
is silently wrong.

Also note `.float()` inside the loss: cross-entropy over `V = 102400` classes
accumulates a `logsumexp` over 100k terms, and doing that in bf16 loses
meaningful precision in the loss *and* the gradient. Upcast the logits.

And `del pred` before `backward()`: the logits tensor is `[B, S, V]` — at
`B=1, S=16384, V=102400` in fp32 that is **6.7 GB**. It is not needed once the
loss is computed (cross-entropy's VJP saves what it needs), so dropping the
Python reference before backward removes it from peak memory. One of the highest
value-per-character lines in the file.

### 4.1 The pipeline-parallel exception

```python
# The PP schedule runs backward on the raw sum loss; non-PP divides by
# global_valid_tokens before backward.
if parallel_dims.pp_enabled:
    for m in self.model_parts:
        for p in m.parameters():
            if p.grad is not None:
                p.grad.div_(global_valid_tokens)
```

Under PP, `pp_schedule.step()` runs forward *and* backward internally — by the
time you get control back, gradients already exist and were computed from the
unnormalized sum loss. Since scaling a loss scales its gradient linearly
(`∂(αL)/∂θ = α ∂L/∂θ`), dividing the gradients after the fact is exactly
equivalent. It is applied on **every** stage, not just the last: gradients on
earlier stages descend from the same unnormalized loss.

This is a good example of a general pattern: when a framework component owns the
backward pass, your only remaining hooks are before the forward and after the
backward. Linearity is what lets you move the correction to the far side.

## 5. Gradient clipping across a mesh

```python
grad_norm = dist_utils.clip_grad_norm_(
    [p for m in self.model_parts for p in m.parameters()],
    self.job_config.training.max_norm,
    foreach=True,
    pp_mesh=parallel_dims.get_optional_mesh("pp"),
    ep_enabled=parallel_dims.ep_enabled,
)
```

Clipping needs the **global** gradient norm:

```
‖g‖₂ = sqrt( Σ_all parameters Σ_all elements g² )
```

and the parameters are scattered across every mesh dimension. So the norm is a
distributed reduction with three distinct cases:

- **DTensor-sharded parameters (TP, FSDP).** Each rank holds part of a tensor.
  DTensor's `norm` handles the reduction internally — which is why it is a
  *collective*, a fact that matters in §6.
- **Pipeline stages.** Each stage holds *different parameters*, and PP
  parameters are not DTensors on the `pp` dimension. So the partial sums of
  squares must be explicitly all-reduced over `pp_mesh` — hence the argument.
- **Expert parallelism.** Experts are sharded across `ep`, but an expert's
  gradient is not replicated, so the `ep` dimension must be included exactly
  once. Hence `ep_enabled` (topic 28).

Get any of these wrong and you get a norm that is too small, so clipping never
engages, so a loss spike that clipping was supposed to absorb takes the run down.
Silent until it is catastrophic.

## 6. Metrics only on logging steps

```python
should_log = self.metrics_processor.should_log(self.step)
parameter_metrics = (
    collect_parameter_norm_metrics(self.model_parts, pp_mesh=...)
    if should_log else {}
)
```

The comment in the source says it plainly: *"since DTensor norm reduction is a
collective"*. Per-parameter norms look like free diagnostics. On a sharded model
each one is an all-reduce, and a 2.88B model has hundreds of parameter tensors.
Collecting them every step would add hundreds of collectives per step and
measurably slow the run.

The general rule, which applies to anything you are tempted to log:

> **Any metric over a sharded tensor is a collective. Gate it behind
> `should_log`.**

And when you do log, log the right things:

```python
global_avg_loss = dist_utils.dist_sum(loss, dp_cp_mesh)
local_avg_loss  = loss * global_valid_tokens / local_valid_tokens
global_max_loss = dist_utils.dist_max(local_avg_loss, dp_cp_mesh)
```

`global_avg_loss` is a **sum**, not a mean — because each rank's `loss` was
already divided by the *global* token count, so summing reassembles the mean
(§4). `global_max_loss` first undoes that normalization to recover each rank's
own per-token average, then takes the max. That max is the useful signal: a
single rank diverging shows up there long before it moves the average.

## 7. Determinism and seeding across mesh dimensions

`set_determinism` is short and every line of it is a decision (topic 11 §6.1):

```python
if seed is None:
    seed_tensor = torch.get_rng_state()[:8].to(device)
    torch.distributed.broadcast(seed_tensor, src=0)      # everyone agrees
    seed = seed_tensor.to("cpu").view(torch.uint64).item()

# then offset the seed by the rank along the "distinct" mesh dims (default ["pp"])
seed_offset = 0; cumulative_size = 1
for distinct_mesh in distinct_meshes:
    seed_offset += distinct_mesh.get_local_rank() * cumulative_size
    cumulative_size *= distinct_mesh.size()
seed = (seed + seed_offset) % 2**64
```

The rule it implements:

```
same seed across SPMD dims (dp, tp, cp)      — ranks must agree
different seed across pp (and optionally ep) — ranks must differ
```

Why the asymmetry, from the docstring:

- **Across `pp`:** different stages hold *different layers*, so their dropout
  masks should be independent — the same seed would correlate masks between
  layer 3 (stage 0) and layer 15 (stage 1) for no reason.
- **Across `tp`/`cp`:** ranks cooperate on *one* logical tensor. DTensor's RNG
  tracker takes the shared seed and derives per-shard offsets so that the union
  of shards looks like one correctly-sampled tensor. Give them different seeds
  and you break that.
- **Not for FSDP:** as the docstring notes, each FSDP rank has its own RNG state
  affecting all layers it owns, so two layers on the same rank naturally get
  different masks without help.

Note also that `seed=None` **broadcasts rank 0's seed** rather than defaulting to
a constant. This matters for reproducibility: the seed is recorded, and every
rank provably starts from the same one rather than from whatever its own
environment happened to provide.

## 8. Two operational details that look like noise

### 8.1 Manual garbage collection

```python
self.gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)
```

with the comment *"take control of garbage collection to avoid stragglers"*.

Python's cyclic GC can fire at any time and take tens of milliseconds. In a
synchronous collective world, **one rank pausing stalls all of them** — every
other rank sits in the next all-reduce waiting. Worse, the pauses are
uncorrelated, so with `P` ranks the probability that *some* rank is in a GC pause
grows with `P`.

The fix: disable automatic GC and call `gc.collect()` at a fixed step interval on
every rank simultaneously. The pause still happens, but it happens *in lockstep*,
so it costs one pause instead of `P` staggered ones. This is a genuine
multi-percent throughput effect at scale, and a good illustration of a rule that
governs all synchronous distributed training:

> **Anything that makes one rank slower makes every rank slower. Variance is
> more expensive than latency.**

### 8.2 `@record`

```python
from torch.distributed.elastic.multiprocessing.errors import record

@record
def __init__(self, job_config): ...
```

Without it, a crash on rank 137 of 512 produces a torn, interleaved traceback
and often only reports that some process exited non-zero. `@record` captures the
exception per rank and surfaces the *first* failure with its rank attributed.
On a 512-GPU run, this is the difference between a five-minute diagnosis and an
afternoon.

## 9. The full step, annotated

```python
def train_step(self, data_iterator):
    self.optimizers.zero_grad()
    lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]     # log before stepping

    # ── phase 1: collect micro-batches, count tokens globally ──────────────
    microbatches, local_valid_tokens = [], torch.tensor(0, dtype=torch.int64, ...)
    for _ in range(self.gradient_accumulation_steps):
        input_dict, labels = next(data_iterator)               # may StopIteration
        local_valid_tokens += (labels != IGNORE_INDEX).sum()
        microbatches.append((input_dict, labels))
    local_valid_tokens //= self.parallel_dims.cp               # CP double-count
    global_valid_tokens = dist_utils.dist_sum(local_valid_tokens,
                                              parallel_dims.get_mesh("loss"))

    # ── phase 2: forward/backward, each normalized by the global count ─────
    accumulated_losses = []
    for input_dict, labels in microbatches:
        loss = self.forward_backward_step(input_dict, labels, global_valid_tokens)
        accumulated_losses.append(loss.detach())

    # ── PP ran backward itself, so rescale gradients after the fact ────────
    if parallel_dims.pp_enabled:
        for m in self.model_parts:
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.div_(global_valid_tokens)

    # ── clip (a global reduction), step, advance the schedule ──────────────
    grad_norm = dist_utils.clip_grad_norm_(..., pp_mesh=..., ep_enabled=...)
    self.optimizers.step()
    self.lr_schedulers.step()

    # ── metrics, gated: every sharded-tensor metric is a collective ────────
    if self.metrics_processor.should_log(self.step):
        ...
```

## 10. Check yourself

1. Give the construction order and explain what breaks if you (a) call
   `init_process_group` before `set_device`, (b) build the dataloader before
   the mesh, (c) call `init_weights` before applying FSDP.
2. Why must the dataloader shard by `dp_rank` from the mesh rather than by
   `global_rank`? What exactly goes wrong under `P_tp = 8`?
3. Why does `train_step` iterate the data in two phases instead of one?
4. Why is `local_valid_tokens` divided by `parallel_dims.cp`?
5. Why is the loss mesh `dp × cp` and not `world`? Argue for each excluded
   dimension.
6. `cross_entropy_loss` uses `reduction="sum"`. Show that summing the per-rank,
   per-micro-batch losses reconstructs the true token-mean.
7. Under PP, gradients are divided by `global_valid_tokens` *after* backward.
   Which property of differentiation makes that legitimate?
8. `del pred` before `backward()`. Compute how much memory that saves at
   `B=1, S=16384, V=102400` in fp32.
9. Why is `collect_parameter_norm_metrics` gated behind `should_log`?
10. `global_avg_loss` is computed with `dist_sum`, not a mean. Why is that
    correct? And why is `global_max_loss` the more useful alarm signal?
11. Why do `pp` ranks need different seeds while `tp` ranks need the same one?
12. Explain how manual `gc.collect()` at a fixed interval improves throughput,
    and state the general principle it illustrates.

## Part II complete

You now have the mathematics (topic 11), the graph picture and the one systems
idea that makes communication cheap (topic 12), and the loop that everything
plugs into (topic 13).

From here on, each topic answers the same three questions:

1. **Which axis is being cut?** (layers, batch, matrices, sequence, experts)
2. **Which collective repairs the cut, forward and backward?** (topic 11 §2)
3. **What does it cost in bytes, and does it fit on the available link?**
   (topic 01 §2)

## Next

→ [14 — Pipeline parallelism from first principles](14-pipeline-parallelism-first-principles.md)
