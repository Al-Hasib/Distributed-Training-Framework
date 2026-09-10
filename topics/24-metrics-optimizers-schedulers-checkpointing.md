# 24 — Metrics, Optimizers, Schedulers and Checkpointing

> Video: 15:41:22 — Metrics, Optimizers, Schedulers and Checkpointing
> Code: `torchfeather/components/{optimizer,lr_scheduler,metrics,checkpoint}.py`

## Why this exists

These four components look like infrastructure you could copy from a tutorial.
Each of them acquires a genuinely distributed problem the moment parameters
become DTensors spread over a device mesh:

- **Optimizer** — parameters live on *different meshes* (dense on `fsdp`, experts
  on `efsdp`), and fused kernels cannot mix them.
- **Scheduler** — one per model part, and resuming with the wrong step silently
  corrupts the schedule.
- **Metrics** — every statistic over a sharded tensor is a collective.
- **Checkpointing** — the sharding plan must not be baked into the saved files,
  or you can never change your parallelism configuration again.

## 1. The optimizer

### 1.1 `OptimizersContainer`: one optimizer per model part

```python
class OptimizersContainer(Optimizer, Stateful, Generic[T]):
    optimizers: list[T]
    model_parts: list[nn.Module]
```

A **list**, for the reason established in topic 13 §1: under interleaved pipeline
parallelism a rank owns several model chunks. The container implements the
`Optimizer` interface so the training loop keeps calling `zero_grad()` and
`step()` on one object, and implements `Stateful` so the whole thing
checkpoints as a unit.

### 1.2 Parameter groups by mesh — a real distributed constraint

```python
def _group_params_by_mesh(params):
    """Group parameters by DTensor device mesh.

    Fused/foreach optimizer ops batch parameters together, but DTensor dispatch
    requires all operands to be on the same mesh. When EP is enabled, expert
    params live on a different mesh than regular FSDP params.
    Splitting into separate param groups ensures each group is processed
    independently.
    """
    mesh_groups = {}
    for p in params:
        key = p.data.device_mesh.mesh_dim_names if isinstance(p.data, DTensor) else None
        mesh_groups.setdefault(key, []).append(p)
    if len(mesh_groups) <= 1:
        return params
    return [{"params": group} for group in mesh_groups.values()]
```

Trace the problem. `foreach` and `fused` Adam batch many parameters into one
kernel — a large speedup, since Adam on thousands of small tensors is
launch-bound. But a batched op over DTensors requires **all operands on the same
mesh**, because the op has to have one well-defined placement.

And from topic 20 §8, with EP enabled the meshes genuinely differ:

```
dense parameters   → mesh ("dp_replicate", "fsdp")
routed experts     → mesh ("dp_replicate", "efsdp")
```

Batch them together and DTensor dispatch fails. So group by
`mesh_dim_names` and let each group be processed independently — you keep
`foreach` *within* a mesh and lose it only *across* meshes, of which there are
at most two.

The `len(mesh_groups) <= 1` early return matters: with no EP there is one mesh,
and returning a flat parameter list rather than a one-element group list keeps
the common path on the simplest code.

This is a good example of a theme: **a performance feature (fused optimizers) and
a correctness constraint (single-mesh dispatch) collide, and the resolution is a
data-layout decision.**

### 1.3 The MoE load-balancing hook

`build_optimizers_with_moe_load_balancing` installs a post-step hook that
implements DeepSeek-V3's auxiliary-loss-free balancing (topic 26; paper
2408.15664):

```python
def _update_expert_bias(model_parts, parallel_dims):
    tokens_per_expert_list = []
    for model_part in model_parts:
        for transformer_block in model_part.layers.values():
            if not transformer_block.moe_enabled: continue
            if moe.load_balance_coeff is None: continue
            tokens_per_expert = moe.tokens_per_expert
            if ...:
                tokens_per_expert = tokens_per_expert // 2
            tokens_per_expert_list.append(tokens_per_expert)

    tokens_per_expert_by_layer = torch.vstack(tokens_per_expert_list)
    if pg is not None:
        torch.distributed.all_reduce(tokens_per_expert_by_layer, group=pg,
                                     op=torch.distributed.ReduceOp.SUM)

    for ... each moe ...:
        expert_bias_delta = moe.load_balance_coeff * torch.sign(
            tokens_per_expert.mean() - tokens_per_expert)
        expert_bias_delta = expert_bias_delta - expert_bias_delta.mean()
        moe.expert_bias.add_(expert_bias_delta)
        moe.tokens_per_expert.zero_()

optimizer.register_step_post_hook(lambda *_a, **_kw: _update_expert_bias(...))
```

Four design decisions worth extracting:

**(a) It is not a loss.** `expert_bias` is updated by a **rule**, outside
autograd, in a `no_grad` post-step hook. This is the entire point of
"auxiliary-loss-free": the classic balancing loss adds a term to the objective
that competes with language modelling. A direct bias nudge balances the router
without touching the gradient at all.

**(b) `torch.sign`, not the magnitude.** The update is
`coeff · sign(mean − count)`: over-loaded experts get their bias reduced by
exactly `coeff`, under-loaded ones raised by exactly `coeff`. A fixed step size
makes the dynamics stable regardless of how extreme the imbalance is —
proportional control here would oscillate.

**(c) `delta - delta.mean()`.** The update is projected to be **zero-sum**, so
the biases do not drift collectively. Only *relative* bias affects a top-k
selection, so a common offset is a null direction; removing it prevents
unbounded growth in an unobservable direction.

**(d) The all-reduce is essential.** `tokens_per_expert` is a *local* count. The
router must be balanced with respect to the **global** token distribution, so the
counts are summed across data-parallel ranks first. Note the `vstack` into one
`[n_layers, n_experts]` tensor: **one collective for the whole model** instead of
one per layer. At 27 layers that is a 27× reduction in launch overhead for a
per-step operation.

Why in a post-step hook rather than the training loop? Because it must run
exactly once per optimizer step, after the weights update, and it must be part of
whatever owns `step()` — including under gradient accumulation, where the
training loop runs several forward/backward passes per step. Attaching it to the
optimizer makes "once per step" structural rather than a convention.

## 2. The learning-rate schedule

### 2.1 Warmup–stable–decay

```python
warmup_steps = int(lr_scheduler_config.warmup_steps)
if warmup_steps > training_steps:
    warmup_steps = training_steps            # clamp, with a warning

if lr_scheduler_config.decay_ratio is not None:
    decay_steps = round(training_steps * lr_scheduler_config.decay_ratio)
    if warmup_steps + decay_steps > training_steps:
        decay_steps = training_steps - warmup_steps      # clamp
else:
    decay_steps = training_steps - warmup_steps

stable_steps = training_steps + 1 - warmup_steps - decay_steps
```

Three phases:

```
lr
 │      ╭──────────────────╮
 │     ╱                    ╲
 │    ╱                      ╲___  min_lr_factor · lr
 │   ╱                           
 └──┴────────┴─────────────┴──────►  steps
    warmup     stable         decay
```

- **Warmup** (linear, from ~0). Adam's second moment `v` starts at zero, so early
  updates are effectively divided by a tiny, high-variance estimate — the update
  magnitude is wild. Warmup lets `v` accumulate before the LR reaches full size.
  This is also the mechanism topic 05 §1.1 said pre-norm *reduces* the need for
  but does not eliminate.
- **Stable.** The bulk of training at peak LR.
- **Decay** to `min_lr_factor · lr`. Late-training decay is empirically worth a
  substantial loss improvement; the WSD shape additionally lets you *branch* a
  decay from a stable-phase checkpoint, so one long stable run yields models at
  several token budgets.

Note the defensive clamping. `warmup + decay > training_steps` is easy to
produce by editing `training.steps` and forgetting the scheduler, and the
un-clamped result is a negative `stable_steps` and a nonsensical schedule. It
warns and fixes rather than failing — the right call for a config error that has
an obvious correct interpretation.

### 2.2 Resuming: `last_epoch` and why it must be exact

```python
def load_state_dict(self, state_dict):
    # Load the same state_dict for all schedulers. The key value we're concerned
    # with in ``LRScheduler.state_dict()`` is ``last_epoch``, which is an integer
    # that is immutable. As long as ``training.steps`` and
    # ``lr_scheduler.warmup_steps`` in ``job_config`` remain unchanged when
    # resuming from a checkpoint, this approach is safe. We call ``copy()`` here
    # to ensure extra safety.
```

All schedulers (one per model part) get the *same* state, since they follow the
same schedule. The stated caveat is real: the saved state is only `last_epoch`,
so the *shape* of the schedule comes from the current config. Change
`training.steps` on resume and the LR jumps discontinuously — a visible loss
spike, from a config edit, with no error.

This is why `Trainer` is `Stateful` and saves `step` (topic 13 §2.1). The three
must agree: the checkpoint's step, the scheduler's `last_epoch`, and the config's
`training.steps`.

## 3. Metrics

The rule from topic 13 §6: **any statistic over a sharded tensor is a
collective**, so gate it.

```python
should_log = self.metrics_processor.should_log(self.step)
parameter_metrics = (
    collect_parameter_norm_metrics(self.model_parts, pp_mesh=...)
    if should_log else {}
)
```

`should_log(self.step)` is a pure function of the step, hence **identical on
every rank** — satisfying topic 19 §5.1's rule for conditionals around
collectives. A condition like `if loss > threshold` would hang.

The three quantities and why each:

```python
global_avg_loss = dist_utils.dist_sum(loss, dp_cp_mesh)
local_avg_loss  = loss * global_valid_tokens / local_valid_tokens
global_max_loss = dist_utils.dist_max(local_avg_loss, dp_cp_mesh)
```

- **`global_avg_loss` is a SUM.** Each rank's `loss` was already divided by the
  *global* token count (topic 13 §4), so summing reassembles the global mean. A
  mean here would divide twice.
- **`global_max_loss`** first *undoes* that normalization to recover each rank's
  own per-token average, then takes the max. This is the alarm signal: one rank
  diverging (a bad data shard, a hardware fault, an expert collapse) moves the
  max long before it moves the average.
- **`grad_norm`**, from `clip_grad_norm_` (topic 13 §5). The single most useful
  diagnostic in training: a spike precedes a loss spike, and a norm trending to
  zero means the model has stopped learning.

## 4. Checkpointing

The hard part is that a checkpoint must be **independent of the parallelism
configuration**. You want to save on 512 GPUs with `P_pp=4, P_tp=8` and resume on
64 with `P_pp=1, P_tp=8`.

### 4.1 Distributed Checkpoint (DCP)

`torch.distributed.checkpoint` saves a **logical** state dict: each DTensor is
written with its global shape and its shards' offsets, so loading redistributes
into whatever sharding the new job has. Each rank writes only its own shards, in
parallel — no gather to rank 0, no 1.2 TB single-file write.

Two things had to be true for this to work, and both were arranged much earlier:

- **`ModuleDict` keyed by layer id** (topic 05 §2.1), so parameter names are
  independent of the pipeline split.
- **`persistent=False` on `freqs_cis`** (topic 04 §6.1), so a `max_seq_len`
  change does not produce a shape mismatch on load.

Those two decisions, made in Part I for reasons that looked local, are what make
resharding possible.

### 4.2 `ModelWrapper`

```python
class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module | list[nn.Module]) -> None: ...
    def _get_state_dict(self) -> dict[str, Any]: ...
```

Accepts a model *or a list* — pipeline parallelism again — and merges the parts
into one logical state dict. Because the keys carry original layer ids, the union
across stages *is* the full model's state dict with no remapping.

### 4.3 Async checkpointing

```python
if self.config.async_mode == AsyncMode.ASYNC:
    # Creates a new CPU process group for async checkpointing in which all
    # ranks participate.
    self.async_save_pg = dist.new_group(backend="gloo")
```

A 2.88B model's full state is ~52 GB; at 512 GPUs a checkpoint is tens of TB and
a synchronous save stalls every rank for minutes. Async save copies the state to
host memory and writes from a background thread while training continues.

Note the **separate `gloo` process group**. Two reasons:

- The write coordination is CPU-side and must not contend with NCCL
  communicators on the GPU stream.
- Using the main NCCL group would interleave checkpoint collectives with training
  collectives, and collectives are matched by call order (topic 19 §5.1) — a
  background thread issuing into the same group is a race that produces a hang.

A dedicated group for a dedicated purpose. This is the standard way to make
background communication safe.

### 4.4 Retention

```python
if self.config.keep_latest_k == 1:
    raise ValueError("keep_latest_k=1 is not supported because the last "
                     "checkpoint ...")
```

`keep_latest_k = 1` is refused because deletion races the write of the next
checkpoint: you would briefly have *zero* complete checkpoints, and a crash in
that window loses the run. The minimum safe value is 2 — write the new one, then
delete the old one.

Deletion happens on a `purge_thread` fed by a queue, with a `TerminateSentinel`
for clean shutdown. Filesystem deletion of a multi-TB checkpoint directory is
slow enough on network storage to stall a training step if done inline.

### 4.5 What must be in a checkpoint

```
model weights            all parts, DTensor-aware
optimizer state          m, v, master weights — per parameter
lr scheduler             last_epoch
trainer                  step, ntokens_seen           (topic 13 §2.1)
dataloader               _sample_idx, _token_buffer   (topic 16 §4)
MoE state                expert_bias, tokens_per_expert   ← easy to forget
```

The last row is the one that catches people. `expert_bias` is a **buffer** updated
by a rule outside autograd (§1.3). Omit it and every resume resets the router's
learned balance to zero, so the model re-enters the imbalanced regime and
throughput degrades for thousands of steps after every restart. It is not a
parameter, it has no gradient, and it is essential state.

The general rule: **anything mutated during training that is not a parameter is
still state.** Buffers, counters, RNG positions, data-loader offsets. Enumerate
them deliberately rather than relying on `state_dict()` to have caught them.

## 5. Check yourself

1. Why is `optimizers` a list rather than a single optimizer?
2. Why must parameters be grouped by device mesh? Which feature forces it, and
   which configuration makes the meshes differ?
3. Why does `_group_params_by_mesh` return a flat list when there is only one
   mesh?
4. Give the four design decisions in the expert-bias update: why not a loss, why
   `sign`, why zero-mean, why an all-reduce.
5. Why is the expert-bias update a *post-step hook* rather than a call in the
   training loop? Consider gradient accumulation.
6. Why does `tokens_per_expert_by_layer` use `vstack` before the all-reduce?
7. Explain warmup in terms of Adam's second moment. What else reduces (but does
   not remove) the need for it?
8. What does the WSD shape allow that cosine decay does not?
9. Why does the scheduler clamp `warmup + decay > training_steps` instead of
   raising?
10. Only `last_epoch` is checkpointed. What must stay constant across a resume,
    and what is the symptom if it does not?
11. Why is `global_avg_loss` a sum? Why is `global_max_loss` the more useful
    alarm?
12. Why must `should_log` be a function of the step and nothing else?
13. Which two Part I decisions make checkpoint resharding possible?
14. Why does async checkpointing create a separate `gloo` process group? Give
    both reasons.
15. Why is `keep_latest_k = 1` refused?
16. List everything a checkpoint must contain, and explain what breaks if
    `expert_bias` is omitted.

## Next

→ [25 — Combining parallelism in the training loop](25-combining-parallelism.md)
