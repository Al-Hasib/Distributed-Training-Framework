# 17 — Coding Pipeline Parallelism

> Video: 07:42:10 — Coding Pipeline Parallelism
> Code: `torchfeather/distributed/pipeline_parallel.py`, `minimal_examples/pp_gpipe.py`

## Why this exists

Topics 14 and 15 gave the theory. Implementing it requires solving four concrete
problems, and only the first is obvious:

1. **Cut the model** so that each rank holds only its layers — without breaking
   checkpoint keys.
2. **Cut the autograd graph** and re-connect it across the network. This is the
   hard one: `detach()` means gradients *cannot* flow, so they must be sent and
   re-injected by hand.
3. **Compose with the other parallelisms**, in the right order.
4. **Fix up the training loop** for the facts that only one stage has a loss and
   the schedule owns the backward pass.

We build (2) from scratch first — a 40-line working pipeline — because
`PipelineStage` hides exactly the mechanism you need to understand.

## 1. The heart of it: cutting and re-connecting the graph

### 1.1 Why the graph must be cut

Stage 0 computes `h = f₀(x)` and sends `h` to stage 1. It must send a **plain
tensor** — you cannot serialize an autograd graph over a socket. So stage 1
receives a tensor with no history: a fresh **leaf**.

That severs the chain rule. Stage 1's `loss.backward()` will compute `∂L/∂h` and
stop, because as far as stage 1 knows, `h` is where the world begins.

### 1.2 The repair, in two calls

```
stage 1:   h_recv.requires_grad_(True)   →  backward populates h_recv.grad
           send h_recv.grad upstream

stage 0:   torch.autograd.backward(tensors=[h_sent], grad_tensors=[grad_recv])
                                            └── re-inject the cotangent here
```

`torch.autograd.backward(tensors, grad_tensors)` is the general form of
`.backward()`: it seeds the reverse traversal with a *given* cotangent instead of
the implicit `1.0` that a scalar loss provides. That is precisely what we need —
stage 0's graph resumes from the gradient stage 1 computed.

This is the one autograd API you must know to implement pipeline parallelism, and
it is the concrete meaning of topic 11 §4.1's "`detach()` cuts the graph".

### 1.3 A complete, working pipeline in 40 lines

```python
# minipp.py — GPipe over 2 ranks, verified against a single-process reference.
import os, tempfile, torch, torch.nn as nn
import torch.distributed as dist, torch.multiprocessing as mp

D, M, MB = 8, 4, 2                          # hidden, micro-batches, micro-batch size

def build(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(D, D), nn.Tanh())

def worker(rank, world, initfile):
    dist.init_process_group("gloo", init_method=f"file:///{initfile}",
                            rank=rank, world_size=world)
    stage = build(100 + rank)                # rank r owns stage r
    torch.manual_seed(0)
    data   = [torch.randn(MB, D) for _ in range(M)]
    labels = [torch.randn(MB, D) for _ in range(M)]

    # ---- GPipe: all forwards, then all backwards --------------------------
    saved = []
    for m in range(M):
        if rank == 0:
            h = stage(data[m])
            dist.send(h.detach(), dst=1)     # send VALUES; the graph stays local
            saved.append((h, None))          # keep h alive: its graph is needed
        else:
            buf = torch.empty(MB, D)
            dist.recv(buf, src=0)
            buf.requires_grad_(True)         # a fresh LEAF — the graph was cut
            out = stage(buf)
            loss = (out - labels[m]).pow(2).sum() / M
            saved.append((loss, buf))

    for m in range(M):
        if rank == 1:
            loss, buf = saved[m]
            loss.backward()                  # fills buf.grad = ∂L/∂h
            dist.send(buf.grad, dst=0)
        else:
            h, _ = saved[m]
            gbuf = torch.empty(MB, D)
            dist.recv(gbuf, src=1)
            torch.autograd.backward(tensors=[h], grad_tensors=[gbuf])   # re-inject
    ...
```

Verified against running the whole model in one process:

```
rank 1: max grad error vs single-process reference = 1.192e-07
rank 0: max grad error vs single-process reference = 1.192e-07
```

fp32 rounding. **The pipeline computes exactly the same gradients as the
undistributed model** — which is the claim topic 11 §3 corollary (a) makes, now
confirmed.

Four details in those lines that are easy to get wrong:

- **`dist.send(h.detach())`.** Sending `h` itself would try to serialize a tensor
  that requires grad. Send values only.
- **`saved.append((h, None))` on stage 0.** `h` must stay referenced, or Python
  frees it and its graph, and the re-injection has nothing to attach to. **This
  is GPipe's `O(M)` memory, in code** (topic 14 §5) — the `saved` list *is* the
  activation stash.
- **`buf.requires_grad_(True)` on stage 1.** Without it, `buf.grad` is `None` and
  there is nothing to send back.
- **Loss scaled by `1/M`** in each micro-batch, since gradients accumulate
  additively.

Turning this into 1F1B is *only* a reordering of the two loops into one
interleaved loop — the mechanism above does not change. That is what a schedule
is.

## 2. Cutting the model: FQNs, deepcopy, prune

The reference implementation does not construct per-stage models. It builds the
**whole** model, then makes a copy per stage and deletes what that stage does not
own.

### 2.1 Specify the split as a list of module names

```python
def generate_llm_fqn_per_model_part(num_stages, num_layers,
                                    input_weight=1, output_weight=1):
    num_effective_layers = num_layers + input_weight + output_weight
    layers_per_stage = num_effective_layers // num_stages
    extra_layers     = num_effective_layers %  num_stages
    ...
```

Fully-qualified names, verified output for `L = 27, P = 4`:

```
stage 0: ['tok_embeddings', 'layers.0' … 'layers.6']            7 layers
stage 1: ['layers.7' … 'layers.13']                             7 layers
stage 2: ['layers.14' … 'layers.20']                            7 layers
stage 3: ['layers.21' … 'layers.26', 'norm', 'output']          6 layers
```

Load-balanced per topic 14 §7, and note `module_fqns_per_model_part` can be
given explicitly in the config to override the heuristic entirely — useful when
you have *measured* per-stage times and know the automatic split is wrong.

### 2.2 Prune a deepcopy

```python
def _build_stage_from_modules(whole_model, pp_mesh, stage_idx, module_names,
                              num_stages, device):
    model = copy.deepcopy(whole_model)          # make a copy so we prune away stuff
    modules_to_keep = set(module_names)

    for module_name, module_value in model.named_children():
        if isinstance(module_value, (nn.ModuleDict, nn.ModuleList)):
            layers_to_keep = {name.split(".", 1)[1] for name in modules_to_keep
                              if name.startswith(f"{module_name}.")}
            if layers_to_keep:
                if isinstance(module_value, nn.ModuleDict):
                    for layer_name in list(module_value.keys()):
                        if layer_name not in layers_to_keep:
                            del module_value[layer_name]          # ← key preserved
                ...
            else:
                setattr(model, module_name, nn.ModuleDict())      # empty, not None
        elif module_name not in modules_to_keep:
            setattr(model, module_name, None)                     # ← the guards fire
```

Three payoffs from topic 05 §2 land here:

- **`del module_value[layer_name]` on a `ModuleDict`** keeps the surviving keys
  at their original names, so stage 1's parameters are still called
  `layers.7.attention.wq`. Checkpoint keys are independent of `P_pp`.
- **`setattr(model, module_name, None)`** is what makes
  `if self.tok_embeddings is not None` in `Transformer.forward` do real work. The
  guards are not defensive style; they are this line's counterpart.
- **`deepcopy` is cheap when the model is on `meta`** — no storage to copy. This
  is why the construction order of topic 13 §2 insists on splitting before
  materializing.

### 2.3 Wrap in `PipelineStage`

```python
stage = PipelineStage(model, stage_idx, num_stages, device,
                      group=pp_mesh.get_group("pp"))
```

`PipelineStage` owns the send/recv buffers and the `torch.autograd.backward`
re-injection from §1 — everything we wrote by hand, plus shape inference and
buffer reuse. `group=pp_mesh.get_group("pp")` is essential: the point-to-point
traffic must go on the **`pp` process group**, not the world group, or a
`P_pp = 4, P_dp = 2` job will send activations to a data-parallel peer.

### 2.4 Placement

```python
style = "v" if schedule_class is ScheduleZBVZeroBubble else "loop"

if style == "v":
    stage_v_pairs = list(zip(range(pp_degree), range(num_stages-1, pp_degree-1, -1)))
    stage_indices = stage_v_pairs[pp_rank]           # rank 0 → (0, 7)
elif style == "loop":
    stage_indices = tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
                                                     # rank 0 → (0, 4)
```

Derived in topic 15 §4 and §5.2. The placement is a function of the schedule,
because the schedule's action ordering assumes a particular assignment.

## 3. Composition order: split, then parallelize

```python
stages, model_parts = pipeline_module_split(model, pp_mesh, schedule, device,
                                            module_names_per_stage)

for i, m in enumerate(model_parts):
    m = parallelize_fn(m, parallel_dims, job_config)    # TP, CP, FSDP, AC
    model_parts[i] = m
    # NOTE: update the model in the stage in case the model is modified
    # e.g. by torch.compile
    stages[i].submod = m
```

**PP first, then the SPMD parallelisms.** The reason is hierarchical: PP cuts the
*module tree*, while TP and FSDP rewrite *parameters within* modules. Applying
TP first would produce DTensors that `deepcopy` and pruning then have to
preserve; applying PP first means each stage is an ordinary `nn.Module` when
`parallelize_fn` sees it, and `parallelize_fn` needs no knowledge of pipelining
at all.

This also explains the mesh design of topic 18: `pp` is the **outermost** mesh
dimension in every view.

`stages[i].submod = m` is easy to overlook and important: `parallelize_fn` may
*return a different object* (notably `torch.compile` returns a wrapper). Without
this line the stage keeps executing the un-parallelized module while the
optimizer updates the parallelized one — two divergent copies of the model, no
error message.

## 4. Training-loop consequences

### 4.1 Only some stages have inputs, only some have losses

```python
has_first_stage = any(stage.is_first for stage in stages)
has_last_stage  = any(stage.is_last  for stage in stages)
```

`any(...)`, not `stage_idx == 0`: with interleaved or V placement a rank can own
both the first and the last stage (ZBV rank 0 owns stages 0 and 7).

```python
targets, losses = (labels, []) if self.pp_has_last_stage else (None, None)
if self.pp_has_first_stage:
    self.pp_schedule.step(inputs, **extra_inputs, target=targets, losses=losses)
else:
    self.pp_schedule.step(**extra_kwargs, target=targets, losses=losses)
```

Note the difference between `extra_inputs` and `extra_kwargs`, and the comment
explaining it:

> For arguments, like attention_masks, we have to put them in a separate dict as
> `extra_inputs` are not forwarded to other stages in PP, but `extra_kwargs` are.

`extra_inputs` are *data* — they enter at the first stage and flow with the
activations. `extra_kwargs` are *configuration* every stage needs
independently. Putting an attention mask in the wrong one means middle stages
compute unmasked attention.

### 4.2 The loss exists on one stage only

```python
if self.pp_has_last_stage:
    loss = (torch.sum(torch.stack(losses)) / global_valid_tokens).to(self.device)
else:
    loss = torch.tensor([-1.0], device=self.device)
```

Non-last stages return a `-1.0` sentinel. It is never reduced into a reported
metric (the loss mesh is `dp × cp`, which excludes `pp` — topic 13 §3), so it is
purely a placeholder that keeps the return type uniform.

### 4.3 The schedule owns the backward pass

`pp_schedule.step()` runs forward **and** backward. Three consequences, all
covered earlier but worth collecting:

- **Gradient normalization moves after the fact** (topic 13 §4.1), legitimate by
  linearity.
- **`scale_grads=False`** so the schedule does not apply its own `1/M`
  (topic 15 §7).
- **`clip_grad_norm_` needs `pp_mesh`** because the global norm spans parameters
  that live on different stages and are not DTensors along `pp` (topic 13 §5).

### 4.4 Every stage needs its own `freqs_cis`

From topic 04 §6.1: `freqs_cis` is a non-persistent buffer, so it does not arrive
via `load_state_dict` and must be regenerated per stage. The context-parallel
setup makes this explicit and is the clearest confirmation:

```python
# ensure CP handles the separate freqs_cis buffer for each pp stage
optional_context_parallel_ctx = dist_utils.create_context_parallel_ctx(
    cp_mesh=parallel_dims.get_mesh("cp"),
    cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
    cp_seq_dims=[1, 1]           + [0 for _ in model_parts],
    cp_no_restore_buffers={inputs, labels},
    ...)
```

One `freqs_cis` per model part, each needing its own sequence-dimension shard
(`0`, since `freqs_cis` is `[S, d/2]` while `inputs`/`labels` are `[B, S]` with
the sequence on dim `1`). If you build a pipeline by hand and forget to call
`init_weights` on a stage, you get a `meta` tensor error — or, worse, zeros.

## 5. Debugging pipeline parallelism

PP fails in characteristic ways. The four you will actually meet:

**Hang at the first step.** A send with no matching recv. Causes: a stage
computing the wrong peer rank; mismatched shapes (the recv buffer must be
pre-sized correctly); using the world group instead of the `pp` group; a schedule
whose action lists are not mirror-images across ranks. Get a stack trace from
every rank (`py-spy dump --pid` per process, or `TORCH_DISTRIBUTED_DEBUG=DETAIL`)
and look for the rank that is *not* in a collective.

**Loss is `-1`.** You are logging a non-last stage's sentinel. Reduce over the
`loss` mesh, not the world.

**Gradients are `None` on the first stage.** The re-injection is missing or the
received activation was not marked `requires_grad_(True)`. In the hand-written
version this is §1.3's third bullet.

**Loss curve is subtly worse than the single-GPU baseline.** Almost always
normalization: `scale_grads` double-counting, or the post-hoc division applied
to only the last stage's parameters.

And the test that catches nearly all of it, which is exactly what §1.3 does:

> **Run the same model and data with `P_pp = 1` and with `P_pp = 2` for a handful
> of steps and compare gradients.** They must agree to floating-point noise. If
> they do not, no amount of loss-curve staring will tell you why.

## 6. Check yourself

1. Why must a pipeline stage send `h.detach()` rather than `h`?
2. What does `torch.autograd.backward(tensors=[h], grad_tensors=[g])` do that
   `h.backward()` cannot?
3. In the 40-line pipeline, why must stage 0 keep `h` in a list? Which cost from
   topic 14 is that list?
4. What happens if stage 1 omits `buf.requires_grad_(True)`?
5. Why does pruning use `del module_dict[key]` rather than rebuilding the
   container? What breaks with a `ModuleList`?
6. Why is `deepcopy` of the whole model acceptable here?
7. Why must PP be applied *before* TP and FSDP? What does that imply about the
   mesh dimension order?
8. What is `stages[i].submod = m` for, and what is the symptom if you delete it?
9. Why `any(stage.is_first ...)` instead of `pp_rank == 0`?
10. Distinguish `extra_inputs` from `extra_kwargs` and give a concrete bug from
    swapping them.
11. Why does each model part need its own `freqs_cis`, and why is its
    `cp_seq_dim` `0` while `inputs`' is `1`?
12. Your run hangs on step 1 with `P_pp = 4`. List four candidate causes and the
    first diagnostic you would run.

## Next

→ [18 — Device meshes and combining PP with DP](18-device-meshes-pp-dp.md)
