# 29 — All-to-All Token Dispatch and Combine

> Video: 19:01:56 — All-to-All Token Dispatch and Combine
> Code: `torchfeather/distributed/expert_parallel.py` (`_token_dispatch`, `_token_combine`),
> `torchfeather/model/moe/utils.py` (`_permute`, `_unpermute`)

## Why this exists

Topic 28 established *that* tokens must travel and *that* the pattern is an
all-to-all. This topic is the implementation, and it is the most intricate code
in the framework — not because all-to-all is hard, but because **the message
sizes are data-dependent**.

Every collective so far had shapes known before the step began. Here the number
of tokens rank `r` sends to rank `s` depends on what the router decided about
this particular batch. That single fact forces:

- **two** all-to-alls instead of one (exchange the counts, then the data),
- a **device-to-host synchronization** (the splits must reach the CPU),
- a **permutation and padding** step (the received layout is not the layout the
  grouped matmul needs).

We work through the reference implementation's own example throughout:
`P_ep = 4`, `E = 12` (3 experts per rank), `k = 4`, 32 routed rows per rank. All
numbers below are verified against the code.

## 1. Why two all-to-alls

`dist.all_to_all_single(out, inp, output_splits, input_splits)` requires the
**receiver** to know how many rows are coming from each sender, because it must
size the output buffer. Under dropless MoE (topic 26 §8) that count is not known
in advance:

```python
# NOTE: We are not using fixed-capacity padded all-to-all, that's why we need to
# do two all-to-all, the first one to figure out how many tokens each rank will
# receive, the second one to actually send/receive them.
```

The alternative — the classic GShard design — fixes a capacity `C` per expert,
pads every message to `C`, and drops the overflow. That gives static shapes and
one all-to-all, at the cost of dropped tokens and wasted bandwidth on padding.
This implementation pays an extra tiny collective instead. The right trade:
the metadata all-to-all moves `E` integers.

## 2. Exchange the counts

```python
num_tokens_per_expert_group = all_to_all_single(
    num_tokens_per_expert, None, None, group=device_mesh.get_group())

# Need to wait explicitly because it is used by a triton kernel later which
# doesn't realize that AsyncCollectiveTensor needs unwrapping
num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
    num_tokens_per_expert_group)
```

An **equal-split** all-to-all (`None, None` splits): each rank's `[E]` count
vector is cut into `P_ep` chunks of `E/P_ep`, and chunk `d` goes to rank `d` —
which is exactly the rank that owns those experts.

```
rank 0 num_tokens_per_expert       [2,3,3, 2,3,3, 3,4,3, 2,2,2]
                                    └E0-2┘ └E3-5┘ └E6-8┘ └E9-11┘
                                      ↓      ↓      ↓      ↓
                                    rank0  rank1  rank2  rank3
```

After the exchange, rank 0 holds the counts for **its** experts (E0–E2) **from
every source**:

```
rank 0 num_tokens_per_expert_group = [2,3,3 | 2,2,2 | 3,2,2 | 3,4,3]
                                      src 0   src 1   src 2   src 3
```

The explicit `wait_tensor` is a real trap worth noting. The functional
collectives return an `AsyncCollectiveTensor` that unwraps lazily on first
*PyTorch* use — but the value is consumed by a **Triton kernel**
(`generate_permute_indices`), which reads raw memory and does not trigger the
unwrap. Without the explicit wait you read uninitialized data: intermittently
wrong routing, no error. **Any hand-written or custom kernel consuming a
collective's output needs an explicit wait.**

## 3. Compute the splits

```python
input_splits = (num_tokens_per_expert.view(ep_degree, -1).sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True))
output_splits = (num_tokens_per_expert_group.view(ep_degree, -1).sum(dim=1)
                 .to(torch.device("cpu"), non_blocking=False))
self.input_splits  = input_splits.tolist()
self.output_splits = output_splits.tolist()
```

Both are "reshape into `[P_ep, E/P_ep]` and sum over the local-expert axis" —
collapsing per-expert counts into per-*rank* counts. Verified:

```
input_splits  (what rank 0 SENDS to each rank):
  [2,3,3, 2,3,3, 3,4,3, 2,2,2] → view(4,-1) → sum(1) → [8, 8, 10, 6]
  Σ = 32 = (local_batch / tp_degree) · top_k = (16/2) · 4   ✓

output_splits (what rank 0 RECEIVES from each rank):
  [2,3,3 | 2,2,2 | 3,2,2 | 3,4,3] → view(4,-1) → sum(1) → [8, 6, 7, 10]
  Σ = 31

other ranks receive 29, 35, 33.
```

Note the last line, and the code's own comment on it:

> NOTE: as you can see, each rank receives a (potentially) different number of
> tokens, that's why load balancing is important

31, 29, 35, 33 — a 21% spread even in this toy example. Under dropless EP the
step runs at the 35-row rank's speed (topic 28 §4). This is the concrete form of
the load-balancing argument.

### 3.1 The device-to-host sync

```python
# NOTE: this would incur a device-to-host sync
```

`all_to_all_single`'s split arguments must be **Python lists** — the host has to
know the message sizes to post the sends and receives. So the counts, computed on
GPU by `histc`, must come back to the CPU. `.tolist()` blocks until the GPU
reaches that point.

Notice the asymmetry in the two transfers:

```
input_splits :  non_blocking=True     ← can be issued early, not needed yet
output_splits:  non_blocking=False    ← must be complete; forces the sync here
```

`input_splits` derives from `num_tokens_per_expert`, already available; its copy
is fired off asynchronously. `output_splits` derives from the metadata
all-to-all's result, so this is where the pipeline actually stalls — and the code
places the blocking copy last, giving the async one maximum time to land.

This one sync has consequences well outside MoE, all traced earlier:

- **It breaks FSDP's prefetch order inference** (topic 20 §7) → hence
  `set_modules_to_forward_prefetch` when `ep_degree > 1`.
- **It stalls the CPU**, so the CPU falls behind in issuing GPU work. With enough
  MoE layers the whole step becomes CPU-launch-bound.
- **It is why `torch.compile` uses `fullgraph=False` for MoE blocks**
  (topic 20's `apply_compile`): a data-dependent host value is a graph break by
  construction.

```python
fullgraph = True
if transformer_block.moe_enabled:
    fullgraph = False
```

Everything above is downstream of one line: message sizes depend on data.

## 4. The dispatch

```python
routed_input = all_to_all_single_autograd(
    routed_input,          # (T/tp · k, dim)
    self.output_splits,    # what each rank needs to RECEIVE
    self.input_splits,     # what each rank needs to SEND
    device_mesh.get_group())
```

Rank 0 sends 32 rows and receives 31 — the shape genuinely changes.

**`all_to_all_single_autograd`**, not the plain version. This is topic 11 §2.5
made concrete: all-to-all is its own adjoint with split and concat exchanged, so
the differentiable wrapper's backward is another all-to-all with `input_splits`
and `output_splits` swapped. Which is exactly what the *combine* does in the
forward direction — the dispatch's backward and the combine's forward are the
same operation. **One implementation, four uses.**

## 5. The layout problem, and `_permute`

After the dispatch, rank 0's 31 rows are ordered **source-major**:

```
x.shape = (31, dim)
source 0: [E0: A0,A5  | E1: A0,A1,A6  | E2: A1,A2,A7 ]     8 rows
source 1: [E0: A11,A14| E1: A12,A15   | E2: A8,A13   ]     6 rows
source 2: [E0: B1,B4,B7| E1: B2,B5    | E2: B3,B6    ]     7 rows
source 3: [E0: B10,B13,B14 | E1: B8,B11,B14,B15 | E2: B9,B12,B15]  10 rows
```

But `torch._grouped_mm` needs **expert-major** rows: all of E0's rows
contiguous, then all of E1's, then E2's (topic 26 §7.3). The received layout
interleaves them.

```python
# num_tokens_per_expert_group = [2,3,3 | 2,2,2 | 3,2,2 | 3,4,3]
# what we want is:  [ 2,  3,  3] +
#                   [ 2,  2,  2] +
#                   [ 3,  2,  2] +
#                   [ 3,  4,  3] =
#                   [10, 11, 10]  <- how many tokens to feed E0, E1, E2
```

Verified: summing the `[4, 3]` view over the *source* axis gives `[10, 11, 10]`.
So `_permute` must gather 31 scattered rows into 3 contiguous expert groups.

### 5.1 Padding, and the sentinel row

```python
# Reserve enough space for every real row plus up to one alignment block per
# local expert — a conservative upper bound.
x_padded_per_expert = x.shape[0] + num_local_experts * TOKEN_GROUP_ALIGN_SIZE_M
padded_max_len = _round_up(x_padded_per_expert, TOKEN_GROUP_ALIGN_SIZE_M)
```

With `TOKEN_GROUP_ALIGN_SIZE_M = 8`: `31 + 3·8 = 55`, rounded up to **56**.

Why pad at all? The grouped matmul wants each expert's row count to be a multiple
of the tile size — a `[10, dim]` matmul on a kernel with `M`-tile 8 wastes most of
a tile and may not be supported at all. So each expert's group is padded to a
multiple of 8:

```
E0: 10 rows → 16      E1: 11 rows → 16      E2: 10 rows → 16
offsets = [16, 32, 48]        grouped MM consumes rows [0:48]
```

The padding rows must contain **zeros** so they contribute nothing. The trick:

```python
# Append one all-zero row with index 31. Negative index -1 in permuted_indices
# selects this padding row.
x = torch.vstack((x, x.new_zeros(x.shape[-1])))     # (31, dim) → (32, dim)
input_shape = x.shape
x = x[permuted_indices, :]                          # (56, dim)
```

One zero row appended, and `permuted_indices` uses `-1` — which Python/PyTorch
resolves to the last row — wherever padding is needed:

```
permuted_indices =
    E0 region: [0,1,8,9,14,15,16,21,22,23, -1,-1,-1,-1,-1,-1]
    E1 region: [2,3,4,10,11,17,18,24,25,26,27, -1,-1,-1,-1,-1]
    E2 region: [5,6,7,12,13,19,20,28,29,30, -1,-1,-1,-1,-1,-1]
    unused:    [-1,-1,-1,-1,-1,-1,-1,-1]
```

**The permutation and the padding are one `index_select`.** No separate
zero-filling pass, no branching. `input_shape` is saved so `_unpermute` can
reconstruct the 31 real rows and then drop the zero row.

`permuted_indices` is produced by `generate_permute_indices`, a Triton kernel —
which is the consumer that needed the explicit `wait_tensor` in §2.

## 6. The combine

```python
def _token_combine(self, _mod, routed_output, device_mesh):
    routed_output = _unpermute(routed_output, self.input_shape, self.permuted_indices)
    routed_output = all_to_all_single_autograd(
        routed_output,
        self.input_splits,     # ← SWAPPED
        self.output_splits,    # ← SWAPPED
        device_mesh.get_group())
    return routed_output
```

Two steps, both exact inverses:

- **`_unpermute`** scatters the 56 padded expert-major rows back to 31
  source-major rows, using the saved `permuted_indices` and `input_shape`, then
  drops the zero row.
- **The all-to-all with `input_splits` and `output_splits` exchanged** sends each
  row back to the rank it came from.

That is the adjoint from topic 11 §2.5, written out. Compare the two calls:

```
dispatch: all_to_all_single_autograd(x, output_splits, input_splits, group)
combine:  all_to_all_single_autograd(x, input_splits,  output_splits, group)
                                        └──────── swapped ────────┘
```

**Two arguments exchanged.** And because both use the `_autograd` wrapper, the
backward passes are also correct without further work: the dispatch's backward is
the combine's forward and vice versa. Four operations, one kernel, no extra code.

This is the payoff of topic 11's adjoint theorem. Had we memorized "expert
parallelism uses all-to-all twice" we would still have had to derive the
backward. Deriving it from the adjoint makes both directions free.

## 7. Cost, and overlap

Per MoE layer, per micro-batch:

```
metadata all-to-all:  E integers            ~ negligible
dispatch:             T · k · d · 2 bytes   × (P−1)/P
combine:              T · k · d · 2 bytes   × (P−1)/P
```

For `T = B·S = 16384`, `k = 8`, `d = 2048`, bf16, `P_ep = 8`:

```
per direction: 16384 · 8 · 2048 · 2 · 7/8 = 470 MB
per layer (dispatch + combine):            940 MB
forward + backward:                       1.9 GB
× 26 MoE layers:                           49 GB per micro-batch
```

Substantial — note the `k` factor, which is why fine granularity (topic 26 §6)
costs bandwidth. Three mitigations, all present in the code:

**(a) The shared expert fills the gap.** Topic 26 §5:

```python
# NOTE: we execute the shared expert before scoring the output of the routed
# expert to "implicitly" overlap the shared expert compute with token combine
# communication
if self.shared_experts is not None:
    out = self.shared_experts(x)
```

The shared expert needs no routing and no communication, so it is placed exactly
where the combine is in flight. Not a stream annotation — just **ordering
independent work to sit where the communication is**.

**(b) All-to-all is in the selective-AC save list.** Topic 20 §10:

```python
torch.ops._c10d_functional.all_to_all_single.default,
```

Recomputing a collective during the backward pass would re-run the
communication *and* require every rank to reach it in the same order
(topic 19 §5.1). Saving the output avoids both.

**(c) `k` is a lever.** Halving `k` halves this traffic. It is the direct
communication cost of granularity, and the reason fine-grained MoE needs a fast
`ep` link.

## 8. The full path of a token, end to end

```
x  [T, d]                                       local, sequence-parallel

router  →  scores, top-k, per-expert counts     fp32, local           (t26 §2)
reorderer: argsort by expert, // top_k          local                 (t26 §7.1)
gather  →  routed_input [T·k, d]                sorted by GLOBAL expert
scale by gate (score_before_experts)                                  (t26 §7.4)

┌─ ExpertParallel._token_dispatch ────────────────────────────────────┐
│ all_to_all_single(counts)             metadata            §2        │
│ wait_tensor                           Triton needs it     §2        │
│ input_splits/output_splits → CPU      D2H SYNC            §3        │
│ all_to_all_single_autograd(rows)      the dispatch        §4        │
│ _permute                              source→expert-major §5        │
│                                       + pad to mult of 8            │
└─────────────────────────────────────────────────────────────────────┘

torch._grouped_mm(w1) → silu → × grouped_mm(w3) → grouped_mm(w2)      (t26 §7.3)
   on LOCAL expert shards                                             (t27 §1.1)

   [shared_experts(x) runs here, overlapping the combine]             (t26 §5)

┌─ ExpertParallel._token_combine ─────────────────────────────────────┐
│ _unpermute                            expert-major→source §6        │
│ all_to_all_single_autograd(swapped)   the combine         §6        │
└─────────────────────────────────────────────────────────────────────┘

scale by gate (if not score_before_experts)
scatter_add into out (which already holds the shared expert's output)  (t26 §7.2)
out [T, d]  Partial → reduce-scatter → Shard(1)                        (t27 §4)
```

Every arrow is either a local kernel or one of the two all-to-alls. Nothing else
is needed, and every step has appeared in an earlier topic.

## 9. Check yourself

1. Why two all-to-alls instead of one? What is the alternative design, and what
   does it cost?
2. The metadata all-to-all uses equal splits (`None, None`). Why is that correct?
3. Trace rank 0's `num_tokens_per_expert` → `input_splits` and
   `num_tokens_per_expert_group` → `output_splits` on the §3 example.
4. Ranks receive 31, 29, 35, 33 rows. What is the step time relative to perfect
   balance, and which mechanism fixes it?
5. Why is `wait_tensor` needed explicitly? What is the failure mode without it?
6. Why must the splits reach the CPU? Why is one copy `non_blocking=True` and the
   other `False`?
7. Name three subsystems affected by that one D2H sync.
8. Why does `apply_compile` use `fullgraph=False` for MoE blocks?
9. After the dispatch, rows are source-major. Why is that wrong for
   `_grouped_mm`, and what does summing the `[P_ep, E/P_ep]` view over the source
   axis give?
10. Derive `padded_max_len = 56` from 31 rows, 3 local experts, align 8.
11. Explain the `-1` sentinel trick. What does it save?
12. Write the dispatch and combine calls side by side. What is the difference, and
    which theorem makes it correct?
13. Compute all-to-all traffic for `T=8192, k=4, d=4096, P_ep=16, L_moe=32`. What
    happens if you double `k`?
14. Where is the shared expert executed and why exactly there?
15. Why is `all_to_all_single` in the selective-AC save list? Give both reasons.

## Next

→ [30 — Expert tensor parallelism](30-expert-tensor-parallelism.md)
