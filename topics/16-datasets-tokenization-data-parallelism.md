# 16 — Datasets, Tokenization and Data Parallelism

> Video: 07:10:29 — Datasets, Tokenization and Data Parallelism
> Code: `torchfeather/datasets/hf_datasets.py`, `torchfeather/components/dataloader.py`,
> `torchfeather/components/tokenizer.py`

## Why this exists

The data pipeline is the part of a distributed training system that most
tutorials wave at and most real runs get wrong. Three things must hold, and each
is a distributed-systems requirement rather than a data-engineering one:

1. **Disjointness.** No two data-parallel ranks may see the same token, or you
   are silently training on duplicated data.
2. **Replication where required.** Ranks that are *not* data-parallel peers —
   TP ranks, CP ranks, PP stages — must see **exactly the same** batch, or the
   parallelism invariants break.
3. **Resumability.** A 30-day run will restart. It must resume at the right
   place in the stream, per rank.

Requirements 1 and 2 pull in opposite directions, and getting them right is the
whole content of §3.

## 1. Tokenization and the packing decision

### 1.1 The loop

```python
def __iter__(self):
    max_buffer_token_len = 1 + self.seq_len

    while True:
        for sample in self._get_data_iter():
            sample_text = self._text_processor(sample)
            sample_tokens = self._tokenizer.encode(sample_text, add_bos=True, add_eos=True)
            self._token_buffer.extend(sample_tokens)
            self._sample_idx += 1

            while len(self._token_buffer) >= max_buffer_token_len:
                x = torch.LongTensor(self._token_buffer[:max_buffer_token_len])
                self._token_buffer = self._token_buffer[max_buffer_token_len:]
                yield {"input": x[:-1]}, x[1:]
```

Read what this does: documents are tokenized, appended to a running buffer, and
the buffer is sliced into fixed `seq_len` chunks. This is **document packing**,
and every design decision in it is deliberate.

### 1.2 Why pack rather than pad

The alternative is one document per sequence, padded to `seq_len`. Consider
FineWeb, whose documents average roughly 500–1000 tokens against `seq_len =
16384`. Padding would waste **90%+ of every batch** — you would be paying full
`O(S²)` attention cost to process mostly `<pad>`.

Packing wastes nothing. The costs it accepts:

- **Cross-document attention.** A packed sequence contains several unrelated
  documents, and causal attention lets tokens of document 2 attend to document 1.
  Strictly this is wrong. In practice, at pretraining scale, it is tolerated
  universally: the `<eos>`/`<bos>` boundary tokens give the model a learnable
  signal, and it learns to ignore what precedes them. The rigorous fix is a
  block-diagonal document mask — which is why the trainer threads
  `extra_kwargs` for things like `attention_masks` — but it costs kernel support
  and is usually judged not worth it for pretraining.
- **Documents split across batches.** A document straddling a chunk boundary is
  truncated, and its tail appears at the start of the next sequence with no
  context. At scale this affects a small fraction of tokens.

### 1.3 The `1 + seq_len` slice, and the off-by-one that matters

```python
max_buffer_token_len = 1 + self.seq_len
x = torch.LongTensor(self._token_buffer[: max_buffer_token_len])
input = x[:-1]      # tokens 0 … S−1
label = x[1:]       # tokens 1 … S
```

Next-token prediction needs `S+1` tokens to produce `S` (input, label) pairs.
Taking `S` tokens and shifting would waste one position and — worse — leave the
last input token with no label. Slice `S+1`, then shift. Simple, and a
frequently-botched detail.

Note also that **labels are not offset inside the model.** The model's `forward`
returns logits for positions `0…S−1`, and the loss compares them against
`label`. Some codebases shift inside the model instead; doing both, or neither,
is a bug that shows up as a model that predicts the *current* token perfectly
and generates gibberish.

### 1.4 Streaming

```python
return load_dataset(dataset_path, name="default", split=split,
                    streaming=True,
                    download_config=DownloadConfig(max_retries=40))
```

`streaming=True` is not an optimization, it is a necessity: FineWeb is ~15 TB and
will not be downloaded to every node. It also means the dataset is an
`IterableDataset` — no `len()`, no random access, no `sampler`. Everything in §3
follows from that constraint.

`max_retries=40` and the 300-second timeouts are the practical reality of
streaming from a remote store for 30 days: transient HTTP failures are certain,
and a data-loader exception kills the whole run.

## 2. Data parallelism revisited: what the dataloader must guarantee

From topic 12, DP is correct because

```
∂L/∂θ = (1/P) Σ_p [ the gradient of rank p's batch ]
```

and this is the gradient of the *global* batch **only if the per-rank batches
partition the global batch**. Two failure modes:

- **Overlap** (ranks see the same data) → you are averaging duplicate gradients.
  The effective batch is smaller than you think and the data is seen twice per
  "epoch".
- **Gaps** (some data never sampled) → less harmful, but you are not training on
  the dataset you think you are.

So: **exact partition, no overlap, no gaps.**

## 3. Sharding on the right mesh dimension

This is the section that matters.

```python
self._data = split_dataset_by_node(ds, dp_rank, dp_world_size)
```

`split_dataset_by_node` assigns shards round-robin over `dp_world_size`. The
entire question is: **what are `dp_rank` and `dp_world_size`?**

The answer is **not** `global_rank` and `world_size`. From topic 13 §2:

```python
if parallel_dims.dp_enabled:
    batch_mesh = parallel_dims.get_mesh("batch")
    dp_degree, dp_rank = batch_mesh.size(), batch_mesh.get_local_rank()
else:
    dp_degree, dp_rank = 1, 0
```

and from topic 18's mesh construction, `batch = dp_replicate × dp_shard`.

### 3.1 Which dimensions want different data, and which want identical data

| Mesh dim | Different data? | Why |
|----------|-----------------|-----|
| `dp_replicate` | **different** | that is the definition of data parallelism |
| `dp_shard` (FSDP) | **different** | FSDP shards *parameters*, not the batch; each rank still processes its own examples |
| `tp` | **identical** | TP ranks cooperate on one logical forward pass over one batch. Different data means they are computing partial results of *different* matmuls, and the all-reduce sums nonsense. |
| `cp` | **identical batch, different slice** | each CP rank loads the *full* sequence and CP splits it internally (topic 23) — hence topic 13's `local_valid_tokens //= cp` |
| `pp` | **identical** | stages process the same micro-batch at different depths |

So the batch is sharded over `dp_replicate × dp_shard` and replicated over
`tp × cp × pp`. Which is exactly what the `"batch"` mesh dimension is.

### 3.2 The concrete failure

`world_size = 32`, `P_tp = 8`, `P_dp = 4`. Correct behaviour:

```
dp_degree = 4,  dp_rank = global_rank // 8      (mesh-derived)
    ranks 0-7  → data shard 0
    ranks 8-15 → data shard 1
    ranks 16-23→ data shard 2
    ranks 24-31→ data shard 3
```

Using `global_rank`/`world_size` instead gives 32 different shards, so the 8 TP
ranks of replica 0 each get *different* tokens. What happens then:

- The forward pass is garbage — a column-parallel matmul on rank 3's data plus a
  row-parallel matmul on rank 4's data is not a matmul of anything.
- **It does not crash.** Shapes are all correct. The all-reduce succeeds. The
  loss is a number, and it even decreases somewhat, because rank 0 alone is
  learning something.
- The effective global batch is `32×` larger than configured, and the model is
  much worse than it should be.

This is the single most consequential line in the data pipeline. The mesh is the
source of truth for who is a data-parallel peer, and nothing else is.

### 3.3 The other half: TP ranks must agree *bit for bit*

Sharding correctly is necessary but not sufficient — the replicated ranks must
also produce *identical* batches. Two things can break that:

- **Independent dataloader workers with different seeds.** If shuffling or
  augmentation is randomized per worker, TP ranks diverge.
- **Non-determinism in the streaming iterator** (e.g. shard files arriving in
  different orders).

The clean approach — and what the mesh-based sharding gives you — is that ranks
sharing a `batch` coordinate run the *same deterministic iterator over the same
shard*, so they agree by construction rather than by coordination. If you ever
need to check, the cheapest assertion in distributed training is:

```python
h = torch.tensor([hash(inputs.cpu().numpy().tobytes()) % 2**31], device=dev)
g = [torch.zeros_like(h) for _ in range(tp_mesh.size())]
dist.all_gather(g, h, group=tp_mesh.get_group())
assert all(x.item() == g[0].item() for x in g), "TP ranks disagree on the batch!"
```

Run it once at startup on any new configuration. It costs microseconds and
catches the §3.2 bug immediately.

## 4. Resumability

A 30-day run restarts, and "restart from the beginning of the data" is not
acceptable — it re-trains on seen tokens and corrupts the data mixture.

### 4.1 What state has to be saved

```python
class HuggingFaceDataset(IterableDataset, Stateful):
    def __init__(self, ...):
        self._sample_idx: int = 0
        self._token_buffer: list[int] = []
```

Two pieces, and both are needed:

- **`_sample_idx`** — how many documents this rank has consumed.
- **`_token_buffer`** — the *partial document* left over from the last yielded
  sequence.

Dropping the buffer would look harmless and would silently discard up to
`seq_len` tokens per restart while shifting every subsequent sequence boundary.
The buffer is genuine state, not a cache.

`Stateful` is the same interface `Trainer` implements (topic 13 §2.1), so the
dataloader position is checkpointed atomically with the weights and the optimizer
— all or nothing. Getting a checkpoint whose weights are from step 10,000 and
whose data position is from step 9,000 is worse than either.

### 4.2 Resuming a stream you cannot seek

```python
def _get_data_iter(self):
    # For map-style datasets, resume by skipping to the correct index
    # For iterable-style datasets, the underlying iterator already points to
    # the correct index
    if isinstance(self._data, Dataset):
        if self._sample_idx == len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))
    return iter(self._data)
```

Map-style datasets support `.skip()`. Streaming datasets do not seek, so the
underlying HuggingFace iterator carries its own resumable position and
`_sample_idx` is used only for bookkeeping. Two different mechanisms behind one
interface — worth knowing which one you are on, because `.skip()` on a huge
map-style dataset is `O(sample_idx)` and can take minutes at step 100,000.

### 4.3 Per-rank state, stored once

```python
def state_dict(self):
    # Store state only for dp rank to avoid replicating the same state across
    # other dimensions.
    return {self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self.dp_world_size}
```

Each rank stores under the key `f"dp_rank_{dp_rank}"`. Two consequences:

- **No duplication.** The 8 TP ranks of a replica share a `dp_rank`, so they
  write the same key — one copy, not eight.
- **Resharding is refused, loudly:**

```python
assert self.dp_world_size == state_dict["world_size"], (
    "dp_degree is inconsistent before and after checkpoint, "
    "dataloader resharding is not supported yet.")
```

Change `P_dp` and the assertion fires. This is the right behaviour: the model
weights *can* be resharded (topic 24) but the data stream cannot — rank 3 of 4
and rank 3 of 8 are reading different shards of the dataset, and there is no
sound way to map one position onto the other. An explicit failure beats a silent
change of data distribution.

## 5. The full path of a token

```
FineWeb (streaming, ~15 TB, remote)
   │
   ├─ split_dataset_by_node(ds, dp_rank, dp_world_size)     ← the "batch" mesh dim
   │      rank's disjoint document shard
   │
   ├─ tokenizer.encode(text, add_bos=True, add_eos=True)
   │
   ├─ _token_buffer.extend(...)                              packing
   │
   ├─ slice [0 : S+1]  →  input = x[:-1], label = x[1:]      next-token shift
   │
   ├─ ParallelAwareDataloader  (batch_size = local_batch_size)
   │
   ├─ train_step phase 1: count (labels != IGNORE_INDEX)     topic 13 §3
   │
   ├─ [CP] each rank loads the FULL sequence; CP splits it   topic 23
   │
   └─ [PP] only the first stage receives `input`;
           only the last stage receives `labels`             topic 13 §9
```

Note the last two lines. Under CP, all `P_cp` ranks pull the same sequence and
the context-parallel context manager shards it in place — which is why token
counting divides by `cp`. Under PP, the trainer conditions on stage position:

```python
targets, losses = (labels, []) if self.pp_has_last_stage else (None, None)
if self.pp_has_first_stage:
    self.pp_schedule.step(inputs, ..., target=targets, losses=losses)
else:
    self.pp_schedule.step(..., target=targets, losses=losses)
```

The dataloader hands every rank the same thing; the trainer decides what each
stage actually consumes.

## 6. Check yourself

1. Give the three requirements a distributed data pipeline must satisfy, and say
   which two are in tension.
2. Why pack documents instead of padding? Quantify the waste for
   `seq_len = 16384` and 750-token documents.
3. Name the two costs of packing, and the rigorous fix for the first.
4. Why is the buffer sliced at `1 + seq_len` rather than `seq_len`?
5. For each mesh dimension (`dp_replicate, dp_shard, tp, cp, pp`), say whether
   its ranks should see the same or different data, and why.
6. `world_size = 32, P_tp = 8, P_dp = 4`. What does correct sharding assign, and
   what exactly goes wrong if you shard by `global_rank`? Why doesn't it crash?
7. Why does FSDP's `dp_shard` dimension want *different* data even though it is
   about parameters?
8. What two pieces of dataset state must be checkpointed? What breaks if you
   drop the second?
9. Why does the dataloader `state_dict` key on `dp_rank` rather than
   `global_rank`?
10. Why is dataloader resharding refused when model resharding is supported?
11. Under CP, why does the trainer divide `local_valid_tokens` by `cp`?

## Next

→ [17 — Coding pipeline parallelism](17-coding-pipeline-parallelism.md)
