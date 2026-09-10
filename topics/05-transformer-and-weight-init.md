# 05 — Building the Transformer and Weight Initialization

> Video: 01:50:16 — Building the Transformer and Weight Initialization
> Code: `torchfeather/model/model.py` (`TransformerBlock`, `DeepSeekV3Model`)

## Why this exists

Two things happen in this topic, and only one of them is about transformers.

The first is the ordinary business of assembling blocks and initializing
weights. The interesting part is that **initialization is a variance-propagation
problem**, and the standard `std = 0.02` you see everywhere is not a magic
number — it is the solution to a specific equation about the residual stream.

The second is structural, and it is why this topic is in Part I rather than
being skipped: **the model class is written to be cut into pieces.** Every
design choice in `DeepSeekV3Model` — `ModuleDict` instead of `ModuleList`, the
`is not None` guards, the separated `init_weights` — exists so that pipeline
parallelism (topic 17) can delete two thirds of the model on each rank without
the remaining third noticing. If you write your model the obvious way, you will
rewrite it when you add PP. So we write it the right way now.

## 1. The block

```python
def forward(self, x, freqs_cis):
    x = x + self.attention(self.attention_norm(x), freqs_cis)
    if self.moe_enabled:
        x = x + self.moe(self.ffn_norm(x))
    else:
        x = x + self.feed_forward(self.ffn_norm(x))
    return x
```

Two sublayers, each `x ← x + F(Norm(x))`. Three decisions are encoded here.

### 1.1 Pre-norm, and why it is not a style choice

```
post-norm (original 2017):   x_{ℓ+1} = Norm( x_ℓ + F(x_ℓ) )
pre-norm  (everything since): x_{ℓ+1} = x_ℓ + F( Norm(x_ℓ) )
```

Differentiate the pre-norm recurrence:

```
∂x_{ℓ+1}/∂x_ℓ = I + J_ℓ        where J_ℓ = ∂F(Norm(x_ℓ))/∂x_ℓ
```

The gradient reaching layer `0` from the loss is a product of these:

```
∂L/∂x_0 = ∂L/∂x_L · Π_{ℓ} (I + J_ℓ) = ∂L/∂x_L · ( I + Σ_ℓ J_ℓ + higher order )
```

The leading term is the **identity**. There is a path from the loss to every
layer's input that passes through no multiplicative factor at all. Gradients
cannot vanish geometrically with depth.

For post-norm, each step applies the normalization's Jacobian, which contains a
`1/‖x‖` factor. The product of `L` such Jacobians attenuates, which is precisely
why the original transformer needed learning-rate warmup to train at all, and
why post-norm models get harder to train as `L` grows. Pre-norm is the reason
you can stack 100 blocks.

The cost of pre-norm: the residual stream is never normalized, so its magnitude
grows with depth. Hence the final `self.norm` before the output head, and hence
the depth-dependent initialization in section 3.

### 1.2 RMSNorm, and what was dropped

```
                       x_i                          ┌     1  d      ┐
RMSNorm(x)_i = g_i · ───────── ,   RMS(x) = sqrt │  ─  Σ  x_j²  │ + ε
                      RMS(x)                        └     d j=1     ┘
```

LayerNorm additionally subtracts the mean and adds a bias:
`(x − μ)/σ · g + b`. RMSNorm drops both.

Why dropping mean-centering is safe: `μ` is **one** degree of freedom out of
`d = 2048`. Removing it changes the vector by a rank-one projection, and the
very next operation is a learned linear map that can represent that projection
itself if it turns out to be useful. Empirically the quality is identical, and
you save a reduction pass and a subtraction. Dropping the bias is safe for the
same reason and is standard across LLMs (which is also why every `nn.Linear` in
this codebase has `bias=False`).

The `ε` is inside the square root and prevents division by zero for an all-zero
input. One inconsistency worth noticing in the reference code:

```python
self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)  # 1e-5
self.ffn_norm       = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)  # 1e-5
...
self.norm = nn.RMSNorm(model_args.dim)     # ← no eps: falls back to the default
```

The final norm uses PyTorch's default rather than `norm_eps`. It is harmless
(the default is smaller, and the final hidden state is never near zero) but it
is the kind of drift worth catching in review.

### 1.3 `moe_enabled` decided at construction

```python
self.moe_enabled = layer_id >= model_args.n_dense_layers
```

The first `n_dense_layers` blocks use a dense SwiGLU FFN; the rest use MoE. This
is a real architectural finding, not a convenience: the earliest layers do
low-level, near-universal feature extraction that every token needs, so routing
them to specialists wastes capacity and destabilizes the router early in
training (DeepSeek-V3 uses 3 dense layers). Note the branch is resolved *once*,
at `__init__` — so different blocks are genuinely different module types, which
matters when you write a parallelization plan that must skip MoE layers or treat
them differently (topics 27–30).

## 2. The model, written to be cut

```python
class DeepSeekV3Model(nn.Module):
    def __init__(self, model_args):
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        self.register_buffer("freqs_cis", precompute_freqs_cis(model_args),
                             persistent=False)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
        self.norm = nn.RMSNorm(model_args.dim)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def forward(self, tokens):
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        for layer in self.layers.values():
            h = layer(h, self.freqs_cis)
        h = self.norm(h) if self.norm is not None else h
        return self.output(h) if self.output is not None else h
```

Read that `forward` again. It looks like defensive programming. It is not — it is
the **pipeline-parallel contract**, and there are three parts to it.

### 2.1 `ModuleDict` keyed by `str(layer_id)`, not `ModuleList`

With a `ModuleList`, parameter names are positional: `layers.0.attention.wq`,
`layers.1...`. Now split the model across 3 pipeline stages. Stage 1 holds
original layers 9–17, but as a `ModuleList` they are re-indexed to `0–8`, so
stage 1's `state_dict` claims to contain `layers.0.*` — which collides with
stage 0's real layer 0. Your checkpoint is now unloadable, and merging stages
requires a name-remapping table that has to know the split points.

With a `ModuleDict`, the key **is** the layer identity. Stage 1 deletes the keys
it does not own:

```python
for i in range(n_layers):
    if i not in my_stage_layers:
        del model.layers[str(i)]
```

and what remains is still called `layers.9.attention.wq`. The `state_dict` of
the union of stages is exactly the `state_dict` of the whole model, with no
remapping. **Checkpoint keys become independent of the parallelism
configuration** — which is what lets you save with `P_pp = 4` and resume with
`P_pp = 2` (topic 24).

### 2.2 The `is not None` guards

Under pipeline parallelism:

| Stage | `tok_embeddings` | `layers` | `norm` / `output` |
|-------|------------------|----------|-------------------|
| first | present | subset | `None` |
| middle | `None` | subset | `None` |
| last | `None` | subset | present |

So `forward` must be **polymorphic in its input**: on the first stage `tokens`
is an integer tensor of token ids; on every other stage the same argument is a
float activation tensor `[B, S, d]` received from the previous stage. The guard
`if self.tok_embeddings is not None` is what makes one `forward` serve both
cases. The same on the output side: a middle stage returns hidden states, the
last stage returns logits.

This is why the parameter is named `tokens` but documented as "or the activation
of the previous pipeline stage". One function, three roles, no subclasses.

### 2.3 `init_weights` separated from `__init__`

`__init__` **builds structure**; `init_weights` **writes values**. They are
split because at scale the model is constructed on the `meta` device — shapes
and names, zero bytes allocated — so that a 671B model can be *described* on a
single host, sharded, and only then materialized shard-by-shard on real GPUs.
If initialization happened in `__init__`, you would need the full model in
memory before you could shard it, which defeats the purpose.

This also settles the `freqs_cis` question from topic 04: because it is
`persistent=False` it is not restored by `load_state_dict`, so `init_weights`
recomputes it on the real device:

```python
buffer_device = buffer_device or self.freqs_cis.device
with torch.device(buffer_device):
    self.freqs_cis = precompute_freqs_cis(self.model_args)
```

Note `buffer_device` is threaded all the way down to `TransformerBlock`, which
*raises* if it is missing:

```python
if buffer_device is None:
    raise ValueError("buffer_device must be provided for ...")
```

A loud failure, because the silent version — a `meta` tensor reaching a matmul,
or a buffer left on CPU — produces either a confusing device error deep in a
kernel or a silent host-device sync that destroys throughput.

## 3. Initialization as a variance problem

Now the mathematics. The question initialization answers is:

> What variance should each weight matrix have so that activations neither
> explode nor vanish as they pass through `L` layers, and so that the initial
> loss is the loss of a uniform predictor?

### 3.1 The basic rule for one linear layer

For `y = x W` with `x ∈ R^{d_in}` and independent zero-mean entries:

```
Var(y_j) = Σ_{i=1}^{d_in} Var(x_i) Var(W_ij) = d_in · Var(x) · Var(W)
```

To preserve variance (`Var(y) = Var(x)`):

```
Var(W) = 1 / d_in        i.e.   std(W) = 1/√d_in
```

That is "fan-in" / He-style initialization. For `d = 2048`, `1/√2048 = 0.0221` —
which is where the ubiquitous **`0.02`** comes from. It is `1/√d` for
`d ≈ 2500`, i.e. GPT-2's scale, and it stuck as a constant even though it
should really track `d`.

### 3.2 The residual-stream correction

Now the part that matters. In a pre-norm network the residual stream accumulates:

```
x_L = x_0 + Σ_{ℓ=0}^{L−1} F_ℓ( Norm(x_ℓ) )
```

Each `Norm` output has unit RMS by construction, so each `F_ℓ` contributes
roughly the same variance `σ_F²`, and — because the contributions are
approximately independent at initialization — variances **add**:

```
Var(x_L) ≈ Var(x_0) + L · σ_F²
```

The residual stream grows linearly in `L`. With `L = 100` layers, activations at
the top are 10× larger than at the bottom, the final norm has to divide by that,
and the effective contribution of any single layer to the output is tiny. The fix
is to shrink each layer's *output* projection:

```
┌────────────────────────────────────────────────────────────┐
│  GPT-2 rule:  std(W_out) = 0.02 / √(2L)                    │
│               ⟹ Var(x_L) ≈ Var(x_0) + L · c/(2L) = O(1)    │
└────────────────────────────────────────────────────────────┘
```

The `2` is because there are **two** residual additions per block (attention and
FFN), so `2L` contributions.

Crucially, this scaling is applied only to the matrices that **write into the
residual stream** — `wo` in attention, `w2` in the FFN — not to the matrices that
read out of it (`wq`, `wkv_a`, `w1`), which keep `std = 0.02`.

### 3.3 What `torchfeather` actually does: per-layer depth scaling

```python
self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
```

This is **not** the GPT-2 rule. GPT-2 uses `0.02/√(2L)` — the same value for
every layer. Here the denominator uses `layer_id + 1`, so the scaling gets
*stronger with depth*:

| layer | GPT-2 (`L=27`) | this codebase |
|-------|----------------|---------------|
| 0 | 0.00272 | 0.01414 |
| 6 | 0.00272 | 0.00535 |
| 26 | 0.00272 | 0.00272 |

Only the last layer agrees. Work out the resulting stream variance:

```
Var(x_L) − Var(x_0) ≈ Σ_{ℓ=0}^{L−1} c/(2(ℓ+1)) = (c/2) · H_L ≈ (c/2)(ln L + γ)
```

where `H_L` is the harmonic number. So instead of the linear `L·σ_F²` of naive
init, or the constant `O(1)` of GPT-2, this gives **logarithmic** growth in
depth — for `L = 27`, `H_27/2 ≈ 1.94` versus GPT-2's `0.5`.

The trade this expresses: early layers get a *larger* share of the initial
residual budget, later layers a smaller one. Deep-net initialization work
(and torchtitan, which this follows) finds this trains more stably than uniform
scaling, because the signal reaching the top is dominated by the layers that have
had the most gradient steps applied to them. Growth is still sub-linear, so the
stream stays controlled.

The code comment flags the divergence explicitly:

```python
# This is different from the GPT2-style initialisation, as visible in the HF
# implementation: .../models/gpt2/modeling_gpt2.py#L448-L458
```

Read that as an invitation to check the reasoning, not as boilerplate.

### 3.4 Which matrices get which std

```python
# Attention.init_weights
for linear in (self.wkv_a, self.wkv_b, self.wq):
    nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)   # readers: fixed
nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)  # writer: depth-scaled

# FeedForward.init_weights
nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
for linear in (self.w2, self.w3):
    nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
```

`wo` and `w2` are the residual writers, correctly depth-scaled. But **`w3` also
receives `init_std`**, and `w3` is the up-projection of the SwiGLU — it does not
write to the residual stream:

```
FFN(x) = w2( silu(x w1) ⊙ (x w3) )
```

Shrinking `w3` shrinks the FFN output a second time, so the FFN branch is
attenuated by roughly `init_std²`-worth of scaling rather than `init_std`. This
looks like it should be `0.02` alongside `w1`, and it is inherited from
torchtitan. In practice it is benign — the effect is a somewhat smaller initial
FFN contribution, and the network recovers within a few hundred steps — but it
is a genuine asymmetry between the gate and up paths that a from-first-principles
derivation would not produce. Flagging it is more useful than reproducing it
uncritically.

**Why `trunc_normal_` rather than `normal_`:** it resamples beyond ±2σ (±3σ where
`a`/`b` are given), removing the rare large weight that a normal distribution
guarantees at these tensor sizes. A single 6σ outlier in `wq` is enough to
produce an attention logit that saturates the softmax at step 0.

### 3.5 The output head: aim for unit-variance logits

```python
final_out_std = self.model_args.dim ** -0.5    # 2048^-0.5 = 0.0221
cutoff_factor = 3
nn.init.trunc_normal_(self.output.weight, mean=0.0, std=final_out_std,
                      a=-3*final_out_std, b=3*final_out_std)
```

Here the derivation is exact, which makes it a good check of section 3.1. The
input to `output` has just passed through `self.norm`, so its RMS is 1, i.e.
`Var(h) ≈ 1`. Then:

```
Var(logit_j) = d · Var(h) · Var(W) = d · 1 · (1/d) = 1
```

Logits start with **unit variance**. Why that is exactly what you want:

- Logits of scale 1 across `V = 102400` classes give a near-uniform softmax, so
  the initial loss is close to `ln V`. Precisely: for `z_i ~ N(0, σ²)` iid,

  ```
  E[CE] = E[ log Σ_i e^{z_i} ] − E[z_y] ≈ ln( V · E[e^z] ) = ln V + σ²/2
  ```

  so with `σ = 1` expect `ln V + 0.5`. For `V = 102400` that is
  `11.54 + 0.5 = 12.04`. **This is your step-0 smoke test.** If the first
  reported loss is not within ~0.1 of it, your initialization, your loss
  reduction, or your output matrix orientation is wrong — check it before
  launching on 512 GPUs, not after.
- Logits of scale 10 would pre-commit the model to arbitrary tokens and produce
  enormous initial gradients through the softmax.
- Logits of scale 0.01 would make the softmax numerically flat and the initial
  gradient signal vanishingly small.

Note `output` is **not tied** to `tok_embeddings`. Untied is the modern default
for large `V`: it costs `V·d` extra parameters but avoids forcing one matrix to
serve two different jobs (retrieving a semantic vector vs. scoring one), and it
matters more for parallelism than for quality — tied weights create a dependency
between the *first* and *last* pipeline stages, which a pipeline schedule then
has to synchronize (topic 17).

### 3.6 The embedding: `std = 1.0`

```python
nn.init.normal_(self.tok_embeddings.weight)     # default std = 1.0
```

Not `0.02`. Fifty times larger than every other matrix, and not truncated.
The justification is that the embedding output goes **straight into an
RMSNorm**:

```
h = tok_embeddings(tokens)          # RMS ≈ 1.0 with std=1.0
h_0 = attention_norm(h)             # RMS ≡ 1.0, whatever the input scale was
```

RMSNorm is positively scale-invariant: `RMSNorm(αx) = RMSNorm(x)` for `α > 0`.
So the forward pass genuinely does not care what the embedding scale is — the
first block sees the same input either way. What the choice *does* affect:

- **Gradient scale on the embedding table.** Larger embeddings mean smaller
  relative updates for a given absolute step, and Adam's per-parameter
  normalization interacts with that.
- **The skip path.** `x_0` itself enters the residual sum un-normalized, so
  `Var(x_0) = 1` sets the reference scale that section 3.2's `O(1)` target is
  measured against. With `std = 0.02` the embedding contribution to the residual
  stream would be negligible compared to the blocks'.

The second point is the real reason: `std = 1.0` makes the token identity a
first-class citizen of the residual stream rather than a whisper the layers must
amplify.

## 4. Putting it together

```python
model = DeepSeekV3Model(model_args)          # structure only (possibly on meta)
# ... apply parallelism plans here: PP split, TP, FSDP wrapping (topics 17-25)
model.to_empty(device="cuda")                 # allocate real storage
model.init_weights(buffer_device="cuda")      # write values + rebuild buffers
```

The ordering is load-bearing and worth memorizing, because getting it wrong is
one of the most common ways to waste a night:

1. **Build structure** (`__init__`), cheap, possibly on `meta`.
2. **Apply parallelism**, which *replaces* modules and *shards* parameters.
3. **Materialize** (`to_empty`), which allocates uninitialized storage.
4. **Initialize** (`init_weights`), which writes values into whatever shard this
   rank ended up owning, and recomputes non-persistent buffers.

Initializing before sharding is not merely wasteful — with FSDP it also changes
the *values*, because each rank would draw from its own RNG stream over a
different slice of the tensor. Topic 20 returns to this with the RNG-seeding
rules that make sharded initialization reproducible.

## 5. A sanity check you should run before every big launch

Two constraints of the reference code shape this script, and both are worth
knowing before you debug something else:

- **`seq_len` must equal `max_seq_len`.** `Transformer.forward` passes the whole
  `self.freqs_cis` down unsliced, and `apply_rotary_emb` reshapes it with
  `freqs_cis.view(1, x.size(1), 1, x.size(-1))`. Feed a shorter sequence and you
  get `RuntimeError: shape '[1, 128, 1, 32]' is invalid for input of size
  16384`. Fixed-length training makes this a non-issue in the training loop, but
  it is the first thing you hit when poking at the model interactively — and it
  is exactly the slice point that KV-cache decode and context parallelism need
  (topic 04 §6.2).
- **The MoE path needs CUDA.** `MoE.forward` calls `torch.histc` on an int64
  tensor (`NotImplementedError: "histogram_cpu" not implemented for 'Long'`) and
  `torch._grouped_mm`. For CPU experiments set `n_dense_layers = n_layers` to
  take the dense SwiGLU branch everywhere.

```python
import torch, math
from torchfeather.model.model import DeepSeekV3Model
from torchfeather.model.model_args import DeepSeekV3ModelArgs

torch.manual_seed(0)
S = 512
args = DeepSeekV3ModelArgs(n_layers=4, n_dense_layers=4,   # all-dense: CPU-safe
                           dim=256, vocab_size=1024, inter_dim=512, n_heads=4,
                           max_seq_len=S, original_seq_len=S)
m = DeepSeekV3Model(args)
m.init_weights(buffer_device=torch.device("cpu"))

tokens = torch.randint(0, args.vocab_size, (2, S))        # S == max_seq_len
logits = m(tokens)

print("logit std   ", logits.std().item())
loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens.flatten())
print("initial loss", loss.item(), " ln(V) =", math.log(args.vocab_size))

h = m.tok_embeddings(tokens)
print(" embed  RMS", h.pow(2).mean().sqrt().item())
for name, layer in m.layers.items():
    h = layer(h, m.freqs_cis)
    print(f" layer {name} RMS", h.pow(2).mean().sqrt().item())
print("init_std per layer:", [l.weight_init_std for l in m.layers.values()])
```

Actual output:

```
logit std    0.9846
initial loss 7.4254   ln(V) = 6.9315
 embed  RMS 0.999
 layer 0 RMS 0.999
 layer 1 RMS 0.999
 layer 2 RMS 0.999
 layer 3 RMS 0.999
init_std per layer: [0.01414, 0.01, 0.00816, 0.00707]
```

Every prediction of the derivation is confirmed:

- **`logit std = 0.985 ≈ 1`** — section 3.5's `Var(logit) = d · 1 · (1/d) = 1`.
- **`loss = 7.425`** vs `ln V + σ²/2 = 6.9315 + 0.485 = 7.417`. The estimator
  from section 3.5 is accurate to three decimals. Note it is *above* `ln V`, not
  equal to it — a common source of false alarm.
- **`init_std = [0.01414, 0.01, 0.00816, 0.00707]`** = `0.02/√2, 0.02/√4,
  0.02/√6, 0.02/√8` — section 3.3's `0.02/√(2(ℓ+1))`, exactly.
- **RMS flat at 0.999.** Not growth — and that is correct here, not a bug. The
  embedding enters with RMS 1 and contributions add in quadrature:
  `RMS(x_L) = √(1 + Σ_ℓ σ_F,ℓ²)`. With `σ_F ≈ 0.01` and `L = 4`, the sum is
  `~4e-4` and invisible at three decimals. To *see* the depth effect you need
  either many more layers or a trained model, where `σ_F` is no longer at its
  initialization scale.

What to look for when it goes wrong:

- `initial loss` far from `ln V + σ²/2` → broken init, wrong loss reduction, or a
  transposed/accidentally-tied output matrix.
- **RMS growing linearly** with depth → the depth scaling is not reaching the
  residual writers. Check that `init_weights` receives the per-block
  `weight_init_std` rather than a single global constant.
- **RMS exploding** → depth scaling applied to the wrong matrices (the readers
  instead of the writers).

The whole check costs a second and catches the majority of initialization bugs.

## 6. Check yourself

1. Derive `∂x_{ℓ+1}/∂x_ℓ` for pre-norm and post-norm, and explain from those
   expressions why post-norm transformers need learning-rate warmup.
2. RMSNorm drops mean-centering. Give the argument for why that costs nothing at
   `d = 2048`, and say when it would start to matter.
3. Why is `std = 0.02/√(2L)` and not `0.02/√L`?
4. This codebase uses `0.02/√(2(ℓ+1))`. Compute the resulting residual variance
   as a function of `L` and compare to GPT-2's rule.
5. `w3` receives the depth-scaled `init_std`. Argue whether it should, from the
   definition of SwiGLU.
6. The output head uses `std = d^{−1/2}`. Show this gives unit-variance logits,
   and state the initial loss you therefore expect for `V = 102400`.
7. The embedding uses `std = 1.0`, 50× everything else. Explain why the forward
   pass is indifferent to this, and what it does affect.
8. Why `ModuleDict` keyed by layer id rather than `ModuleList`? Give a concrete
   checkpoint failure that `ModuleList` would cause under `P_pp = 3`.
9. `init_weights` is separate from `__init__`. Give two reasons, one about memory
   and one about correctness under FSDP.

## Next

→ [06 — Attention, the KV cache and arithmetic intensity](06-attention-kv-cache-arithmetic-intensity.md)
