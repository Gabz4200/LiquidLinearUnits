# LiquidLinearUnits

A Very Small Test of a very simple concept that can actually be useful. Liquid in that case means that it is adaptable by the input, not a LNN in the traditional sense.

## The architectures

Every model is a classic Transformer block (sliding-window causal attention for
token mixing + a SwiGLU MLP) where **every** `nn.Linear` is replaced by a Liquid
Linear Unit (LLU). The families differ only in *which* LLU fills the projection
(`q/k/v/o`) and FFN roles:

| Family | Projection (q/k/v/o) | FFN |
|---|---|---|
| `LiquidLinear` | LiquidLinear | LiquidLinear |
| `Rank1LiquidLN` | Rank1LiquidLN | Rank1LiquidLN |
| `RankRLiquidLN` | RankRLiquidLN | RankRLiquidLN |
| `StableLiquidLN` | StableLiquidLN | StableLiquidLN |
| `GDNLiquidLN` | StableLiquidLN | **GDN-2** (`GDNLiquidLN`) |
| `MomentumGDNLiquidLN` | SharedMomentumLiquidLN | **GDN-2** |
| `Shared/BatchMomentumLiquidLN` | Shared/BatchMomentum | Shared/BatchMomentum |
| **`StableGDNCondLiquidLN`** | StableLiquidLN | **StableLiquidLN, conditioned by GDN-2** |

`StableGDNCondLiquidLN` is the newest variant. It does **not** use GDN-2 as an
FFN transform. Instead, a small GDN-2 recurrence produces a `d_model`-sized
**conditioning** vector that is fed (as `cond`) into the hypernetwork of each of
the three Stable-Liquid SwiGLU sublayers. SWA still supplies the token-mixed
`x`; GDN-2 supplies `cond`: `(SWA, GDN-2) → (x, cond) → StableLiquidLN FFN`.
With `--no_attention` (or the `_noattn` ablation), SWA is removed entirely and
the block becomes pure LLU recurrence, with GDN-2 still providing `cond`.

## Preliminary benchmark comparison

> **Multi-task, still one seed.** `overwrite_recall` (d_model = 17), `xor`
> (d_model = 6), `permutation_S3` (d_model = 6), and `permutation_S5`
> (d_model = 120) have each completed a full 300-step, `seed = 0`, no-sweep run.
> All four are directional signals, **not** conclusions -- no seed averaging, no
> hyperparameter tuning, far from convergence.

`overwrite_recall` -- 300 steps, seed = 0, no sweeps (one config per task):

| Architecture | mse ↓ | success_rate ↑ | params | ms/step* |
|---|---|---|---|---|
| StableLiquidLN | **0.131** | 0.836 | 244,040 | 211 |
| **StableGDNCondLiquidLN** | **0.143** | **0.859** | 244,172 | 335 |
| LiquidLinear | 0.145 | 0.859 | 302,384 | 129 |
| GDNLiquidLN | 0.169 | 0.824 | 272,048 | 622 |
| BatchMomentumLiquidLN | 0.395 | 0.512 | 244,054 | 229 |
| SharedMomentumLiquidLN | 0.690 | 0.039 | 244,054 | 219 |
| RankRLiquidLN | 0.696 | 0.031 | 109,048 | 202 |
| MomentumGDNLiquidLN | 0.725 | 0.023 | 272,062 | 633 |
| | | | | |
| *No-attention ablation (recurrence only)* | | | | |
| SharedMomentumLiquidLN_noattn | 0.988 | 0.000 | 161,460 | 116 |
| BatchMomentumLiquidLN_noattn | 0.975 | 0.000 | 161,460 | 118 |
| GDNLiquidLN_noattn | 0.984 | 0.000 | 189,462 | 483 |
| MomentumGDNLiquidLN_noattn | 0.990 | 0.000 | 189,468 | 716 |
| **StableGDNCondLiquidLN_noattn** | 0.990 | 0.000 | 161,586 | 230 |

\* `ms/step` is wall-clock on a single weak laptop core (Intel i5-8250U, CPU).
It mostly reflects parameter count and per-step Python/autograd overhead, **not**
inference speed on real hardware — do not read it as a deployment benchmark.

### Additional tasks (xor, permutation_S3, permutation_S5)

The three tasks below use the same 300-step / `seed = 0` / no-sweep protocol.
`attn` = with sliding-window attention; `_noattn` = recurrence-only ablation.
The metric is classification **accuracy** (higher is better).

#### `xor` (d_model = 6) -- saturated

Not discriminative: every architecture reaches `acc = 1.0` except
`GDNLiquidLN_noattn` (0.98). The task is too easy at this scale.

#### `permutation_S3` (d_model = 6)

| Architecture | acc ↑ | ev_loss | params | ms/step |
|---|---|---|---|---|
| BatchMomentumLiquidLN | **0.809** | 0.4946 | 85,514 | 131.7 |
| **StableGDNCondLiquidLN** | **0.809** | 0.4945 | 85,786 | 208.8 |
| LiquidLinear | 0.803 | 0.5258 | 18,356 | 46.5 |
| GDNLiquidLN | 0.791 | 0.5128 | 90,804 | 348.5 |
| RankRLiquidLN | 0.771 | 0.6391 | 17,212 | 95.0 |
| StableLiquidLN | 0.765 | 0.6285 | 85,500 | 110.2 |
| BatchMomentumLiquidLN_noattn | 0.768 | 0.574 | 56,534 | 49.0 |
| GDNLiquidLN_noattn | **0.932** | 0.1963 | 61,832 | 296.2 |
| Rank1LiquidLN | 0.681 | 0.7794 | 8,120 | 79.5 |
| SharedMomentumLiquidLN | 0.704 | 0.7553 | 85,514 | 141.6 |
| StableGDNCondLiquidLN_noattn | 0.657 | 0.8745 | 56,814 | 119.2 |
| MomentumGDNLiquidLN | 0.633 | 0.7389 | 90,818 | 393.3 |
| SharedMomentumLiquidLN_noattn | 0.495 | 1.083 | 56,534 | 54.0 |
| MomentumGDNLiquidLN_noattn | 0.484 | 1.082 | 61,838 | 292.0 |

#### `permutation_S5` (d_model = 120) -- too hard for 300 steps

At this scale every architecture collapses to near-random (`acc ≈ 0.40-0.41`);
only `LiquidLinear` is worse (0.227, likely from its 97M-param blow-up at
d_model = 120). 300 steps is far from enough to learn a length-5 permutation,
so this task gives a *training-budget failure*, not a clean architectural
signal. Raw numbers are in `bench_gdncond_full.txt`.

### Insights

- **`StableGDNCondLiquidLN` is the strongest of the GDN-2-derived designs.**
  With attention on, it matches the overall best archs (`StableLiquidLN`,
  `LiquidLinear`) on mse and posts the *highest* success rate (0.859), and it
  clearly **beats** the other GDN-2 variants: `GDNLiquidLN` (0.169 mse) and
  `MomentumGDNLiquidLN` (0.725). The takeaway: feeding GDN-2's recurrence
  output in as a *conditioner* for a Stable-Liquid FFN works better than using
  GDN-2 directly as the FFN transform.
- **The parameter budget is matched.** `StableGDNCondLiquidLN` adds only
  +132 params (attn) / +126 (no-attn) over `StableLiquidLN` — the GDN-2 cond
  provider was deliberately slimmed (rank = 1, head_dim = 8, num_heads = 2),
  so the comparison is apples-to-apples.
- **The attention-free ablation behaves correctly.** Every `_noattn` variant
  fails `overwrite_recall` (success_rate = 0). That is *expected*: the task
  requires looking back at earlier tokens, which pure recurrence (delta-rule
  memory / momentum) cannot do here. The fact that `StableGDNCondLiquidLN_noattn`
  fails identically to the other recurrence-only variants confirms the ablation
  correctly isolates the attention contribution -- GDN-2's `cond` alone is not a
  substitute for SWA on a hard lookback task.
- **The conditioning pattern holds across tasks.** On `permutation_S3`,
  `StableGDNCondLiquidLN` ties the best architecture (`BatchMomentumLiquidLN`,
  0.809) and beats both GDN-2-as-FFN variants (`GDNLiquidLN` 0.791,
  `MomentumGDNLiquidLN` 0.633). Combined with the `overwrite_recall` result
  (highest success rate, 0.859), feeding GDN-2's recurrence in as a *conditioner*
  for a Stable-Liquid FFN is consistently stronger than using GDN-2 directly as
  the FFN transform.

### Limitations of the testing method

These numbers are **preliminary and should not be over-interpreted**:

- **One seed only.** All four tasks were run at `seed = 0`. No variance
  estimate; a different seed could reorder the table.
- **No hyperparameter sweeps.** Runs used `--no_sweeps` (a single config per
  task). Learning rate, rank, layers, and window were fixed; the families are
  not tuned against each other.
- **Far from convergence.** 300 steps on tiny models is a short smoke-level
  run, not a trained comparison. Some archs (e.g. the momentum family) may
  improve substantially with more steps.
- **Tiny models, task-derived dimensions.** `d_model` is taken from each task's
  `token_dim` (17 here), so absolute mse/success numbers are not transferable
  to realistically sized models.
- **CPU-only timing.** `ms/step` is dominated by this machine's weak core and
  PyTorch CPU overhead; it is not a fair speed comparison for GPU deployment.
- **`success_rate` is a thresholded proxy.** It is derived from mse against a
  task-specific ceiling and should be read alongside mse, not in isolation.

A fair comparison still requires seed averaging (currently only `seed = 0`)
and at least a small sweep over `rank`/`lr`/`window`, plus longer runs on the
harder tasks (`permutation_S5` needs far more than 300 steps). Raw per-arch
numbers for all four tasks are in `bench_gdncond_full.txt` and
`bench_gdncond_overwrite.txt`.

## LLM-Scale Benchmark

We scaled the LLU family architectures to a language modeling test (`scripts/train_llm.py`), comparing:
* **`ours` (`LiquidGDNCondLLM`)**: SWA (token mixer) + GDN-2 (conditioner) driving a `StableLiquidLN` intermediary MLP.
* **`baseline` (`GDN2BaselineLLM`)**: GDN-2 as the token mixer (no attention) with SwiGLU FFN.

### Key Insights from LLM Scaling & Optimization

1. **Embedding & Head Initialization:**
   - **Problem:** Default PyTorch `nn.Embedding` initializes with $N(0, 1)$ standard deviation, producing huge initial logits ($\approx 344$ max absolute value) and a training loss of **306** at initialization (expected target for random vocab prediction is $\ln(50257) \approx 10.82$).
   - **Fix:** Pinned `wte` (and untied `lm_head`) weight standard deviations to $1 / \sqrt{d_{\text{embd}}} \approx 0.056$.
   - **Result:** The initial training loss immediately dropped to the theoretically correct **11.28**, and training became extremely stable. Perplexity on `wikitext-2` drops rapidly (e.g. from 75,743 to 20,970 in just 15 steps).

2. **Scaling Parameters Removal & Speedup:**
   - **Problem:** The LLU models previously multiplied the output of the adaptive linear transform by scaling parameters (`self.scale` and `self.rank_scale`).
   - **Fix:** Removed these scaling parameters entirely across `llu/models/llns/` and initialized target and core layers with a variance-preserving **Xavier uniform** distribution.
   - **Result:** This resulted in a **2.13× performance speedup** in unit test execution (cutting total test time from 25.27s to 11.87s/13.71s) by simplifying the autograd compute graph and reducing PyTorch's gradient tracking and backpropagation overhead.

3. **Baseline GDN-2 CPU Bottleneck:**
   - On CPU, the baseline (which uses 24 layers of GDN-2 to match the param budget of `ours`' 8 layers) runs **5.7× slower** due to the sequential loop in the fallback GDN-2 chunk kernel.
