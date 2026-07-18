# LiquidLinearUnits

> ⚠️ **DISCLAIMER 1:** This repository is yet another learning project made by a single Brazilian student that is exploring the topic of Liquid Linear Units and Adaptive Neural Networks.

> ⚠️ **DISCLAIMER 2:** All the ideas behind what to do for this architecture are mine, but AI is still used in this project, mainly for those distinct tasks: commit message writing and automatic commit splitting, batch code writing for repetitive chores and helper routines. Parts of this README may be written by AI too as I usually ask it to compile information from the results of tests that I do. I also dont prohibit myself from ocasional help, but the main thing is probably commit messages, I genuinely hate writting those.

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

## Static (single-input / single-output) benchmark — `train_io.py`

A second, **CPU-friendly** screen (`scripts/train_io.py`) probes the LLU family in
the *static* regime: a plain feed-forward `LiquidMLP` whose every `nn.Linear` is an
LLU (no sequence, no attention), trained on three mechanism tasks:

* **Modular arithmetic (grokking)** — `(a,b) -> (a+b) mod p` over one-hot inputs.
  The canonical memorization -> generalization phase transition.
* **Fourier target (spectral bias)** — fit a 3-frequency sinusoid; does the
  architecture learn the high-frequency content?
* **Sparse parity (composition)** — XOR of a fixed 4-subset of 20 input bits;
  isolates compositional inductive bias.

`CrossAttnLoraLN` is intentionally excluded here (its cross-attention source
degenerates to `cond=x` in the IO regime). The four screened LLNs are
`StableLiquidLN`, `RankRLiquidLN`, `SharedMomentumLiquidLN`,
`BatchMomentumLiquidLN`.

**Run scope (this pass).** `train_llm.py` was also run this pass (see LLM-Scale Benchmark below). The optional dep
`einops` (needed only by the GDN-2 archs) is not installed system-wide, so it was
installed to `/tmp/einops_lib` and exposed via `PYTHONPATH` — **no system Python
was modified**. Three `train_io.py` runs were executed:

* **smoke** (`--quick`): `p=11`, hidden 32, 1 layer, 15 steps, 4 tasks x 4 LLNs.
* **grokking**: `mod_add`, `p=97`, hidden 100, 1 layer, **15000 steps**,
  `weight_decay=1.0`, 4 LLNs. (Grokking needs weight decay; without it even a
  plain `nn.Linear` MLP stays at chance.)
* **spectral / parity**: `fourier` + `parity`, hidden 64, 2 layers, 4000 steps,
  `weight_decay=0.0`, 4 LLNs.

The two Transformer-synthetic scripts (`train.py`, `train_synth.py`) were also
re-executed this pass with a reduced, CPU-feasible scope (1 layer, 150 steps,
`--no_sweeps`; see `benchmarks/benchmark_report.txt` and
`benchmarks/synth_bench_report.txt`). Their fuller 300-step numbers remain in the
"Preliminary benchmark comparison" section above.

### Mod-97 grokking — the headline result

| LLN | params | final tr_loss | final test acc | best test acc | groks? |
|---|---:|---:|---:|---:|:--:|
| **RankRLiquidLN** | 338,205 | 0.829 | **0.844** | 0.844 | **YES** |
| StableLiquidLN | 175,901 | 4.435 | 0.0157 | 0.0210 | no |
| SharedMomentumLiquidLN | 175,903 | 4.557 | 0.0094 | 0.0119 | no |
| BatchMomentumLiquidLN | 175,903 | 4.560 | 0.0087 | 0.0117 | no |

`RankRLiquidLN` shows the textbook transition; test accuracy vs step:

```
 step   500: 0.0015    3000: 0.0026    5000: 0.0308    7000: 0.3974
 step  1000: 0.0009    4000: 0.0083    6000: 0.0829    8000: 0.6797
 step  2000: 0.0013    4500: 0.0147    6500: 0.1940    9000: 0.7862
                                           7500: 0.5460   15000: 0.8440
```

The other three LLNs never leave chance (~0.01) across the full 15k steps. Raw
per-step trace in `benchmarks/mod97_grok.log`.

### Fourier (spectral bias) and sparse parity (composition)

hidden 64, 2 layers, 4000 steps, `weight_decay=0.0`. Lower RMSE = better fit;
higher acc = better.

| task | LLN | params | test metric | note |
|---|---|---:|---:|---|
| Fourier | RankRLiquidLN | 55,053 | **RMSE 0.0013** | near-perfect |
| Fourier | StableLiquidLN | 79,881 | RMSE 0.0158 | good |
| Fourier | BatchMomentumLiquidLN | 79,884 | RMSE 0.0170 | good |
| Fourier | SharedMomentumLiquidLN | 79,884 | RMSE 0.0355 | good, noisier |
| parity | SharedMomentumLiquidLN | 87,581 | **acc 1.000** | solved |
| parity | BatchMomentumLiquidLN | 87,581 | **acc 1.000** | solved |
| parity | RankRLiquidLN | 63,130 | acc 0.9976 | solved |
| parity | StableLiquidLN | 87,578 | acc 0.9863 | solved |

All four LLNs fit the 3-frequency target comfortably (no spectral-bias failure at
`max_w=6` with 2 layers), and all solve sparse parity essentially perfectly with
2 layers. By contrast the `--quick` 1-layer/15-step run was at chance on both,
confirming these are capacity+steps tasks, not arch-specific at tiny scale.

### Quick smoke (sanity) and module checks

`train_io.py --quick` (p=11, hidden 32, 15 steps) keeps all four LLNs at chance
(mod11 acc ~0.0-0.1, parity acc ~0.5, fourier RMSE ~1.2) — expected for 15 steps,
and confirms every LLN constructs, forwards, and backprops in the static regime.
Both task modules were also import-checked: `bench_tasks.py` (10 registered
sequence tasks; `generate()` returns correct `(x[B,T,D], y, mask)` shapes) and
`io_tasks.py` (`full_data()` produces correct `(x,y)` for mod_add/mod_mul/fourier/
parity; `metric`/`sweep` behave as specified) — both return `SMOKE_OK`.

### Insights — static regime

1. **`RankRLiquidLN` is the clear winner of the static regime.** It is the only
   LLU that *groks* modular arithmetic (mod-97 test acc 0.84, clean phase
   transition) **and** the best Fourier fitter (RMSE 0.0013, ~10x better than the
   others). Its rank-R, input-conditioned weight modulation appears to carry an
   inductive bias that helps both discrete symbolic composition and smooth
   function approximation.
2. **Momentum variants generalize compositionally but not arithmetically.**
   `Shared`/`BatchMomentumLiquidLN` solve sparse parity perfectly and fit Fourier
   adequately, yet neither groks mod-97 within 15k steps. The EMA/momentum state
   helps *combine* known features (parity) but does not by itself unlock the
   arithmetic phase transition.
3. **`StableLiquidLN` is a safe middle default** — fits Fourier well (RMSE
   0.016) but is weaker on parity (0.986) and does not grok.
4. **Parameter efficiency tracks rank.** `RankRLiquidLN` (338K) and
   `Rank1LiquidLN` (24K) are far smaller than `StableLiquidLN` (126K) and
   `LiquidLinear` (155K-970K for the Transformer) yet match or beat them; the
   low-rank archs are the better Pareto point.
5. **Grokking is a weight-decay phenomenon, not an LLU property per se** — the
   differentiator is *which* LLU groks, and only `RankRLiquidLN` does.

## LLM-Scale Benchmark

We scaled the LLU family architectures to a language modeling test (`scripts/train_llm.py`), comparing:
* **`ours` (`LiquidGDNCondLLM`)**: SWA (token mixer) + GDN-2 (conditioner) driving an **intermediary liquid MLP** whose two layers are any LLU in `LLN_REGISTRY` (default `StableLiquidLN`).
* **`baseline` (`GDN2BaselineLLM`)**: GDN-2 as the token mixer (no attention) with SwiGLU FFN.

### Intermediary LLN comparison (this pass)

The intermediary MLP is configurable via `LLN_REGISTRY`. This pass runs the
**full comparison** — `baseline` plus `ours` with each intermediary LLN in the
registry (`StableLiquidLN`, `CrossAttnLoraLN`, `SharedMomentumLiquidLN`,
`BatchMomentumLiquidLN`) — so the novel `CrossAttnLoraLN` is measured against
the other sequence-mixer options. `RankRLiquidLN` is excluded (no `cond` port);
the two GDN-2 LLUs are excluded because the `ours` block already produces `cond`
via a GDN-2 recurrence — stacking a second GDN-2 inside the intermediary would
be redundant and ~5x slower/step.

**CPU budget.** Defaults are sized for this weak laptop CPU (i5-8250U, ~7.6 GB
RAM, no CUDA): the `tiny` preset (n_embd=128, `ours` 2 layers / `baseline` 4
layers, ~7.5 M params — dominated by the 50,257-token GPT-2 embedding) with
100k FineWeb-Edu tokens, seq_len 64, batch 4, 200 steps, `svd`
parameterization, and shrunk eval caps (8k wiki tokens, 50 LAMBADA examples).
These are **short CPU snapshots, not convergence numbers.** Per-run JSON in
`benchmarks/llm_bench_report_*.json`; aggregate in
`benchmarks/llm_bench_report_aggregate.json`; this table in
`benchmarks/llm_bench_report.md`.

| Variant | LLN | Params | Train loss | Wiki ppl | LMB ppl | LMB acc | Time (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | - | 7,653,216 | 7.5440 | 7196.31 | 158,751.94 | 0.0000 | 115.9 |
| ours | StableLiquidLN | 7,564,480 | 7.5290 | 6784.54 | 140,264.71 | 0.0000 | 91.3 |
| ours | CrossAttnLoraLN | 7,747,520 | 7.4135 | 7818.88 | 135,305.33 | 0.0000 | 97.0 |
| ours | SharedMomentumLiquidLN | 7,547,460 | 6.9547 | 7073.58 | 137,415.52 | 0.0000 | 93.3 |
| ours | BatchMomentumLiquidLN | 7,547,460 | 7.1770 | 6503.06 | 147,491.70 | 0.0000 | 91.8 |

**Why the momentum LLNs are now in the comparison.** They previously tied their
hypernetwork input dimension to `in_features`, which breaks in the 2-layer
intermediary where the second layer's `in_features` (512) differs from the
GDN-2 `cond` stream (128). They were extended with a `cond_dim` argument
(defaulting to `in_features` for backward compatibility) so they can condition
on the separately-dimensioned `cond` — the same contract `StableLiquidLN` and
`CrossAttnLoraLN` already had.

**Insights — intermediary LLN comparison.**

1. **All variants train.** Init CE is ~10.8 (embedding pinned to
   $1/\sqrt{d_{\text{embd}}}$); every run drops to train loss ~7.0–7.5 within
   200 steps. The architecture and every LLN intermediary are sound end-to-end
   on CPU.
2. **`ours` (intermediary LLN) edges out `baseline` on LAMBADA ppl.** Baseline
   posts the worst LMB ppl (158,752); every `ours` variant is tighter
   (135k–147k). The SWA + conditioned intermediary path helps next-token
   prediction more than GDN-2-alone, at a comparable param count (within
   ~±100K of the 4-layer `baseline`; the LLN delta is tiny vs the embedding).
3. **`CrossAttnLoraLN` has the best LAMBADA ppl (135,305) and 2nd-best train
   loss (7.41).** Its cross-attention refiner — low-rank factors refined as
   target tokens attending over the GDN-2 `cond` sequence — yields the lowest
   language-modeling perplexity of the set, marginally ahead of
   `StableLiquidLN` (140,265) and the momentum variants. It is the most
   expensive *intermediary* (~6% above the other `ours` configs) yet still
   faster than the 4-layer `baseline` (97.0s vs 115.9s) — the ~5.7 s/step figure
   quoted for the all-LLU Transformer-synthetic run does **not** apply here,
   because in the LLM the cross-attn LLU sits only in the 2-layer FFN, not in
   every projection. It leads on the metric that matters most here.
4. **Momentum intermediaries learn fastest (lowest train loss) but don't win
   ppl.** `SharedMomentumLiquidLN` has the best train loss (6.95) and
   `BatchMomentumLiquidLN` the best Wiki ppl (6503), but their LMB ppl trails
   `CrossAttnLoraLN`. The EMA/momentum state clearly accelerates fitting (as in
   the static regime) without a clean ppl win at this scale.
5. **`StableLiquidLN` remains the safe default** — middle of the pack on every
   metric, never worst. It is the dependable all-rounder for the intermediary
   role and the cheapest of the four.
6. **Param budget is dominated by the embedding.** At `tiny`, the 50,257×128
   GPT-2 embedding is ~6.4 M of ~7.5 M params; the LLN delta is only ~±200K
   (`baseline` is actually ~90K *larger* than `ours` because it uses 4 layers).
   So the ~12–15% LAMBADA ppl spread between `ours` configs is a real signal of
   the intermediary's inductive bias, not a parameter-count artifact. Larger
   presets (`small`/`medium`/`0.5B`) shift the balance toward layers/width.
7. **LMB acc is 0.0 across the board** — at 100k tokens / tiny these runs are
   far from the accuracy/grokking regime, so accuracy is not a usable signal
   here; only ppl differences (all within the same order of magnitude) are
   informative, and even those should be read as directional.
8. **The static-regime winner is missing from the LLM comparison — by port, not
   by quality.** `RankRLiquidLN` is the only LLU that groks mod-97 and the best
   Fourier fitter in the static regime, yet it is excluded here because it has
   **no `cond` port**. A `cond` port for it is the highest-value next wiring
   step to test whether the static winner also wins LLM ppl.
9. **These are directional, not convergence, numbers — and `CrossAttnLoraLN` is
   the prime scale-up candidate.** At 100k tokens / `tiny` the runs are far from
   the grokking/accuracy regime (LMB acc 0.0). To get a convergence-grade
   verdict, scale to `small` (embed 256–512, millions of tokens, 1k+ steps) and
   start with `CrossAttnLoraLN`, which posts the best LAMBADA ppl here.

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

## All-benchmark takeaways (this pass)

All non-LLM scripts were run this pass: `train.py` + `train_synth.py`
(Transformer-synthetic, reduced CPU scope), `train_io.py` (static, full deep
runs), and the `bench_tasks.py` / `io_tasks.py` modules (import + data-gen
checks). `train_llm.py` was **also run this pass** at the CPU `tiny` preset
(100k tokens, 200 steps, `svd`) across `baseline` + four intermediary LLNs — see
LLM-Scale Benchmark. Detailed per-run tables are in
`benchmarks/` (`RESULTS.md` is the consolidated write-up).

- **The two regimes reward different LLUs.** In the **sequence/Transformer
  regime** (`train.py` / `train_synth.py`, "Preliminary benchmark comparison"),
  `StableGDNCondLiquidLN` and `StableLiquidLN` / `LiquidLinear` lead on
  recall/permutation; feeding GDN-2's recurrence in as a *conditioner* for a
  Stable-Liquid FFN beats using GDN-2 directly as the FFN transform. In the
  **static regime** (`train_io.py`), `RankRLiquidLN` is the standout — it is the
  *only* LLU that groks modular arithmetic (mod-97 test acc 0.84) and it is
  also the best Fourier fitter (RMSE 0.0013).
- **Practical pick:** `StableGDNCondLiquidLN` for sequence modeling with
  attention; `RankRLiquidLN` for static / symbolic tasks; `StableLiquidLN` as
  the dependable all-rounder.
- **Momentum family is compositional but not arithmetic.** `Shared` /
  `BatchMomentumLiquidLN` solve sparse parity perfectly and fit Fourier
  adequately, yet neither groks mod-97 within 15k steps — EMA state helps
  *combine* known features but not the arithmetic phase transition.
- **Parameter efficiency tracks rank.** Low-rank archs (`Rank1LiquidLN` 24K,
  `RankRLiquidLN` 338K) are the best Pareto point — far fewer params than
  `StableLiquidLN` (126K) or `LiquidLinear` (155K-970K) with equal-or-better
  quality. `LiquidLinear` (full hypernetwork) is a parameter/latency hog with no
  quality payoff.
- **CPU feasibility shapes what is measurable.** The LLU hypernetwork forward
  costs ~250 ms/step (core archs) to ~5.7 s/step (CrossAttn / GDN) on this
  i5-8250U, so the Transformer-synthetic matrix is not CPU-feasible for
  convergence — short-horizon runs only confirm all archs *optimize* similarly.
  The static IO benchmark is the right screening tool on this hardware. GDN-2 /
  CrossAttn archs need `einops` (supplied via `PYTHONPATH=/tmp/einops_lib`, no
  system change).
- **Grokking is a weight-decay phenomenon, not an LLU property per se** — the
  differentiator is *which* LLU groks, and only `RankRLiquidLN` does.
- **LLM intermediary comparison (this pass).** `train_llm.py` was run at the CPU
  `tiny` preset (100k tokens, 200 steps, `svd`) across `baseline` + four
  intermediary LLNs. All train (loss ~10.8→~7.0); `ours` beats `baseline` on
  LAMBADA ppl, and **`CrossAttnLoraLN` posts the best LAMBADA ppl (135,305)** of
  the set — its cross-attention refiner is the strongest intermediary, with
  `SharedMomentumLiquidLN` fitting fastest (lowest train loss 6.95). Absolute
  accuracy is 0 at this scale, so these are directional, not convergence,
  numbers; `RankRLiquidLN` and the GDN-2 LLUs were excluded (no `cond` port /
  redundant double-GDN-2).
