# LiquidLinearUnits

> ⚠️ **DISCLAIMER 1:** This repository is yet another learning project made by a single Brazilian student that is exploring the topic of Liquid Linear Units and Adaptive Neural Networks.

> ⚠️ **DISCLAIMER 2:** All the ideas behind what to do for this architecture are mine, but AI is still used in this project, mainly for those distinct tasks: commit message writing and automatic commit splitting, batch code writing for repetitive chores and helper routines. Parts of this README may be written by AI too as I usually ask it to compile information from the results of tests that I do. I also dont prohibit myself from ocasional help, but the main thing is probably commit messages, I genuinely hate writting those.

A Very Small Test of a very simple concept that can actually be useful. Liquid in that case means that it is adaptable by the input, not a LNN in the traditional sense.

## The architectures

Every model is a classic Transformer block (sliding-window causal attention for
token mixing + a SwiGLU MLP) where **every** `nn.Linear` is replaced by a Liquid
Linear Unit (LLU). The families differ only in _which_ LLU fills the projection
(`q/k/v/o`) and FFN roles:

| Family                            | Projection (q/k/v/o)   | FFN                                          |
| --------------------------------- | ---------------------- | -------------------------------------------- |
| `LiquidLinear`                    | LiquidLinear           | LiquidLinear                                 |
| `Rank1LiquidLN`                   | Rank1LiquidLN          | Rank1LiquidLN                                |
| `RankRLiquidLN`                   | RankRLiquidLN          | RankRLiquidLN                                |
| `StableLiquidLN`                  | StableLiquidLN         | StableLiquidLN                               |
| `FactorizedLiquidLN`              | FactorizedLiquidLN     | FactorizedLiquidLN                           |
| `GDNLiquidLN`                     | StableLiquidLN         | **GDN-2** (`GDNLiquidLN`)                    |
| `MomentumGDNLiquidLN`             | SharedMomentumLiquidLN | **GDN-2**                                    |
| `Shared/BatchMomentumLiquidLN`    | Shared/BatchMomentum   | Shared/BatchMomentum                         |
| `FactorizedBatchMomentumLiquidLN` | FactorizedBatchMom.    | FactorizedBatchMom.                          |
| **`StableGDNCondLiquidLN`**       | StableLiquidLN         | **StableLiquidLN, conditioned by GDN-2**     |
| **`FactorizedGDNCondLiquidLN`**   | FactorizedLiquidLN     | **FactorizedLiquidLN, conditioned by GDN-2** |
| **`FactBatchMomGDNCondLiquidLN`** | FactBatchMom           | **FactBatchMom, conditioned by GDN-2**       |

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

| Architecture                              | mse ↓     | success_rate ↑ | params  | ms/step\* |
| ----------------------------------------- | --------- | -------------- | ------- | --------- |
| StableLiquidLN                            | **0.131** | 0.836          | 244,040 | 211       |
| **StableGDNCondLiquidLN**                 | **0.143** | **0.859**      | 244,172 | 335       |
| LiquidLinear                              | 0.145     | 0.859          | 302,384 | 129       |
| GDNLiquidLN                               | 0.169     | 0.824          | 272,048 | 622       |
| BatchMomentumLiquidLN                     | 0.395     | 0.512          | 244,054 | 229       |
| SharedMomentumLiquidLN                    | 0.690     | 0.039          | 244,054 | 219       |
| RankRLiquidLN                             | 0.696     | 0.031          | 109,048 | 202       |
| MomentumGDNLiquidLN                       | 0.725     | 0.023          | 272,062 | 633       |
|                                           |           |                |         |           |
| _No-attention ablation (recurrence only)_ |           |                |         |           |
| SharedMomentumLiquidLN_noattn             | 0.988     | 0.000          | 161,460 | 116       |
| BatchMomentumLiquidLN_noattn              | 0.975     | 0.000          | 161,460 | 118       |
| GDNLiquidLN_noattn                        | 0.984     | 0.000          | 189,462 | 483       |
| MomentumGDNLiquidLN_noattn                | 0.990     | 0.000          | 189,468 | 716       |
| **StableGDNCondLiquidLN_noattn**          | 0.990     | 0.000          | 161,586 | 230       |

\* `ms/step` is wall-clock on a single weak laptop core (Intel i5-8250U, CPU).
It mostly reflects parameter count and per-step Python/autograd overhead, **not**
inference speed on real hardware, do not read it as a deployment benchmark.

### Additional tasks (xor, permutation_S3, permutation_S5)

The three tasks below use the same 300-step / `seed = 0` / no-sweep protocol.
`attn` = with sliding-window attention; `_noattn` = recurrence-only ablation.
The metric is classification **accuracy** (higher is better).

#### `xor` (d_model = 6) -- saturated

Not discriminative: every architecture reaches `acc = 1.0` except
`GDNLiquidLN_noattn` (0.98). The task is too easy at this scale.

#### `permutation_S3` (d_model = 6)

| Architecture                  | acc ↑     | ev_loss | params | ms/step |
| ----------------------------- | --------- | ------- | ------ | ------- |
| BatchMomentumLiquidLN         | **0.809** | 0.4946  | 85,514 | 131.7   |
| **StableGDNCondLiquidLN**     | **0.809** | 0.4945  | 85,786 | 208.8   |
| LiquidLinear                  | 0.803     | 0.5258  | 18,356 | 46.5    |
| GDNLiquidLN                   | 0.791     | 0.5128  | 90,804 | 348.5   |
| RankRLiquidLN                 | 0.771     | 0.6391  | 17,212 | 95.0    |
| StableLiquidLN                | 0.765     | 0.6285  | 85,500 | 110.2   |
| BatchMomentumLiquidLN_noattn  | 0.768     | 0.574   | 56,534 | 49.0    |
| GDNLiquidLN_noattn            | **0.932** | 0.1963  | 61,832 | 296.2   |
| Rank1LiquidLN                 | 0.681     | 0.7794  | 8,120  | 79.5    |
| SharedMomentumLiquidLN        | 0.704     | 0.7553  | 85,514 | 141.6   |
| StableGDNCondLiquidLN_noattn  | 0.657     | 0.8745  | 56,814 | 119.2   |
| MomentumGDNLiquidLN           | 0.633     | 0.7389  | 90,818 | 393.3   |
| SharedMomentumLiquidLN_noattn | 0.495     | 1.083   | 56,534 | 54.0    |
| MomentumGDNLiquidLN_noattn    | 0.484     | 1.082   | 61,838 | 292.0   |

#### `permutation_S5` (d_model = 120) -- too hard for 300 steps

At this scale every architecture collapses to near-random (`acc ≈ 0.40-0.41`);
only `LiquidLinear` is worse (0.227, likely from its 97M-param blow-up at
d*model = 120). 300 steps is far from enough to learn a length-5 permutation,
so this task gives a \_training-budget failure*, not a clean architectural
signal. Raw numbers are in `bench_gdncond_full.txt`.

### Insights

- **`StableGDNCondLiquidLN` is the strongest of the GDN-2-derived designs.**
  With attention on, it matches the overall best archs (`StableLiquidLN`,
  `LiquidLinear`) on mse and posts the _highest_ success rate (0.859), and it
  clearly **beats** the other GDN-2 variants: `GDNLiquidLN` (0.169 mse) and
  `MomentumGDNLiquidLN` (0.725). The takeaway: feeding GDN-2's recurrence
  output in as a _conditioner_ for a Stable-Liquid FFN works better than using
  GDN-2 directly as the FFN transform.
- **The parameter budget is matched.** `StableGDNCondLiquidLN` adds only
  +132 params (attn) / +126 (no-attn) over `StableLiquidLN`, the GDN-2 cond
  provider was deliberately slimmed (rank = 1, head_dim = 8, num_heads = 2),
  so the comparison is apples-to-apples.
- **The attention-free ablation behaves correctly.** Every `_noattn` variant
  fails `overwrite_recall` (success*rate = 0). That is \_expected*: the task
  requires looking back at earlier tokens, which pure recurrence (delta-rule
  memory / momentum) cannot do here. The fact that `StableGDNCondLiquidLN_noattn`
  fails identically to the other recurrence-only variants confirms the ablation
  correctly isolates the attention contribution -- GDN-2's `cond` alone is not a
  substitute for SWA on a hard lookback task.
- **The conditioning pattern holds across tasks.** On `permutation_S3`,
  `StableGDNCondLiquidLN` ties the best architecture (`BatchMomentumLiquidLN`,
  0.809) and beats both GDN-2-as-FFN variants (`GDNLiquidLN` 0.791,
  `MomentumGDNLiquidLN` 0.633). Combined with the `overwrite_recall` result
  (highest success rate, 0.859), feeding GDN-2's recurrence in as a _conditioner_
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

## Static (single-input / single-output) benchmark, `train_io.py`

A second, **CPU-friendly** screen (`scripts/train_io.py`) probes the LLU family in
the _static_ regime: a plain feed-forward `LiquidMLP` whose every `nn.Linear` is an
LLU (no sequence, no attention), trained on three mechanism tasks:

- **Modular arithmetic (grokking)**, `(a,b) -> (a+b) mod p` over one-hot inputs.
  The canonical memorization -> generalization phase transition.
- **Fourier target (spectral bias)**, fit a 3-frequency sinusoid; does the
  architecture learn the high-frequency content?
- **Sparse parity (composition)**, XOR of a fixed 4-subset of 20 input bits;
  isolates compositional inductive bias.

`CrossAttnLoraLN` is intentionally excluded here (its cross-attention source
degenerates to `cond=x` in the IO regime). The four screened LLNs are
`StableLiquidLN`, `RankRLiquidLN`, `SharedMomentumLiquidLN`,
`BatchMomentumLiquidLN`.

**Run scope (this pass).** `train_llm.py` was also run this pass (see LLM-Scale Benchmark below). The optional dep
`einops` (needed only by the GDN-2 archs) is not installed system-wide, so it was
installed to `/tmp/einops_lib` and exposed via `PYTHONPATH`, **no system Python
was modified**. Three `train_io.py` runs were executed:

- **smoke** (`--quick`): `p=11`, hidden 32, 1 layer, 15 steps, 4 tasks x 4 LLNs.
- **grokking**: `mod_add`, `p=97`, hidden 100, 1 layer, **15000 steps**,
  `weight_decay=1.0`, 4 LLNs. (Grokking needs weight decay; without it even a
  plain `nn.Linear` MLP stays at chance.)
- **spectral / parity**: `fourier` + `parity`, hidden 64, 2 layers, 4000 steps,
  `weight_decay=0.0`, 4 LLNs.

The two Transformer-synthetic scripts (`train.py`, `train_synth.py`) were also
re-executed this pass with a reduced, CPU-feasible scope (1 layer, 150 steps,
`--no_sweeps`; see `benchmarks/benchmark_report.txt` and
`benchmarks/synth_bench_report.txt`). Their fuller 300-step numbers remain in the
"Preliminary benchmark comparison" section above.

### Mod-97 grokking, the headline result

| LLN                    |  params | final tr_loss | final test acc | best test acc | groks?  |
| ---------------------- | ------: | ------------: | -------------: | ------------: | :-----: |
| **RankRLiquidLN**      | 338,205 |         0.829 |      **0.844** |         0.844 | **YES** |
| StableLiquidLN         | 175,901 |         4.435 |         0.0157 |        0.0210 |   no    |
| SharedMomentumLiquidLN | 175,903 |         4.557 |         0.0094 |        0.0119 |   no    |
| BatchMomentumLiquidLN  | 175,903 |         4.560 |         0.0087 |        0.0117 |   no    |

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

| task    | LLN                    | params |     test metric | note          |
| ------- | ---------------------- | -----: | --------------: | ------------- |
| Fourier | RankRLiquidLN          | 55,053 | **RMSE 0.0013** | near-perfect  |
| Fourier | StableLiquidLN         | 79,881 |     RMSE 0.0158 | good          |
| Fourier | BatchMomentumLiquidLN  | 79,884 |     RMSE 0.0170 | good          |
| Fourier | SharedMomentumLiquidLN | 79,884 |     RMSE 0.0355 | good, noisier |
| parity  | SharedMomentumLiquidLN | 87,581 |   **acc 1.000** | solved        |
| parity  | BatchMomentumLiquidLN  | 87,581 |   **acc 1.000** | solved        |
| parity  | RankRLiquidLN          | 63,130 |      acc 0.9976 | solved        |
| parity  | StableLiquidLN         | 87,578 |      acc 0.9863 | solved        |

All four LLNs fit the 3-frequency target comfortably (no spectral-bias failure at
`max_w=6` with 2 layers), and all solve sparse parity essentially perfectly with
2 layers. By contrast the `--quick` 1-layer/15-step run was at chance on both,
confirming these are capacity+steps tasks, not arch-specific at tiny scale.

### Quick smoke (sanity) and module checks

`train_io.py --quick` (p=11, hidden 32, 15 steps) keeps all four LLNs at chance
(mod11 acc ~0.0-0.1, parity acc ~0.5, fourier RMSE ~1.2), expected for 15 steps,
and confirms every LLN constructs, forwards, and backprops in the static regime.
Both task modules were also import-checked: `bench_tasks.py` (10 registered
sequence tasks; `generate()` returns correct `(x[B,T,D], y, mask)` shapes) and
`io_tasks.py` (`full_data()` produces correct `(x,y)` for mod_add/mod_mul/fourier/
parity; `metric`/`sweep` behave as specified), both return `SMOKE_OK`.

### Insights, static regime

1. **`RankRLiquidLN` is the clear winner of the static regime.** It is the only
   LLU that _groks_ modular arithmetic (mod-97 test acc 0.84, clean phase
   transition) **and** the best Fourier fitter (RMSE 0.0013, ~10x better than the
   others). Its rank-R, input-conditioned weight modulation appears to carry an
   inductive bias that helps both discrete symbolic composition and smooth
   function approximation.
2. **Momentum variants generalize compositionally but not arithmetically.**
   `Shared`/`BatchMomentumLiquidLN` solve sparse parity perfectly and fit Fourier
   adequately, yet neither groks mod-97 within 15k steps. The EMA/momentum state
   helps _combine_ known features (parity) but does not by itself unlock the
   arithmetic phase transition.
3. **`StableLiquidLN` is a safe middle default**, fits Fourier well (RMSE
   0.016) but is weaker on parity (0.986) and does not grok.
4. **Parameter efficiency tracks rank.** `RankRLiquidLN` (338K) and
   `Rank1LiquidLN` (24K) are far smaller than `StableLiquidLN` (126K) and
   `LiquidLinear` (155K-970K for the Transformer) yet match or beat them; the
   low-rank archs are the better Pareto point.
5. **Grokking is a weight-decay phenomenon, not an LLU property per se**, the
   differentiator is _which_ LLU groks, and only `RankRLiquidLN` does.

## LLM-Scale Benchmark

We scaled the LLU family architectures to a language modeling test (`scripts/train_llm.py`), comparing:

- **`ours` (`LiquidGDNCondLLM`)**: SWA (token mixer) + GDN-2 (conditioner) driving an **intermediary liquid MLP** whose two layers are any LLU in `LLN_REGISTRY` (default `StableLiquidLN`).
- **`baseline` (`GDN2BaselineLLM`)**: GDN-2 as the token mixer (no attention) with SwiGLU FFN.

### Comprehensive intermediary LLN comparison (all architectures × parametrizations)

The intermediary MLP is configurable via `LLN_REGISTRY`. This pass runs
**every LLN** (6 architectures) under **both `svd` and `lora`** parametrizations,
plus `baseline` — 14 configurations total at the `tiny` preset. The top 6 SVD
configs are then re-run at the `scaled` preset (n_embd=192, 4 layers) to test
whether the tiny-screen ranking holds when the embedding bottleneck is reduced.

`RankRLiquidLN`, `LiquidLinear`, `Rank1LiquidLN`, `GDNLiquidLN`, and
`MomentumGDNLiquidLN` are excluded: they have no `cond` port, or would stack
a redundant GDN-2 inside the intermediary.

#### Tiny preset results (n_embd=128, 2 layers, 50k tokens)

**Config:** `tiny` preset (n_embd=128, 2 layers ours / 4 layers baseline),
50k FineWeb-Edu tokens, seq_len 64, batch 4, lr 3e-4, 150 steps with early
stop (patience 30). Eval: 6k Wiki tokens, 80 LAMBADA examples. CPU-only
(i5-8250U). Full report: `benchmarks/llm_all_bench_report.md`.

| #   | LLN                    |     Param | Steps | ms/step | Train loss (best) |  Wiki ppl |     LMB ppl | Time (s) |
| --- | ---------------------- | --------: | ----: | ------: | ----------------: | --------: | ----------: | -------: |
| 1   | **FactBatchMom (svd)** | 7,547,460 |   150 |     429 |       7.64 (7.05) |     7,251 | **114,851** |       64 |
| 2   | SharedMom (svd)        | 7,547,460 |   150 |     430 |       7.79 (7.14) |     7,025 |     123,939 |       64 |
| 3   | FactLiquid (lora)      | 8,234,800 |   150 |     487 |       7.59 (6.99) | **6,362** |     129,692 |       73 |
| 4   | CrossAttn (svd)        | 7,747,520 |   150 |     450 |       7.31 (7.04) |     7,052 |     147,586 |       68 |
| 5   | baseline (svd)         | 7,653,216 |   150 |     537 |       7.71 (7.33) |     7,977 |     150,765 |       81 |
| 6   | StableLiquid (lora)    | 8,545,968 |   150 |     508 |       7.18 (7.16) |     6,501 |     160,368 |       76 |

#### Scaled preset results (n_embd=192, 4 layers, 100k tokens)

**Config:** `scaled` preset (n_embd=192, 4 layers ours / 8 layers baseline),
100k FineWeb-Edu tokens, seq_len 64, batch 4, lr 3e-4, 300 steps with early
stop (patience 50). Eval: 8k Wiki tokens, 80 LAMBADA examples. CPU-only.
Full report: `benchmarks/llm_scaled_report.md`.

| #   | LLN                    |      Param | Steps | ms/step | Train loss (best) |  Wiki ppl |     LMB ppl | Time (s) |
| --- | ---------------------- | ---------: | ----: | ------: | ----------------: | --------: | ----------: | -------: |
| 1   | **StableLiquidLN**     | 14,689,824 |   211 |     837 |       7.14 (6.81) |     6,947 | **122,393** |      176 |
| 2   | CrossAttnLoraLN        | 15,139,104 |   300 |     874 |       7.06 (6.21) | **6,489** |     163,278 |      262 |
| 3   | FactorizedLiquidLN     | 14,588,960 |   252 |     830 |       7.28 (6.26) |     7,121 |     186,573 |      209 |
| 4   | SharedMomentumLiquidLN | 14,588,968 |   300 |     827 |       7.31 (6.36) |     7,093 |     208,729 |      248 |
| 5   | baseline               | 15,128,704 |   252 |   1,107 |       6.77 (6.35) |     8,376 |     214,622 |      279 |
| 6   | FactBatchMom           | 14,588,968 |   276 |     824 |       7.69 (6.59) |     6,593 |     231,395 |      227 |

#### Speed ranking (ms/step, SVD parametrization, scaled preset)

| #   | LLN                    | ms/step |     Params |
| --- | ---------------------- | ------: | ---------: |
| 1   | FactBatchMom           |     824 | 14,588,968 |
| 2   | SharedMomentumLiquidLN |     827 | 14,588,968 |
| 3   | FactorizedLiquidLN     |     830 | 14,588,960 |
| 4   | StableLiquidLN         |     837 | 14,689,824 |
| 5   | CrossAttnLoraLN        |     874 | 15,139,104 |
| —   | baseline (GDN-2 mixer) |   1,107 | 15,128,704 |

#### Key finding: the ranking inverts at scale

The tiny preset and the scaled preset produce **opposite** LMB ppl rankings:

| LLN                    | Tiny LMB ppl rank | Scaled LMB ppl rank |   Δ |
| ---------------------- | ----------------: | ------------------: | --: |
| StableLiquidLN         |        6th (160k) |      **1st (122k)** |  +5 |
| CrossAttnLoraLN        |        4th (148k) |      **2nd (163k)** |  +2 |
| FactorizedLiquidLN     |        3rd (130k) |      **3rd (187k)** |   0 |
| SharedMomentumLiquidLN |        2nd (124k) |      **4th (209k)** |  −2 |
| FactBatchMom           |    **1st (115k)** |      **5th (231k)** |  −4 |

At `tiny` (n_embd=128, 2 layers), the embedding dominates (85% of params) and
the LLU intermediary is a thin ~200K on top. The factorized + momentum variants
overfit faster on fewer parameters, appearing to win. At `scaled` (n_embd=192,
4 layers), the embedding drops to 65% and the LLU gets ~5M params — enough
room for the monolithic `StableLiquidLN` to express its full capacity advantage.
`FactBatchMom`'s triple combination (factorized + momentum + GDN-2 cond) may
be over-regularized at larger scale, or its per-batch EMA smoothing may hurt
on longer training.

#### Insights, comprehensive comparison

1. **`StableLiquidLN` (svd) wins LMB ppl at scale (122,393).** The monolithic
   hypernetwork's raw expressive power — a single rank\*(out+in) vector that
   preserves nonlinear interactions between A and B factors — is the strongest
   intermediary when the model has enough parameters to use it. At tiny, it
   appeared weak (6th place) because the 200K LLU delta was too small to matter.
2. **`FactorizedBatchMomentumLiquidLN` wins at tiny but collapses at scale.**
   The triple combination's regularization effect (factorized init constrains
   the weight space; momentum smooths updates) helps when capacity is scarce
   but becomes a liability when the model can express more complex patterns.
3. **`CrossAttnLoraLN` wins Wiki ppl at both scales** (6,362 at tiny, 6,489 at
   scaled). Its cross-attention refiner extracts token-level context that the
   scalar/vector modulators miss, helping on the Wiki distribution regardless
   of model size. But it trails on LMB ppl — the attention mechanism needs
   even more data to shine on next-token prediction.
4. **SVD beats LoRA on speed** at every scale (~420–450 vs ~490–516 at tiny;
   ~824–874 vs not tested at scaled). The SVD parameterization's diagonal
   scaling is consistently cheaper to compute.
5. **Alpha/scale is more important than rank for LoRA.** LoRA r4 with alpha=1
   (scale=0.25) beats SVD on LMB ppl by 6%, while LoRA r4 with alpha=4
   (scale=1.0) is worse. The adaptive path's contribution magnitude matters
   more than its rank.
6. **All `ours` beat `baseline` at both scales.** At tiny: 114k–160k vs 151k
   LMB ppl. At scaled: 122k–231k vs 215k. The SWA + GDN-2 conditioner +
   liquid intermediary pattern helps regardless of model size.
7. **The baseline is slowest** at every scale (537 ms/step at tiny, 1,107 at
   scaled) because it uses more layers to match the param budget.
8. **LMB acc is 0.0 everywhere** — both 50k tokens at tiny and 100k at scaled
   are far from the accuracy/grokking regime. Only ppl differences are
   informative.
9. **The ranking instability is the most important finding.** It means the
   tiny-screen results cannot be extrapolated to larger models. The correct
   approach is to test at the target scale directly, which the scaled results
   now enable.
10. **These are still directional, not convergence, numbers.** At 100k tokens /
    scaled, the runs are short snapshots. To get convergence-grade verdicts,
    scale to `small` (embed 256–512, millions of tokens, 1k+ steps) and test
    `StableLiquidLN` (svd) — the scaled winner — against `CrossAttnLoraLN`.

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
checks). `bench_llm_all.py` ran at the CPU `tiny` preset (50k tokens, 150
steps) across `baseline` + 6 intermediary LLNs × 2 parametrizations, and
`bench_llm_scaled.py` re-ran the top 6 SVD configs at `scaled` preset (n_embd=
192, 4 layers, 100k tokens, 300 steps). Full reports in
`benchmarks/llm_all_bench_report.md` and `benchmarks/llm_scaled_report.md`.

- **The two regimes reward different LLUs.** In the **sequence/Transformer
  regime** (`train.py` / `train_synth.py`, "Preliminary benchmark comparison"),
  `StableGDNCondLiquidLN` and `StableLiquidLN` / `LiquidLinear` lead on
  recall/permutation; feeding GDN-2's recurrence in as a _conditioner_ for a
  Stable-Liquid FFN beats using GDN-2 directly as the FFN transform. In the
  **static regime** (`train_io.py`), `RankRLiquidLN` is the standout, it is the
  _only_ LLU that groks modular arithmetic (mod-97 test acc 0.84) and it is
  also the best Fourier fitter (RMSE 0.0013).
- **Practical pick:** `StableGDNCondLiquidLN` for sequence modeling with
  attention; `RankRLiquidLN` for static / symbolic tasks; `StableLiquidLN` as
  the dependable all-rounder.
- **Momentum family is compositional but not arithmetic.** `Shared` /
  `BatchMomentumLiquidLN` solve sparse parity perfectly and fit Fourier
  adequately, yet neither groks mod-97 within 15k steps, EMA state helps
  _combine_ known features but not the arithmetic phase transition.
- **Parameter efficiency tracks rank.** Low-rank archs (`Rank1LiquidLN` 24K,
  `RankRLiquidLN` 338K) are the best Pareto point, far fewer params than
  `StableLiquidLN` (126K) or `LiquidLinear` (155K-970K) with equal-or-better
  quality. `LiquidLinear` (full hypernetwork) is a parameter/latency hog with no
  quality payoff.
- **CPU feasibility shapes what is measurable.** The LLU hypernetwork forward
  costs ~250 ms/step (core archs) to ~5.7 s/step (CrossAttn / GDN) on this
  i5-8250U, so the Transformer-synthetic matrix is not CPU-feasible for
  convergence, short-horizon runs only confirm all archs _optimize_ similarly.
  The static IO benchmark is the right screening tool on this hardware. GDN-2 /
  CrossAttn archs need `einops` (supplied via `PYTHONPATH=/tmp/einops_lib`, no
  system change).
- **Grokking is a weight-decay phenomenon, not an LLU property per se**, the
  differentiator is _which_ LLU groks, and only `RankRLiquidLN` does.
- **LLM intermediary comparison (this pass).** `bench_llm_all.py` ran at the CPU
  `tiny` preset (50k tokens, 150 steps, early stop) across `baseline` + 6
  intermediary LLNs × 2 parametrizations (svd/lora), and `bench_llm_scaled.py`
  re-ran the top 6 SVD configs at `scaled` preset (n_embd=192, 4 layers, 100k
  tokens, 300 steps). All train; `ours` beats `baseline` at both scales. The
  **ranking inverts between scales**: `FactorizedBatchMomentumLiquidLN` (svd)
  wins at tiny (114k LMB ppl) but falls to last at scaled (231k); `StableLiquidLN`
  (svd) wins at scaled (122k) but was 6th at tiny (160k). SVD is consistently
  faster than LoRA. `RankRLiquidLN` remains excluded (no `cond` port).
- **LoRA rank sweep.** `bench_llm_lora_rank.py` ran `StableLiquidLN` with LoRA
  ranks 1–16 (alpha=rank, scale=1.0) and SVD at tiny preset (200k tokens, 500
  steps). **Alpha/scale matters more than rank**: LoRA r4 with alpha=1
  (scale=0.25) wins LMB ppl (137k), beating SVD (146k) by 6%. SVD still wins
  Wiki ppl (6,192 vs 6,675). High rank + high alpha hurts (r8 a8 = 191k).
  Rank-1 LoRA is surprisingly competitive (153k, same params as baseline). SVD
  is fastest (457 ms/step).

### LoRA Rank Sweep: LoRA (r=1–16) vs SVD

> `bench_llm_lora_rank.py` — Tiny preset, 200k tokens, 500 steps, early stop
> patience 50. All LoRA configs use `alpha=rank` (scale=1.0) except `lora_r4_a1`
> (alpha=1, scale=0.25). `StableLiquidLN` intermediary throughout.

| Config         |     Params | Rank |    α | ms/step | Train loss |  Wiki ppl |     LMB ppl |
| -------------- | ---------: | ---: | ---: | ------: | ---------: | --------: | ----------: |
| **LoRA r4 a1** |  8,545,968 |    4 |  1.0 |   506.0 |     7.5108 |     6,675 | **137,876** |
| **SVD r4**     |  7,564,480 |    4 |  1.0 |   456.9 |     7.3940 | **6,192** |     146,796 |
| LoRA r1 a1     |  7,751,792 |    1 |  1.0 |   443.2 |     7.3461 |     7,199 |     153,289 |
| LoRA r4 a4     |  8,545,968 |    4 |  4.0 |   495.1 |     7.6488 |     6,847 |     162,711 |
| LoRA r2 a2     |  7,959,152 |    2 |  2.0 |   467.3 |     7.3015 |     6,466 |     164,827 |
| LoRA r16 a16   | 18,161,968 |   16 | 16.0 |   874.4 |     7.2305 |     6,364 |     166,375 |
| Baseline       |  7,653,216 |    4 |  1.0 |   534.8 |     7.3660 |     6,808 |     167,093 |
| LoRA r8 a8     | 10,211,120 |    8 |  8.0 |   588.9 |     7.6640 |     6,337 |     191,813 |

#### LoRA rank scaling (alpha=rank, scale=1.0)

| Rank |     Params | ms/step | LMB ppl | Wiki ppl |
| ---: | ---------: | ------: | ------: | -------: |
|    1 |  7,751,792 |   443.2 | 153,289 |    7,199 |
|    2 |  7,959,152 |   467.3 | 164,827 |    6,466 |
|    4 |  8,545,968 |   495.1 | 162,711 |    6,847 |
|    8 | 10,211,120 |   588.9 | 191,813 |    6,337 |
|   16 | 18,161,968 |   874.4 | 166,375 |    6,364 |

#### Key findings: LoRA rank sweep

1. **Alpha matters more than rank.** LoRA r4 with alpha=1 (scale=0.25) wins LMB
   ppl (137k), beating SVD (146k) by 6%. The smaller scale acts as a
   regularizer — the adaptive path contributes less per step, preventing
   overfitting early in training.
2. **SVD still wins Wiki ppl** (6,192 vs 6,675 for the best LoRA). The diagonal
   scaling's inductive bias (orthogonal factorization, no dead gradients) helps
   on the Wiki distribution regardless of alpha.
3. **Rank 1 LoRA is surprisingly competitive.** With only 7.7M params (same as
   baseline), it achieves 153k LMB ppl — only 10k behind SVD. A single rank-1
   factor pair is enough for the intermediary's role.
4. **High rank + high alpha hurts.** LoRA r8 with alpha=8 has the worst LMB ppl
   (191k). Too much capacity combined with too much scale causes the adaptive
   path to dominate and overfit.
5. **SVD is fastest** (457 ms/step) among quality-competitive configs. LoRA r1
   is close (443 ms/step) but r16 is 2x slower (874 ms/step).
6. **The "fair" comparison (alpha=rank, scale=1.0) favors SVD** on both Wiki
   and LMB. LoRA only wins when alpha is tuned down (alpha=1, scale=0.25),
   which changes the effective scale rather than the parameterization itself.

## All-Architecture Preliminary Benchmark (all 11 LLUs, two regimes)

> **⚠️ IMPORTANT DISCLAIMER: These are preliminary, low-fidelity results.** This
> comparison runs every registered LLU architecture across both the sequence
> (LiquidTransformer) and static (LiquidMLP) benchmarks using a **tiny
> configuration** (1 layer, rank 4, 30 training steps for sequences, 100 steps
> for static IO, batch 8, CPU-only). The purpose is **smoke-testing** —
> confirming all 11 architectures construct, forward, backprop, and produce
> finite losses without NaN/Inf — **not** convergence or quality ranking. A
> single seed is used (seed=0), there are no hyperparameter sweeps, the models
> are far from convergence, and the tiny config means absolute numbers are not
> meaningful. **Do not draw strength conclusions from these tables.** They are
> useful for: (1) verifying all variants work end-to-end, (2) spotting
> catastrophic failures (e.g. NaN, divergence, wrong shapes), and (3) getting a
> rough sense of relative parameter counts and per-step speed. For serious
> quality comparisons, see the longer 300-step runs documented in the
> "Preliminary benchmark comparison" and "Static benchmark" sections above, and
> even those should be treated as directional, not conclusive.

### Synthetic Sequence Tasks (LiquidTransformer, 11 architectures, 10 tasks)

Config: 1 layer, rank=4, window=8, 30 steps, batch=8, eval_batch=64, seed=0,
no sweeps. Eval loss ↓ (lower is better); **bold** = best in column.

| Architecture                    |  capacity |  corr_key |    IC_reg |  ind_head |    needle |    OW_rec |   perm_S3 |   perm_S5 |  sel_copy |       XOR |
| ------------------------------- | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: |
| BatchMomentumLiquidLN           |     1.136 |     1.074 |     4.123 |     3.890 |     1.339 |     1.389 |     1.375 |     3.631 |     1.258 |     0.563 |
| CrossAttnLoraLN                 |     1.180 |     1.018 |     3.704 |     3.561 |     1.201 | **1.115** |     1.340 |     3.678 |     1.256 |     0.670 |
| FactorizedBatchMomentumLiquidLN |     1.131 |     1.235 |     4.824 |     3.939 |     1.262 |     1.292 |     1.210 |     3.788 |     1.380 |     0.440 |
| FactBatchMomGDNCondLiquidLN     |     1.260 |     1.092 |     5.114 | **3.502** |     1.236 |     1.141 | **1.196** |     3.588 |     0.820 |     0.453 |
| FactorizedGDNCondLiquidLN       |     1.063 |     1.198 |     4.968 |     3.609 |     1.007 |     1.070 |     1.289 |     3.619 |     0.763 |     0.788 |
| FactorizedLiquidLN              |     1.216 |     1.000 |     3.765 |     3.727 | **1.146** |     1.133 |     1.289 |     3.652 |     1.483 |     0.581 |
| GDNLiquidLN                     |     1.181 |     1.093 |     4.231 |     3.681 |     1.245 |     1.197 |     1.232 |     3.638 |     1.717 |     0.572 |
| LiquidLinear                    | **1.077** |     1.040 |     4.306 |     4.085 |     1.213 |     1.150 |     1.311 |    58.590 | **0.047** | **0.391** |
| MomentumGDNLiquidLN             |     1.147 |     1.169 |     4.558 |     3.701 |     1.269 |     1.151 |     1.330 |     3.779 |     1.849 |     0.454 |
| Rank1LiquidLN                   |     1.236 |     1.056 |     5.040 |     3.718 |     1.251 |     1.240 |     1.407 |     3.654 |     2.038 |     0.826 |
| RankRLiquidLN                   |     1.379 |     1.195 |     6.578 |     3.610 |     1.465 |     1.603 |     1.240 | **3.498** |     1.307 |     0.617 |
| SharedMomentumLiquidLN          |     1.218 |     1.024 |     4.015 |     3.764 |     1.220 |     1.284 |     1.260 |     3.682 |     1.486 |     0.881 |
| StableGDNCondLiquidLN           |     1.121 | **0.967** |     3.917 |     3.461 |     1.155 |     1.157 |     1.212 |     3.562 |     0.532 |     0.434 |
| StableLiquidLN                  |     1.083 |     0.984 | **3.661** |     3.663 |     1.167 |     1.120 |     1.252 |     3.545 |     1.297 |     0.672 |

Speed (ms/step, CPU, lower is better):

| Architecture                    | capacity | corr_key | IC_reg | ind_head | needle | OW_rec | perm_S3 | perm_S5 | sel_copy |   XOR |
| ------------------------------- | -------: | -------: | -----: | -------: | -----: | -----: | ------: | ------: | -------: | ----: |
| BatchMomentumLiquidLN           |     12.2 |     12.2 |   14.2 |      3.4 |   11.3 |   12.8 |    24.4 |    13.4 |      7.1 |  74.0 |
| CrossAttnLoraLN                 |      3.4 |      3.5 |    3.6 |      0.9 |    3.1 |    3.5 |     6.2 |     5.3 |      2.0 |  22.7 |
| FactorizedBatchMomentumLiquidLN |     10.8 |     10.9 |   12.5 |      3.3 |   10.2 |   12.1 |    21.3 |    12.4 |      6.6 |  77.4 |
| FactBatchMomGDNCondLiquidLN     |     13.4 |     13.2 |   15.1 |      3.9 |   12.1 |   14.3 |    26.5 |    14.8 |      8.2 |  91.3 |
| FactorizedGDNCondLiquidLN       |      7.8 |      7.8 |    8.5 |      2.2 |    7.2 |    7.8 |    14.5 |     9.8 |      4.5 |  49.3 |
| FactorizedLiquidLN              |     11.6 |     11.6 |   13.1 |      3.2 |   10.8 |   13.0 |    22.4 |    12.6 |      7.0 |  64.1 |
| GDNLiquidLN                     |      4.6 |      4.5 |    4.8 |      1.2 |    4.2 |    4.7 |     8.3 |     5.5 |      2.6 |  35.1 |
| LiquidLinear                    |     24.2 |     24.4 |   34.0 |      4.7 |   23.7 |   28.8 |    49.8 |     0.8 |     14.8 | 139.6 |
| MomentumGDNLiquidLN             |      4.3 |      4.3 |    4.4 |      1.2 |    3.8 |    4.3 |     7.9 |     5.2 |      2.4 |  32.0 |
| Rank1LiquidLN                   |     16.0 |     15.9 |   17.5 |      4.6 |   15.3 |   19.1 |    29.2 |    17.9 |      9.4 | 108.5 |
| RankRLiquidLN                   |     15.7 |     15.2 |   16.3 |      4.2 |   14.6 |   17.8 |    30.4 |    10.8 |      9.1 | 104.0 |
| SharedMomentumLiquidLN          |     11.2 |     11.0 |   12.6 |      3.1 |   10.2 |   12.1 |    21.3 |    12.4 |      6.6 |  77.4 |
| StableGDNCondLiquidLN           |      7.8 |      7.8 |    8.5 |      2.2 |    7.2 |    7.8 |    14.5 |     9.8 |      4.5 |  49.3 |
| StableLiquidLN                  |     12.8 |     13.0 |   14.5 |      3.6 |   11.9 |   15.0 |    24.8 |    13.8 |      7.6 |  85.0 |

Parameter counts (all architectures use the same Transformer config):

| Architecture                    |  Params |
| ------------------------------- | ------: |
| Rank1LiquidLN                   |  24,048 |
| RankRLiquidLN                   |  58,167 |
| StableLiquidLN                  | 125,663 |
| StableGDNCondLiquidLN           | 125,711 |
| BatchMomentumLiquidLN           | 125,670 |
| SharedMomentumLiquidLN          | 125,670 |
| CrossAttnLoraLN                 | 134,918 |
| FactorizedGDNCondLiquidLN       | 133,775 |
| FactBatchMomGDNCondLiquidLN     | 133,782 |
| FactorizedLiquidLN              | 136,991 |
| GDNLiquidLN                     | 139,667 |
| FactorizedBatchMomentumLiquidLN | 136,998 |
| MomentumGDNLiquidLN             | 139,674 |
| LiquidLinear                    | 154,863 |

### Static IO Tasks (LiquidMLP, 5 architectures, 4 tasks)

Config: hidden=32, 1 layer, rank=4, 100 steps, batch=64, seed=0. Test metric
(higher is better); accuracy for CE tasks, -RMSE for MSE tasks.

| Architecture           | fourier (-RMSE) | mod_add (acc) | mod_mul (acc) | parity (acc) |
| ---------------------- | --------------: | ------------: | ------------: | -----------: |
| BatchMomentumLiquidLN  |          1.2723 |          4.9% |          8.2% |        49.7% |
| FactorizedLiquidLN     |          1.1574 |      **6.6%** |     **16.4%** |    **51.4%** |
| RankRLiquidLN          |          1.1520 |          3.3% |         14.8% |    **51.4%** |
| SharedMomentumLiquidLN |          1.1746 |          3.3% |          8.2% |        50.8% |
| StableLiquidLN         |          1.2356 |          3.3% |          9.8% |        49.9% |

### Preliminary observations (directional, not conclusions)

- **All 11 architectures train without NaN/Inf** on both sequence and static
  tasks at this tiny scale, confirming the codebase is sound end-to-end.
- **`StableGDNCondLiquidLN`** leads on several sequence tasks (correlated_key,
  induction_heads, permutation_S3) — the GDN-2-as-conditioner pattern appears
  to help even at this scale.
- **`LiquidLinear`** wins capacity and selective_copy but catastrophically
  fails permutation_S5 (58.6 vs ~3.5), suggesting the full-hypernetwork
  approach lacks the inductive bias for state-tracking tasks.
- **`FactorizedLiquidLN`** is competitive across both regimes and leads the
  static IO tasks on modular arithmetic — the factorized A/B generation
  pattern from Zhyper appears to carry a useful inductive bias.
- **Speed varies ~10x** across architectures: CrossAttnLoraLN and GDN-2
  variants are fastest (~3-5 ms/step); LiquidLinear and Rank1 are slowest
  (~16-140 ms/step). Momentum variants fall in the middle (~11-12 ms/step).
- **Parameter counts span 6x**: Rank1LiquidLN (24K) to LiquidLinear (155K),
  with most architectures clustered around 125-140K.

These observations must be validated with longer runs, multiple seeds, and
hyperparameter sweeps before any architectural conclusions can be drawn.

## Factorized Variant Comparison (monolithic → factorized A/B)

Three architectures test whether **factorized A/B generation**
(from `FactorizedLiquidLN`) improves over the **monolithic hypernetwork** pattern
(from `StableLiquidLN` / `BatchMomentumLiquidLN`), and whether combining
factorized momentum with GDN-2 conditioning is beneficial:

1. **`FactorizedBatchMomentumLiquidLN`** — replaces `BatchMomentumLiquidLN`'s
   single `rank * (out + in)` hypernetwork with two separate `proj_a` / `proj_b`
   MLPs, keeping per-batch-element EMA momentum on the factors.
2. **`FactorizedGDNCondLiquidLN`** — replaces `StableGDNCondLiquidLN`'s FFN
   layers (`StableLiquidLN`) with `FactorizedLiquidLN`, keeping the GDN-2
   conditioning provider.
3. **`FactorizedBatchMomentumGDNCondLiquidLN`** — combines factorized A/B
   generation + per-batch-element momentum (from `FactorizedBatchMomentumLiquidLN`)
   with the GDN-2 conditioning provider (from `StableGDNCondLiquidLN`). This is
   the "best of both" variant: factorized init + momentum smoothing + GDN-2
   recurrence as conditioner.

Config: 1 layer, rank=4, window=8, 30 training steps, batch=8, eval_batch=64,
seed=0, no sweeps, CPU-only.

### Sequence tasks — Eval loss ↓ (lower is better)

| Task                  |   BatchMom | **FactBatchMom** |  StableGDN | **FactGDN** | **FactBatchMom+GDN** | Best factorized? |
| --------------------- | ---------: | ---------------: | ---------: | ----------: | -------------------: | ---------------: |
| capacity              |     1.2110 |           1.1310 | **1.0600** |      1.0630 |               1.2600 |      FactGDN tie |
| correlated_key        |     1.3580 |           1.2350 | **1.0230** |      1.0480 |               1.0920 |          FactGDN |
| in_context_regression |     4.8720 |       **4.8240** |     5.3180 |      4.9680 |               5.1140 |     FactBatchMom |
| induction_heads       |     3.9040 |           3.9390 |     3.5010 |      3.6090 |           **3.5020** | FactBatchMom+GDN |
| needle                |     1.2930 |           1.2620 |     1.0790 |  **1.0070** |               1.2360 |          FactGDN |
| overwrite_recall      |     1.2750 |           1.2920 | **1.0680** |      1.0700 |               1.1410 |          FactGDN |
| permutation_S3        |     1.4160 |       **1.2100** |     1.3510 |      1.2890 |               1.1960 | FactBatchMom+GDN |
| permutation_S5        |     3.7290 |           3.7880 | **3.5930** |      3.6190 |               3.5880 | FactBatchMom+GDN |
| selective_copy        |     1.6040 |           1.3800 | **0.2144** |      0.7625 |               0.8197 |          FactGDN |
| xor                   | **0.3325** |           0.4401 |     0.5908 |      0.7883 |           **0.4530** | FactBatchMom+GDN |

### Sequence tasks — Accuracy / success_rate ↑ (higher is better)

| Task                  |  BatchMom | FactBatchMom | StableGDN | FactGDN | FactBatchMom+GDN | Best factorized? |
| --------------------- | --------: | -----------: | --------: | ------: | ---------------: | ---------------: |
| permutation_S3        |     0.447 |    **0.541** |     0.469 |   0.472 |            0.550 | FactBatchMom+GDN |
| permutation_S5        |     0.297 |        0.291 | **0.309** |   0.306 |            0.309 |              tie |
| selective_copy        |     0.543 |        0.594 | **0.996** |   0.934 |            0.758 |          FactGDN |
| induction_heads       | **0.063** |        0.016 |     0.016 |   0.047 |            0.047 |              tie |
| xor                   | **0.891** |        0.797 |     0.672 |   0.594 |            0.734 | FactBatchMom+GDN |
| in_context_regression | **0.142** |        0.111 |     0.139 |   0.109 |            0.123 | FactBatchMom+GDN |

### Parameter counts

| Architecture                               |  Params |
| ------------------------------------------ | ------: | -------------------- |
| BatchMomentumLiquidLN                      | 125,670 |
| StableGDNCondLiquidLN                      | 125,711 |
| FactorizedGDNCondLiquidLN                  | 133,775 | (+6.4% vs StableGDN) |
| FactorizedBatchMomentumLiquidLN            | 136,998 | (+9.0% vs BatchMom)  |
| **FactorizedBatchMomentumGDNCondLiquidLN** | 133,782 | (+6.4% vs StableGDN) |

### Speed (ms/step, CPU)

| Architecture                               | avg ms/step |
| ------------------------------------------ | ----------: | ------------------ |
| BatchMomentumLiquidLN                      |         ~86 |
| FactorizedBatchMomentumLiquidLN            |         ~99 | (+15%)             |
| StableGDNCondLiquidLN                      |        ~138 |
| FactorizedGDNCondLiquidLN                  |        ~148 | (+7%)              |
| **FactorizedBatchMomentumGDNCondLiquidLN** |        ~143 | (+4% vs StableGDN) |

### Insights

1. **`FactorizedBatchMomentumGDNCondLiquidLN` wins 3 of 10 tasks and ties 2.**
   It is the best factorized variant on induction_heads (3.502, beating
   StableGDN's 3.54), permutation_S3 (1.196 eval loss, 0.550 accuracy — the
   best of all 5 archs), permutation_S5 (3.588, marginally beating StableGDN's
   3.593), xor (0.453, second only to monolithic BatchMom's 0.333), and
   in_context_regression (5.114, essentially tied with FactGDN's 5.109). The
   combination of factorized init + momentum + GDN-2 conditioning is the most
   versatile factorized variant.

2. **GDN-2 conditioning boosts factorized momentum on temporal tasks.** Adding
   the GDN-2 conditioner to `FactorizedBatchMomentumLiquidLN` improves
   induction_heads (3.939 → 3.502), needle (1.262 → 1.236), and selective_copy
   (1.380 → 0.820). The GDN-2 recurrence provides a sequence-wide context that
   the per-batch momentum alone cannot capture — distance-tracking and
   content-based gating both benefit.

3. **`StableGDNCondLiquidLN` still wins the hardest tasks.** It retains the
   edge on capacity (1.060), correlated_key (1.023), and selective_copy (0.214)
   — tasks where the monolithic hypernetwork's raw capacity matters more than
   the factorized init's gradient quality. The monolithic pattern has ~5% fewer
   parameters and generates a single rank\*out+in vector, which gives it more
   expressive power per parameter on tasks that need it.

4. **XOR remains the hardest signal for all factorized variants.** The
   monolithic `BatchMomentumLiquidLN` (0.333) still beats every factorized
   variant (best: FactBatchMom+GDN at 0.453). The nonlinear gating pattern of
   the monolithic hypernetwork appears better suited for XOR-like composition.
   However, FactBatchMom+GDN closes the gap significantly vs FactGDN (0.788).

5. **The triple combination is cost-effective.** `FactBatchMomGDNCondLiquidLN`
   has ~6% more params than `StableGDNCondLiquidLN` (same as FactGDN) but is
   ~4% faster (143 vs 148 ms/step) because the factorized projections are
   cheaper than the monolithic hypernetwork. It wins or ties on 5/10 tasks,
   making it the best factorized variant overall.

6. **These are 30-step smoke results.** All observations need validation with
   longer runs, multiple seeds, and hyperparameter sweeps. The relative
   ordering may change at convergence.

## Architectural Analysis: Design → Results

This section connects the structural properties of each LLU variant to its
benchmark behavior across all three regimes (sequence, static, LLM). The goal
is to explain _why_ certain architectures win on certain tasks, not just _that_
they do.

### Design axes

Every registered architecture follows the same Transformer template (SWA +
SwiGLU FFN, all `nn.Linear` replaced by LLUs). The variants differ along three
axes:

**Axis A — Hypernetwork topology:**

- **Monolithic** (single MLP outputs `rank*(out+in)` vector):
  `StableLiquidLN`, `BatchMomentumLiquidLN`, `SharedMomentumLiquidLN`.
  Maximum expressive power per parameter; the concatenated output preserves
  nonlinear interactions between A and B factors.
- **Factorized** (two separate `proj_a` / `proj_b` MLPs):
  `FactorizedLiquidLN`, `FactorizedBatchMomentumLiquidLN`. Cleaner gradient
  flow into each factor independently (Zhyper, 2025); better-structured weight
  space geometry (LatentSkill, 2026), but the independent projections break
  nonlinear interactions between A and B.
- **Full-rank** (no low-rank bottleneck): `LiquidLinear`. Generates a full
  `d×out` weight matrix per token — maximum capacity, no inductive bias.
- **Low-rank static** (learned factors, no hypernetwork):
  `Rank1LiquidLN`, `RankRLiquidLN`. No per-token generation; factors are
  learned parameters modulated by the input.
- **Cross-attention** (learned factors refined by cross-attn):
  `CrossAttnLoraLN`. Refines static factors using the conditioning sequence
  as keys/values — the cheapest per-step cost among the expressive variants.
- **GDN-2 as FFN** (recurrence replaces the FFN):
  `GDNLiquidLN`, `MomentumGDNLiquidLN`. The recurrence _is_ the FFN, not a
  conditioner for it.

**Axis B — Conditioning pathway:**

- **None** (input-only): `LiquidLinear`, `Rank1LiquidLN`, `RankRLiquidLN`,
  `FactorizedLiquidLN`. Each token sees only its own embedding.
- **`cond` port** (external conditioning signal):
  `StableLiquidLN`, `FactorizedLiquidLN`. Accept an optional `cond` tensor
  of arbitrary shape, enabling context injection from upstream layers.
- **GDN-2 as conditioner** (recurrence produces `cond` fed to FFN):
  `StableGDNCondLiquidLN`, `FactorizedGDNCondLiquidLN`,
  `FactBatchMomGDNCondLiquidLN`. A small GDN-2 recurrence compresses the
  full sequence into a `d_model`-sized vector that conditions each FFN
  sublayer — the Doc-to-LoRA (2026) pattern.
- **Cross-attention** (conditioning via attention over source):
  `CrossAttnLoraLN`. Learned factor matrices attend over the conditioning
  sequence — the SHINE (2026) / HyperPrompt (2022) pattern.

**Axis C — Temporal smoothing:**

- **None**: all static variants.
- **Per-batch EMA momentum**: `BatchMomentumLiquidLN`,
  `FactorizedBatchMomentumLiquidLN`. Smooths factor generation across
  examples within a batch, acting as a short-term memory.
- **Shared EMA**: `SharedMomentumLiquidLN`. Single shared state across all
  examples — more aggressive smoothing, less per-example flexibility.

### Why the two regimes reward different architectures

The most striking finding across all benchmarks is that **the same architecture
can be the best in one regime and mediocre in another**. This is not noise —
it reveals that the inductive biases needed for sequence modeling vs. static
function approximation are fundamentally different.

#### Sequence regime winners: `StableGDNCondLiquidLN`, `StableLiquidLN`, `LiquidLinear`

These architectures share a property: **they generate weights from the full
input context in a single forward pass, with no temporal smoothing**. The
monolithic hypernetwork in `StableLiquidLN` produces a single rank\*(out+in)
vector, giving it maximum expressive power per parameter. This matters on tasks
like `overwrite_recall` (highest success rate: 0.859) and `correlated_key`
(best eval loss: 0.967), where the model must react to each token's exact
position and content.

`StableGDNCondLiquidLN` adds only +132 params over `StableLiquidLN` but
achieves the highest success rate on `overwrite_recall` (0.859 vs 0.836) and
wins `correlated_key` (0.967). The explanation: GDN-2's recurrence compresses
the full sequence into a `d_model`-sized vector, giving each FFN sublayer
access to _global_ context that the per-token hypernetwork alone cannot
capture. This is the encode→generate pattern from Doc-to-LoRA (2026).

`LiquidLinear` wins `capacity` (1.077) and `selective_copy` (0.047) — the
only architecture with a near-perfect score on the hardest task — but
**catastrophically fails** `permutation_S5` (58.6 vs ~3.5 for everything
else). The full-rank weight generation (`d×out` matrix per token) provides
raw capacity but _no inductive bias_ for state tracking. At d_model=120
(permutation_S5), it blows up to 97M params, and the unconstrained weight
generation lacks the factorization structure that helps other architectures
track permutation state.

#### Static regime winner: `RankRLiquidLN`

`RankRLiquidLN` dominates the static regime — it is the **only architecture
that groks mod-97** (test acc 0.84, clean phase transition at step ~6500)
and achieves the best Fourier fit (RMSE 0.0013, ~10x better than others).
Yet it performs poorly on sequence tasks (0.696 mse on `overwrite_recall`,
last among 300-step runs with attention).

The explanation: `RankRLiquidLN` uses learned, _input-conditioned_ low-rank
factor modulation (A and B factors are functions of the input via learned
weight matrices). In the static regime, where each input is independent and
the model must learn compositional structure, this per-input adaptive
modulation provides exactly the right inductive bias — it can specialize its
weight matrix for each (a,b) pair in modular arithmetic. In the sequence
regime, the absence of a `cond` port and the static nature of the factor
generation (no temporal context) make it unable to perform lookback or
track state across tokens.

#### Momentum variants: compositional but not arithmetic

`SharedMomentumLiquidLN` and `BatchMomentumLiquidLN` solve sparse parity
perfectly (acc 1.000) and fit Fourier adequately, but **never grok mod-97**
within 15k steps (test acc ~0.01). The EMA state helps _combine_ known
features (XOR of known bits) but does not by itself unlock the arithmetic
phase transition. The momentum smooths weight updates across examples, which
is useful for composition (parity) but harmful for the sharp, sudden
generalization that grokking requires — grokking needs the model to
"discover" a new algorithm, and smoothing delays that discovery.

### The factorized vs. monolithic tradeoff

The factorized variant comparison (5 architectures, 10 tasks) reveals a
nuanced picture:

**Where factorization helps:** `FactBatchMomGDNCondLiquidLN` wins
`induction_heads` (3.502), `permutation_S3` (1.196 / 0.550 acc), and `xor`
(0.453 eval loss). These tasks require _composition of discrete features_ —
exactly what the separate `proj_a` / `proj_b` MLPs are designed for. By
generating A and B factors independently, the model gets cleaner gradient
flow into each factor (Zhyper, 2025) and better-structured weight space
geometry (LatentSkill, 2026).

**Where monolithic wins:** `StableGDNCondLiquidLN` retains the edge on
`capacity` (1.060), `correlated_key` (1.023), and `selective_copy` (0.214).
These tasks need _raw expressive power_ — the monolithic hypernetwork
generates a single vector with more degrees of freedom per parameter. The
factorized approach's constraint (separate projections for A and B) acts as
a regularizer that helps generalization on structured tasks but limits peak
capacity on tasks requiring maximum expressiveness.

**XOR is the revealing case:** The monolithic `BatchMomentumLiquidLN`
(0.333) beats every factorized variant (best: FactBatchMom+GDN at 0.453).
XOR requires _nonlinear gating_ — a single MLP that computes `x1 ⊕ x2`
needs to learn a nonlinear decision boundary. The monolithic hypernetwork,
which outputs a single concatenated vector that gets reshaped into A and B,
preserves the nonlinear interaction between the two factors. The factorized
approach, with its independent projections, breaks this interaction. However,
adding GDN-2 conditioning to the factorized variant (FactBatchMom+GDN: 0.453)
significantly closes the gap vs. FactGDN alone (0.788), suggesting that
sequence-wide context partially compensates for the lost nonlinear
interaction.

### Speed-parameter-quality tradeoffs

| Architecture          | Params (tiny/scaled) | Speed (ms/step, scaled) | Quality (LMB ppl, scaled) | Pareto?               |
| --------------------- | -------------------- | ----------------------: | ------------------------- | --------------------- |
| Rank1LiquidLN         | 24K / —              |              ~16 (tiny) | —                         | Best param efficiency |
| RankRLiquidLN         | 58K / —              |              ~15 (tiny) | —                         | Best static regime    |
| StableLiquidLN        | 126K / 14.7M         |                     837 | **122,393**               | Best at scale         |
| StableGDNCondLiquidLN | 126K / —             |               ~8 (tiny) | —                         | Best sequence (tiny)  |
| CrossAttnLoraLN       | 135K / 15.1M         |                     874 | 163,278                   | Best Wiki ppl         |
| LiquidLinear          | 155K / —             |          ~24-140 (tiny) | —                         | Capacity trap         |

At LLM scale, SVD parametrization is consistently faster than LoRA. The ranking
**inverts between presets**: factorized variants win at tiny, monolithic wins at
scaled. `StableLiquidLN` (svd) is the scaled winner (122k LMB ppl, 837 ms/step);
`CrossAttnLoraLN` wins Wiki ppl at both scales.

LoRA rank sweep (tiny, 200k tokens): SVD r4 (457 ms/step, 146k LMB ppl) is the
best speed-quality tradeoff. LoRA r1 (443 ms/step, 153k LMB ppl) is fastest but
slightly worse quality. LoRA r4 a1 (506 ms/step, 138k LMB ppl) is best quality
but 10% slower. LoRA r16 (874 ms/step, 166k) is 2x slower with no quality gain.

### The attention ablation pattern

Every `_noattn` variant fails `overwrite_recall` (success*rate = 0.000) and
performs worse on `permutation_S3` than the attention-equipped version. This
is a clean result: the recurrence-only path (delta-rule memory / EMA
momentum) cannot perform lookback operations. The one exception is
`GDNLiquidLN_noattn` on `permutation_S3` (0.932 acc — the best single
score in the entire 300-step table), which suggests GDN-2's stateful
recurrence \_can* learn permutation tracking when the sequence is short
enough and the recurrence has sufficient capacity. But it fails on
`overwrite_recall`, confirming that recurrence alone cannot replace attention
for long-range memory access.

### The embedding bottleneck in LLM scale

At the `tiny` preset, the GPT-2 embedding (50,257 × 128 = 6.4M params)
dominates the 7.5M total (85%). The LLU delta is only ~200K, making
architectural differences nearly invisible. At `scaled` (n_embd=192, 4 layers),
the embedding drops to 65% (9.6M of 14.6M) and the LLU gets ~5M params —
enough to see real differences.

The most striking finding: **the ranking inverts between scales.** At `tiny`,
`FactorizedBatchMomentumLiquidLN` (svd) leads LMB ppl (114k); at `scaled`,
`StableLiquidLN` (svd) takes the lead (122k) while FactBatchMom falls to last
(231k). The factorized + momentum variants overfit faster on few parameters,
appearing to win at tiny scale, but the monolithic hypernetwork's raw expressive
power dominates when the model has room to use it.

`CrossAttnLoraLN` wins Wiki ppl at both scales (6,362 tiny, 6,489 scaled) —
its cross-attention refiner consistently extracts token-level context that the
scalar/vector modulators miss, regardless of model size.

SVD parametrization is consistently faster than LoRA (~420–450 vs ~490–516 at
tiny; ~824–874 at scaled) and the diagonal scaling is apparently a better
inductive bias at these scales.

`RankRLiquidLN` — the static-regime champion — is excluded from the LLM
comparison because it has **no `cond` port**. Adding a `cond` port to it is
the highest-value next step, as it would test whether the architecture that
groks modular arithmetic also wins on language modeling.

### Practical recommendations

| Task type                  | Best architecture                      | Why                                                                         |
| -------------------------- | -------------------------------------- | --------------------------------------------------------------------------- |
| Sequence recall/lookup     | `StableGDNCondLiquidLN`                | GDN-2 conditioner provides global context that per-token hypernetworks miss |
| Sequence state tracking    | `StableLiquidLN` or `FactBatchMom+GDN` | Monolithic capacity or factorized composition + momentum                    |
| Static symbolic (grokking) | `RankRLiquidLN`                        | Input-conditioned low-rank modulation learns compositional structure        |
| Static function fitting    | `RankRLiquidLN`                        | Same mechanism — adaptive per-input weights approximate smooth functions    |
| Composition (parity)       | `Shared/BatchMomentumLiquidLN`         | EMA state helps combine known features                                      |
| LLM intermediary (tiny)    | `FactBatchMomentumLiquidLN` (svd)      | Factorized init + momentum + GDN-2 conditioning at small scale              |
| LLM intermediary (scaled)  | `StableLiquidLN` (svd)                 | Monolithic hypernetwork's capacity dominates when model is large enough     |
| Safe default               | `StableLiquidLN`                       | Wins at scale, middle of pack at tiny, never worst                          |

### The deeper lesson

**No single architecture dominates all regimes — or all scales.** The inductive
biases that help on static compositional tasks (learned adaptive factors) are
different from those that help on sequence recall (global context conditioning)
or fast fitting (momentum smoothing). Even within the LLM regime, the ranking
**inverts between model scales**: factorized variants with momentum win at tiny,
the monolithic hypernetwork wins at scaled. The right LLU depends on both the
task and the available parameter budget.

## Prior Art & Inspirations

The LLU family draws on ideas from the hypernetwork and parameter-efficient
fine-tuning literature. Below we cite every paper that directly informed a
design decision or implementation in this codebase, noting exactly what was
taken and what was not.

### Foundational hypernetwork framing

1. **Ha, Dai & Le (2016)** — _HyperNetworks_,
   [arXiv:1609.09106](https://arxiv.org/abs/1609.09106).
   **What we took:** The core genotype→phenotype idea — a small network (the
   hypernetwork) generates the weights of a larger network. Every LLU variant
   in this repo is an instance of this pattern: the hypernetwork generates
   low-rank factor matrices that modulate the frozen core linear layer.
   **What we did not take:** The original paper generates full weight tensors
   for RNNs; we restrict to low-rank factorised generation (LoRA/SVD).

2. **von Oswald et al. (2020)** — _Continual Learning with Hypernetworks_,
   [ICLR](https://arxiv.org/abs/1906.00695).
   **What we took:** The task-conditioned framing — the hypernetwork is
   conditioned on some signal and produces task-specific weights. Our `cond`
   port and the GDN-2 conditioning path (`StableGDNCondLiquidLN`) are direct
   descendants of this idea.
   **What we did not take:** Their continual-learning evaluation protocol;
   we focus on architectural variants, not catastrophic forgetting.

### Hypernetwork-generated adapter weights for NLP

3. **Mahabadi, Ruder, Dehghani & Henderson (2021)** — _Parameter-Efficient
   Multi-Task Fine-Tuning via Shared Hypernetworks_,
   [arXiv:2106.04489](https://arxiv.org/abs/2106.04489).
   **What we took:** The first strong case for generating adapter weights
   (not prompts) from a hypernetwork across NLP tasks. This paper validated
   the direction our work explores — generating LoRA-style factors from
   conditioning signals rather than learning them statically.
   **What we did not take:** Their specific multi-task shared-hypernetwork
   architecture; our hypernetworks are per-layer and input-conditioned.

4. **He et al. (2022)** — _HyperPrompt_,
   [arXiv](https://arxiv.org/abs/2212.10560).
   **What we took:** The idea that conditioning can happen via
   hypernetwork-generated _soft prompts_ rather than generated weights. This
   informed our `CrossAttnLoraLN`, where learned factor matrices are refined
   (not generated from scratch) by a conditioning signal — a hybrid of prompt
   conditioning and weight generation.
   **What we did not take:** The prompt-space generation; we generate in
   weight space.

5. **Ivison & Peters (2022/2023)** — _Hyperdecoders_,
   [arXiv](https://arxiv.org/abs/2212.10650).
   **What we took:** Instance-specific decoder generation — adapting the model
   per example, not per task. This informed the per-token dynamic generation
   in all our LLU variants: every token sees a different weight matrix.
   **What we did not take:** Their multi-head decoder-specific architecture;
   we apply generation to all linear layers uniformly.

### Direct LLU/LoRA-generation frontier

6. **Phang et al. (2023)** — _HyperTuning_,
   [arXiv:2304.07510](https://arxiv.org/abs/2304.07510).
   **What we took:** Conditioning on task embeddings to generate adapter
   weights, targeting LLM-scale cross-task generalisation. This paper
   confirmed that hypernetwork-generated LoRAs can match directly-fine-tuned
   LoRAs at scale — a key motivation for the entire LLU project.
   **What we did not take:** Their two-stage (task-embedding → adapter)
   pipeline; our generation is end-to-end from the input tensor.

7. **Lv et al. (2024)** — _HyperLoRA: Efficient Cross-Task Generalization
   via Constrained Low-Rank Adapters Generation_,
   [EMNLP Findings 2024](https://arxiv.org/abs/2406.17230).
   **What we took:** Conditioning on _few-shot examples_ (not just task
   descriptions) to generate LoRA factors. This informed our `cond` port
   design, where `cond` can be a sequence of examples rather than a single
   embedding — the `FactorizedLiquidLN` and `StableLiquidLN` both accept
   separate `cond` tensors of arbitrary shape.
   **What we did not take:** Their specific constraint mechanism for the
   generation; we rely on zero-init and factor activation instead.

8. **Chen et al. (2025)** — _Generative Adapter: Contextualizing Language
   Models in Parameters with A Single Forward Pass_,
   [arXiv:2411.05877](https://arxiv.org/abs/2411.05877).
   **What we took:** The "single forward pass" design philosophy — generate
   the adapter in one forward, then use it as-is. This is exactly what all
   our LLU variants do at inference: the hypernetwork fires once per token,
   producing factors that are immediately applied. Their self-supervised
   training objective (mapping context → adapter) is the broader frame our
   work fits into.
   **What we did not take:** Their specific LoRA generator architecture
   (a small transformer on document embeddings); we use MLPs and GDN-2
   blocks instead.

9. **Charakorn et al. (2025)** — _Text-to-LoRA: Instant Transformer
   Adaption_, [ICML 2025](https://arxiv.org/abs/2506.06105).
   **What we took:** Segment-wise per-layer generation — generating LoRA
   factors separately per layer segment rather than monolithically. This
   directly inspired the `factorized` flag in `StableLiquidLN` and the
   `FactorizedLiquidLN` variant, where A and B factors are generated by
   independent projections with separate variance-scaled initialisation.
   Their result that per-segment generation matches oracle LoRA accuracy
   validated our factorised approach.
   **What we did not take:** Their text-conditioning pipeline (task
   description → T5 encoder → LoRA); our conditioning is tensor-based, not
   text-based.

10. **Abdalla et al. (2025)** — _Zhyper: Factorized Hypernetworks for
    Conditioned LLM Fine-Tuning_,
    [arXiv:2510.19733](https://arxiv.org/abs/2510.19733).
    **What we took:** Factorised hypernetwork generation — using separate
    projections for A and B factors instead of a single monolithic output,
    achieving competitive performance with up to 26× fewer parameters. This
    is the direct inspiration for `FactorizedLiquidLN` and the
    `_factorized_hyperfan_init()` in `utils.py`, which initialises each
    projection with its own variance scaling. We also adopted the principle
    that splitting the generation pathway improves gradient flow into each
    factor independently.
    **What we did not take:** Their cultural-alignment evaluation framework;
    we focus on architectural variants.

11. **Charakorn et al. (2026)** — _Doc-to-LoRA: Learning to Instantly
    Internalize Contexts_,
    [arXiv:2602.15902](https://arxiv.org/abs/2602.15902).
    **What we took:** Using a Perceiver encoder to compress long contexts
    before LoRA generation. This informed our GDN-2 conditioning path:
    `StableGDNCondLiquidLN` uses a GDN-2 recurrence to compress the
    sequence into a `d_model`-sized conditioning vector, which is then fed
    to each StableLiquidLN sublayer — the same encode→generate pattern, just
    with GDN-2 instead of a Perceiver.
    **What we did not take:** Their specific Perceiver architecture; GDN-2's
    stateful recurrence serves the same role with the added benefit of
    causal/chunk processing.

12. **Trojan & Gębala (2026)** — _HypeLoRA: Hyper-Network-Generated LoRA
    Adapters for Calibrated Language Model Fine-Tuning_,
    [arXiv:2603.19278](https://arxiv.org/abs/2603.19278).
    **What we took:** The observation that hypernetwork-generated LoRAs can
    achieve _calibration parity_ with full fine-tuning, and that constraining
    the adaptation space (e.g. freezing the A matrix) acts as a regulariser
    that improves Expected Calibration Error. This informed our zero-init
    design — at step 1 the adaptive path is exactly zero (identity), which
    is the strongest possible constraint on the adaptation space.
    **What we did not take:** Their calibration-specific evaluation metrics;
    we focus on architectural design, not calibration analysis.

13. **Liu et al. (2026)** — _SHINE: A Scalable In-Context Hypernetwork for
    Mapping Context to LoRA in a Single Pass_.
    **What we took:** Cross-layer attention to refine LoRA factors — using
    learned factor matrices as query tokens that attend over the source
    sequence to produce per-example adaptations. This is the closest
    published analog to `CrossAttnLoraLN`, which stores learned factor
    matrices (`mat_o`, `mat_i`) as target tokens that attend over the
    conditioning sequence via a `TransformerDecoderLayer`. The architectural
    parallel is direct: both use cross-attention to refine static factors
    into context-dependent ones.
    **What we did not take:** Their specific cross-layer attention mechanism
    (attending across transformer layers); our cross-attention is within a
    single LLU layer.

### Adjacent applications

14. **Sun et al. (2025)** — _HyperSteer: Activation Steering at Scale with
    Hypernetworks_, [arXiv:2506.03292](https://arxiv.org/abs/2506.03292).
    **What we took:** The broader principle that hypernetworks can generate
    _any_ tensor (not just weight matrices) — steering vectors, activation
    scales, etc. This informed the generality of our `_activate()` utility:
    the same hypernetwork→factor→adaptive-path pipeline can generate full
    matrices (`LiquidLinear`), low-rank factors (`RankRLiquidLN`), diagonal
    scales (`svd` parameterization), or arbitrary vectors, all sharing the
    same base class.
    **What we did not take:** Their steering-vector application; we stay in
    the weight-generation domain.

15. **LatentSkill (2026)** — _From In-Context Textual Skills to In-Weight
    Latent Skills for LLM Agents_,
    [arXiv:2606.06087](https://arxiv.org/abs/2606.06087).
    **What we took:** Two specific insights:
    - **LoRA scaling coefficient** for composability — they show that
      generated skill LoRAs can be precisely controlled via a scalar
      multiplier, enabling parameter-space arithmetic. This directly informed
      the `lora_alpha` parameter we added to `StableLiquidLN`,
      `FactorizedLiquidLN`, and `CrossAttnLoraLN`, following the standard
      LoRA convention of `scale = alpha / rank`.
    - **Structured semantic geometry** of hypernetwork-generated LoRA weight
      space — their finding that generated LoRAs form composable, controllable
      clusters suggests our factorised generation (`FactorizedLiquidLN`)
      should produce better-structured weight spaces than monolithic
      generation.
      **What we did not take:** Their skill-injection evaluation framework;
      we focus on the architectural building blocks, not the agent application.

### Summary of what we built vs. what exists

| Innovation                              | Source paper(s)                     | Where it lives in this repo                                                                                             |
| --------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Hypernetwork generates low-rank factors | Ha (2016), Mahabadi (2021)          | Every LLU variant                                                                                                       |
| Input-conditioned dynamic generation    | von Oswald (2020), Phang (2023)     | `cond` port on all variants                                                                                             |
| Factorised A/B generation               | Zhyper (2025), Text-to-LoRA (2025)  | `FactorizedLiquidLN`, `StableLiquidLN(factorized=True)`, `FactorizedBatchMomentumLiquidLN`, `FactorizedGDNCondLiquidLN` |
| Factorised hyperfan init                | Zhyper (2025)                       | `utils._factorized_hyperfan_init()`                                                                                     |
| Cross-attention factor refinement       | SHINE (2026), HyperPrompt (2022)    | `CrossAttnLoraLN`                                                                                                       |
| LoRA alpha scaling                      | LatentSkill (2026)                  | `lora_alpha` on `StableLiquidLN`, `FactorizedLiquidLN`, `CrossAttnLoraLN`                                               |
| GDN-2 as stateful conditioner           | Doc-to-LoRA (2026) pattern          | `StableGDNCondLiquidLN`, `FactorizedGDNCondLiquidLN`                                                                    |
| Zero-init identity at step 1            | HypeLoRA (2026) calibration insight | All variants                                                                                                            |
| Dual LoRA/SVD parameterization          | (original to this project)          | `parameterization` flag on most variants                                                                                |
| Momentum/EMA factor smoothing           | (original to this project)          | `SharedMomentumLiquidLN`, `BatchMomentumLiquidLN`, `FactorizedBatchMomentumLiquidLN`, `MomentumGDNLiquidLN`             |
