# LLU Architecture Benchmark Comparison

Generated: 2026-07-23 09:48:47
Device: CPU | Config: quick (tiny model, 30 steps)

Legend: **bold** = best in column, tr = train loss, ev = eval loss

## Synthetic Sequence Tasks (LiquidTransformer)

### Eval Loss by Task (lower is better)

| Architecture | capacity | correlated_key | in_context_regression | induction_heads | needle | overwrite_recall | permutation_S3 | permutation_S5 | selective_copy | xor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BatchMomentumLiquidLN | 1.136 | 1.074 | 4.123 | 3.890 | 1.339 | 1.389 | 1.375 | 3.631 | 1.258 | 0.563 |
| CrossAttnLoraLN | 1.180 | 1.018 | 3.704 | 3.561 | 1.201 | **1.115** | 1.340 | 3.678 | 1.256 | 0.670 |
| FactorizedLiquidLN | 1.216 | 1.000 | 3.765 | 3.727 | **1.146** | 1.133 | 1.289 | 3.652 | 1.483 | 0.581 |
| GDNLiquidLN | 1.181 | 1.093 | 4.231 | 3.681 | 1.245 | 1.197 | 1.232 | 3.638 | 1.717 | 0.572 |
| LiquidLinear | **1.077** | 1.040 | 4.306 | 4.085 | 1.213 | 1.150 | 1.311 | 58.590 | **0.047** | **0.391** |
| MomentumGDNLiquidLN | 1.147 | 1.169 | 4.558 | 3.701 | 1.269 | 1.151 | 1.330 | 3.779 | 1.849 | 0.454 |
| Rank1LiquidLN | 1.236 | 1.056 | 5.040 | 3.718 | 1.251 | 1.240 | 1.407 | 3.654 | 2.038 | 0.826 |
| RankRLiquidLN | 1.379 | 1.195 | 6.578 | 3.610 | 1.465 | 1.603 | 1.240 | **3.498** | 1.307 | 0.617 |
| SharedMomentumLiquidLN | 1.218 | 1.024 | 4.015 | 3.764 | 1.220 | 1.284 | 1.260 | 3.682 | 1.486 | 0.881 |
| StableGDNCondLiquidLN | 1.121 | **0.967** | 3.917 | **3.461** | 1.155 | 1.157 | **1.212** | 3.562 | 0.532 | 0.434 |
| StableLiquidLN | 1.083 | 0.984 | **3.661** | 3.663 | 1.167 | 1.120 | 1.252 | 3.545 | 1.297 | 0.672 |

### Speed (ms/step, lower is better)

| Architecture | capacity | correlated_key | in_context_regression | induction_heads | needle | overwrite_recall | permutation_S3 | permutation_S5 | selective_copy | xor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BatchMomentumLiquidLN | 12.2 | 12.2 | 14.2 | 3.4 | 11.3 | 12.8 | 24.4 | 13.4 | 7.1 | 74.0 |
| CrossAttnLoraLN | 3.4 | 3.5 | 3.6 | 0.9 | 3.1 | 3.5 | 6.2 | 5.3 | 2.0 | 22.7 |
| FactorizedLiquidLN | 11.6 | 11.6 | 13.1 | 3.2 | 10.8 | 13.0 | 22.4 | 12.6 | 7.0 | 64.1 |
| GDNLiquidLN | 4.6 | 4.5 | 4.8 | 1.2 | 4.2 | 4.7 | 8.3 | 5.5 | 2.6 | 35.1 |
| LiquidLinear | 24.2 | 24.4 | 34.0 | 4.7 | 23.7 | 28.8 | 49.8 | 0.8 | 14.8 | 139.6 |
| MomentumGDNLiquidLN | 4.3 | 4.3 | 4.4 | 1.2 | 3.8 | 4.3 | 7.9 | 5.2 | 2.4 | 32.0 |
| Rank1LiquidLN | 16.0 | 15.9 | 17.5 | 4.6 | 15.3 | 19.1 | 29.2 | 17.9 | 9.4 | 108.5 |
| RankRLiquidLN | 15.7 | 15.2 | 16.3 | 4.2 | 14.6 | 17.8 | 30.4 | 10.8 | 9.1 | 104.0 |
| SharedMomentumLiquidLN | 11.2 | 11.0 | 12.6 | 3.1 | 10.2 | 12.1 | 21.3 | 12.4 | 6.6 | 77.4 |
| StableGDNCondLiquidLN | 7.8 | 7.8 | 8.5 | 2.2 | 7.2 | 7.8 | 14.5 | 9.8 | 4.5 | 49.3 |
| StableLiquidLN | 12.8 | 13.0 | 14.5 | 3.6 | 11.9 | 15.0 | 24.8 | 13.8 | 7.6 | 85.0 |

### Parameter Counts

| Architecture | Params |
|---|---:|
| BatchMomentumLiquidLN | 125,670 |
| CrossAttnLoraLN | 134,918 |
| FactorizedLiquidLN | 136,991 |
| GDNLiquidLN | 139,667 |
| LiquidLinear | 154,863 |
| MomentumGDNLiquidLN | 139,674 |
| Rank1LiquidLN | 24,048 |
| RankRLiquidLN | 58,167 |
| SharedMomentumLiquidLN | 125,670 |
| StableGDNCondLiquidLN | 125,711 |
| StableLiquidLN | 125,663 |

## Static IO Tasks (LiquidMLP)

### Test Metric (higher is better)

| Architecture | fourier_d1_f3 | mod11_add | mod11_mul | parity_d20_k4 |
|---|---:|---:|---:|---:|
| BatchMomentumLiquidLN | 1.2723 | 4.9% | 8.2% | 49.7% |
| FactorizedLiquidLN | 1.1574 | **6.6%** | **16.4%** | **51.4%** |
| RankRLiquidLN | 1.1520 | 3.3% | 14.8% | **51.4%** |
| SharedMomentumLiquidLN | 1.1746 | 3.3% | 8.2% | 50.8% |
| StableLiquidLN | 1.2356 | 3.3% | 9.8% | 49.9% |

### Train Loss (lower is better)

| Architecture | fourier_d1_f3 | mod11_add | mod11_mul | parity_d20_k4 |
|---|---:|---:|---:|---:|
| BatchMomentumLiquidLN | 1.482 | 1.827 | 1.595 | 0.689 |
| FactorizedLiquidLN | 1.382 | 2.117 | 2.028 | 0.733 |
| RankRLiquidLN | 1.427 | 1.746 | 1.542 | 1.083 |
| SharedMomentumLiquidLN | 1.445 | 2.190 | 1.936 | 0.808 |
| StableLiquidLN | 1.514 | 2.150 | 2.042 | 0.697 |

### Parameter Counts

| Architecture | Params |
|---|---:|
| BatchMomentumLiquidLN | 19,499 |
| FactorizedLiquidLN | 21,737 |
| RankRLiquidLN | 4,717 |
| SharedMomentumLiquidLN | 19,499 |
| StableLiquidLN | 19,497 |
