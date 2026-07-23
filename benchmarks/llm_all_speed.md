# LLM Benchmark: All Intermediary LLNs × Parameterizations

Preset `tiny`, 50,000 tokens, seq_len 64, batch 4, lr 0.0003, max_steps 150.
Early stop patience: 30.

Lower ppl is better; higher acc is better.

| Tag | Variant | LLN | Param | Steps | ms/step | Train loss | Wiki ppl | LMB ppl | LMB acc | Time (s) | Early stop |
|-----|---------|-----|------:|------:|--------:|-----------:|---------:|--------:|--------:|---------:|:----------:|
| baseline_baseline_p_svd | baseline | - | 7,653,216 | 150 | 537.3 | 7.7066 | - | - | - | 80.6 | no |
| baseline_baseline_p_lora | baseline | - | 7,653,216 | 150 | 542.3 | 7.5021 | - | - | - | 81.3 | no |
| ours_StableLiquidLN_p_svd | ours | StableLiquidLN | 7,564,480 | 132 | 421.7 | 7.8025 | - | - | - | 55.7 | yes |
| ours_StableLiquidLN_p_lora | ours | StableLiquidLN | 8,545,968 | 150 | 507.7 | 7.1776 | - | - | - | 76.2 | no |
| ours_CrossAttnLoraLN_p_svd | ours | CrossAttnLoraLN | 7,747,520 | 150 | 449.9 | 7.3147 | - | - | - | 67.5 | no |
| ours_CrossAttnLoraLN_p_lora | ours | CrossAttnLoraLN | 7,747,504 | 131 | 451.6 | 7.2978 | - | - | - | 59.2 | yes |
| ours_SharedMomentumLiquidLN_p_svd | ours | SharedMomentumLiquidLN | 7,547,460 | 150 | 429.6 | 7.7875 | - | - | - | 64.4 | no |
| ours_SharedMomentumLiquidLN_p_lora | ours | SharedMomentumLiquidLN | 8,201,780 | 150 | 487.9 | 7.6097 | - | - | - | 73.2 | no |
| ours_BatchMomentumLiquidLN_p_svd | ours | BatchMomentumLiquidLN | 7,547,460 | 150 | 433.7 | 7.4515 | - | - | - | 65.1 | no |
| ours_BatchMomentumLiquidLN_p_lora | ours | BatchMomentumLiquidLN | 8,201,780 | 129 | 515.5 | 7.5819 | - | - | - | 66.5 | yes |
| ours_FactorizedLiquidLN_p_svd | ours | FactorizedLiquidLN | 7,547,456 | 150 | 426.4 | 7.3857 | - | - | - | 64.0 | no |
| ours_FactorizedLiquidLN_p_lora | ours | FactorizedLiquidLN | 8,234,800 | 150 | 487.0 | 7.5895 | - | - | - | 73.1 | no |
| ours_FactorizedBatchMomentumLiquidLN_p_svd | ours | FactorizedBatchMomentumLiquidLN | 7,547,460 | 150 | 429.0 | 7.6401 | - | - | - | 64.4 | no |
| ours_FactorizedBatchMomentumLiquidLN_p_lora | ours | FactorizedBatchMomentumLiquidLN | 8,234,804 | 150 | 492.0 | 7.5980 | - | - | - | 73.8 | no |

### Speed ranking (ms/step, lower is better)

1. **ours_StableLiquidLN_p_svd** — 421.7 ms/step (7,564,480 params)
2. **ours_FactorizedLiquidLN_p_svd** — 426.4 ms/step (7,547,456 params)
3. **ours_FactorizedBatchMomentumLiquidLN_p_svd** — 429.0 ms/step (7,547,460 params)
4. **ours_SharedMomentumLiquidLN_p_svd** — 429.6 ms/step (7,547,460 params)
5. **ours_BatchMomentumLiquidLN_p_svd** — 433.7 ms/step (7,547,460 params)
6. **ours_CrossAttnLoraLN_p_svd** — 449.9 ms/step (7,747,520 params)
7. **ours_CrossAttnLoraLN_p_lora** — 451.6 ms/step (7,747,504 params)
8. **ours_FactorizedLiquidLN_p_lora** — 487.0 ms/step (8,234,800 params)
9. **ours_SharedMomentumLiquidLN_p_lora** — 487.9 ms/step (8,201,780 params)
10. **ours_FactorizedBatchMomentumLiquidLN_p_lora** — 492.0 ms/step (8,234,804 params)
11. **ours_StableLiquidLN_p_lora** — 507.7 ms/step (8,545,968 params)
12. **ours_BatchMomentumLiquidLN_p_lora** — 515.5 ms/step (8,201,780 params)
13. **baseline_baseline_p_svd** — 537.3 ms/step (7,653,216 params)
14. **baseline_baseline_p_lora** — 542.3 ms/step (7,653,216 params)

Total wall time: 965s