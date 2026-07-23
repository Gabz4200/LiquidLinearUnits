# LLM Benchmark: Scaled Intermediary LLN Comparison

Preset `scaled` (n_embd=192, 4 layers ours / 8 layers baseline), 100,000 tokens, seq_len 64, batch 4, lr 0.0003, max_steps 300.
Early stop patience: 50.

Lower ppl is better; higher acc is better.

| # | LLN | Param | Steps | ms/step | Train loss (best) | Wiki ppl | LMB ppl | LMB acc | Time (s) |
|---|-----|------:|------:|--------:|-------------------:|---------:|--------:|--------:|---------:|
| 1 | **StableLiquidLN** (svd) | 14,689,824 | 211 | 837 | 7.1362 (6.8089) | 6946.5328 | 122392.7330 | 0.0000 | 176 |
| 2 | **CrossAttnLoraLN** (svd) | 15,139,104 | 300 | 874 | 7.0574 (6.2104) | 6489.4744 | 163277.5554 | 0.0000 | 262 |
| 3 | **FactorizedLiquidLN** (svd) | 14,588,960 | 252 | 830 | 7.2844 (6.2637) | 7120.5380 | 186572.5207 | 0.0000 | 209 |
| 4 | **SharedMomentumLiquidLN** (svd) | 14,588,968 | 300 | 827 | 7.3063 (6.3583) | 7093.4824 | 208729.4121 | 0.0000 | 248 |
| 5 | **-** (svd) | 15,128,704 | 252 | 1107 | 6.7673 (6.3473) | 8375.9818 | 214622.2712 | 0.0000 | 279 |
| 6 | **FactorizedBatchMomentumLiquidLN** (svd) | 14,588,968 | 276 | 824 | 7.6882 (6.5919) | 6592.8588 | 231394.9890 | 0.0000 | 227 |

Total wall time: 1403s (23.4 min)