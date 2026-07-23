# LLM Benchmark: LoRA Rank Sweep vs SVD

Preset `tiny`, 200,000 tokens, seq_len 64, batch 4, lr 0.0003, max_steps 500.
Early stop patience: 50.

All LoRA configs use `alpha=rank` (scale=1.0) except `lora_r4_a1` (alpha=1, scale=0.25).
Lower ppl is better; higher acc is better.

| Tag | Variant | LLN | Param | Rank | α | Steps | ms/step | Train loss | Wiki ppl | LMB ppl | LMB acc | Time (s) |
|-----|---------|-----|------:|-----:|--:|------:|--------:|-----------:|---------:|--------:|--------:|---------:|
| baseline_baseline_svd_r4 | baseline | - | 7,653,216 | 4 | 1.0 | 338 | 534.8 | 7.3660 | 6808.1165 | 167093.3001 | 0.0000 | 180.8 |
| ours_StableLiquidLN_lora_r1_a1 | ours | StableLiquidLN | 7,751,792 | 1 | 1.0 | 294 | 443.2 | 7.3461 | 7199.4716 | 153289.2834 | 0.0000 | 130.3 |
| ours_StableLiquidLN_lora_r2_a2 | ours | StableLiquidLN | 7,959,152 | 2 | 2.0 | 267 | 467.3 | 7.3015 | 6466.2013 | 164827.4764 | 0.0000 | 124.8 |
| ours_StableLiquidLN_lora_r4_a4 | ours | StableLiquidLN | 8,545,968 | 4 | 4.0 | 267 | 495.1 | 7.6488 | 6846.7693 | 162710.8463 | 0.0000 | 132.2 |
| ours_StableLiquidLN_lora_r8_a8 | ours | StableLiquidLN | 10,211,120 | 8 | 8.0 | 265 | 588.9 | 7.6640 | 6337.4114 | 191813.2310 | 0.0000 | 156.1 |
| ours_StableLiquidLN_lora_r16_a16 | ours | StableLiquidLN | 18,161,968 | 16 | 16.0 | 322 | 874.4 | 7.2305 | 6364.4153 | 166374.6252 | 0.0000 | 281.6 |
| ours_StableLiquidLN_lora_r4_a1.0 | ours | StableLiquidLN | 8,545,968 | 4 | 1.0 | 187 | 506.0 | 7.5108 | 6675.3838 | 137876.3633 | 0.0000 | 94.6 |
| ours_StableLiquidLN_svd_r4 | ours | StableLiquidLN | 7,564,480 | 4 | 1.0 | 264 | 456.9 | 7.3940 | 6191.7390 | 146795.9537 | 0.0000 | 120.6 |

### Speed ranking (ms/step, lower is better)

1. **ours_StableLiquidLN_lora_r1_a1** — 443.2 ms/step (7,751,792 params, lora r=1)
2. **ours_StableLiquidLN_svd_r4** — 456.9 ms/step (7,564,480 params, svd r=4)
3. **ours_StableLiquidLN_lora_r2_a2** — 467.3 ms/step (7,959,152 params, lora r=2)
4. **ours_StableLiquidLN_lora_r4_a4** — 495.1 ms/step (8,545,968 params, lora r=4)
5. **ours_StableLiquidLN_lora_r4_a1.0** — 506.0 ms/step (8,545,968 params, lora r=4)
6. **baseline_baseline_svd_r4** — 534.8 ms/step (7,653,216 params, svd r=4)
7. **ours_StableLiquidLN_lora_r8_a8** — 588.9 ms/step (10,211,120 params, lora r=8)
8. **ours_StableLiquidLN_lora_r16_a16** — 874.4 ms/step (18,161,968 params, lora r=16)

### Quality ranking (LAMBADA ppl, lower is better)

1. **ours_StableLiquidLN_lora_r4_a1.0** — LMB ppl 137876.4 (lora r=4, α=1.0)
2. **ours_StableLiquidLN_svd_r4** — LMB ppl 146796.0 (svd r=4, α=1.0)
3. **ours_StableLiquidLN_lora_r1_a1** — LMB ppl 153289.3 (lora r=1, α=1.0)
4. **ours_StableLiquidLN_lora_r4_a4** — LMB ppl 162710.8 (lora r=4, α=4.0)
5. **ours_StableLiquidLN_lora_r2_a2** — LMB ppl 164827.5 (lora r=2, α=2.0)
6. **ours_StableLiquidLN_lora_r16_a16** — LMB ppl 166374.6 (lora r=16, α=16.0)
7. **baseline_baseline_svd_r4** — LMB ppl 167093.3 (svd r=4, α=1.0)
8. **ours_StableLiquidLN_lora_r8_a8** — LMB ppl 191813.2 (lora r=8, α=8.0)

### Quality ranking (Wiki ppl, lower is better)

1. **ours_StableLiquidLN_svd_r4** — Wiki ppl 6191.7 (svd r=4, α=1.0)
2. **ours_StableLiquidLN_lora_r8_a8** — Wiki ppl 6337.4 (lora r=8, α=8.0)
3. **ours_StableLiquidLN_lora_r16_a16** — Wiki ppl 6364.4 (lora r=16, α=16.0)
4. **ours_StableLiquidLN_lora_r2_a2** — Wiki ppl 6466.2 (lora r=2, α=2.0)
5. **ours_StableLiquidLN_lora_r4_a1.0** — Wiki ppl 6675.4 (lora r=4, α=1.0)
6. **baseline_baseline_svd_r4** — Wiki ppl 6808.1 (svd r=4, α=1.0)
7. **ours_StableLiquidLN_lora_r4_a4** — Wiki ppl 6846.8 (lora r=4, α=4.0)
8. **ours_StableLiquidLN_lora_r1_a1** — Wiki ppl 7199.5 (lora r=1, α=1.0)

### Efficiency ranking (LMB ppl / params, lower = better quality per param)

1. **ours_StableLiquidLN_lora_r16_a16** — 0.0092 ppl/param (18,161,968 params, LMB 166374.6)
2. **ours_StableLiquidLN_lora_r4_a1.0** — 0.0161 ppl/param (8,545,968 params, LMB 137876.4)
3. **ours_StableLiquidLN_lora_r8_a8** — 0.0188 ppl/param (10,211,120 params, LMB 191813.2)
4. **ours_StableLiquidLN_lora_r4_a4** — 0.0190 ppl/param (8,545,968 params, LMB 162710.8)
5. **ours_StableLiquidLN_svd_r4** — 0.0194 ppl/param (7,564,480 params, LMB 146796.0)
6. **ours_StableLiquidLN_lora_r1_a1** — 0.0198 ppl/param (7,751,792 params, LMB 153289.3)
7. **ours_StableLiquidLN_lora_r2_a2** — 0.0207 ppl/param (7,959,152 params, LMB 164827.5)
8. **baseline_baseline_svd_r4** — 0.0218 ppl/param (7,653,216 params, LMB 167093.3)

### LoRA rank scaling (alpha=rank, scale=1.0)

| Rank | Params | ms/step | LMB ppl | Wiki ppl | Train loss |
|-----:|-------:|--------:|--------:|---------:|-----------:|
| 1 | 7,751,792 | 443.2 | 153289.2834 | 7199.4716 | 7.3461 |
| 2 | 7,959,152 | 467.3 | 164827.4764 | 6466.2013 | 7.3015 |
| 4 | 8,545,968 | 495.1 | 162710.8463 | 6846.7693 | 7.6488 |
| 8 | 10,211,120 | 588.9 | 191813.2310 | 6337.4114 | 7.6640 |
| 16 | 18,161,968 | 874.4 | 166374.6252 | 6364.4153 | 7.2305 |

### Alpha scaling (rank=4, varying alpha)

| α | Scale (α/r) | Params | LMB ppl | Wiki ppl | Train loss |
|--:|------------:|-------:|--------:|---------:|-----------:|
| 1.0 | 0.25 | 8,545,968 | 137876.3633 | 6675.3838 | 7.5108 |
| 4.0 | 1.00 | 8,545,968 | 162710.8463 | 6846.7693 | 7.6488 |

Total wall time: 1221s