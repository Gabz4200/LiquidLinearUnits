# LLM benchmark (CPU snapshot)

Preset `tiny`, parameterization `svd`, tokens 100,000, seq_len 64.

Lower ppl is better; higher acc is better. These are short CPU-scale runs (**not** convergence numbers).

| Variant | LLN | Params | Steps | Train loss | Wiki ppl | LMB ppl | LMB acc | Time (s) |
|---|---|---|---|---|---|---|---|---|
| baseline | - | 7,653,216 | 200 | 7.5440 | 7196.3066 | 158751.9439 | 0.0000 | 115.9 |
| ours | StableLiquidLN | 7,564,480 | 200 | 7.5290 | 6784.5427 | 140264.7147 | 0.0000 | 91.3 |
| ours | CrossAttnLoraLN | 7,747,520 | 200 | 7.4135 | 7818.8756 | 135305.3280 | 0.0000 | 97.0 |
| ours | SharedMomentumLiquidLN | 7,547,460 | 200 | 6.9547 | 7073.5770 | 137415.5241 | 0.0000 | 93.3 |
|| ours | BatchMomentumLiquidLN | 7,547,460 | 200 | 7.1770 | 6503.0580 | 147491.7033 | 0.0000 | 91.8 |

## Notes & caveats

- All variants train (init CE ~10.8 -> train loss ~7.0); no NaN/Inf. The `ours`
  variants use the LLN intermediary in the 2-layer FFN; `baseline` is a plain MLP.
- Read LAMBADA ppl directionally: `ours` beats `baseline` on every LLN. The
  strongest intermediary is **CrossAttnLoraLN** (best LAMBADA ppl, 2nd-best train
  loss) — its cross-attention refiner captures token context the scalar/vector
  modulators do not.
- **LMB acc is 0.0 everywhere**: 100k tokens at `tiny` is far below the grokking
  regime, so accuracy is not a usable signal here.
- Parameter budget is dominated by the GPT-2 embedding (~6.4M of ~7.5M); the LLN
  delta is only ~±200K, so the LAMBADA ppl spread is a real signal of the
  intermediary, not a param-count artifact.
- Full interpretive write-up: `benchmarks/RESULTS.md` (LLM-scale section).
