# AGENTS.md

## Project

LiquidLinearUnits — PyTorch library of Liquid Linear Units (input-adaptive linear transformations) for transformer models. Package name `llu`, version 0.1.0.

## Key commands

- Install deps: `uv sync` (requires Linux x86_64, Python >=3.11,<3.14)
- Run all tests: `python -m pytest` (tests in `tests/`)
- Run single test: `python -m pytest tests/test_stable_liquid.py::test_when_input_shape_mismatched_then_raises_runtime_error`
- Mark slow tests: `pytest -m slow`
- Typecheck: `pyrefly check` (configured in `pyrefly.toml`, preset `strict`)
- Lint: `ruff check .` / `ruff format .`
- LSP: uses **pyrefly**, not pyright (see `.lsp.json` — pyright is explicitly disabled)

## Architecture

Every model is a Transformer block (SWA + SwiGLU MLP) where **every** `nn.Linear` is replaced by a Liquid Linear Unit (LLU).

Package layout:
- `llu/models/llns/` — 6 LLU variant modules (`stable_liquid.py`, `factorized_liquid.py`, `rankr_liquid.py`, `gdn_liquid.py`, `cross_attn_lora.py`, `engram_retrieved_lora.py`) + shared `base.py` / `utils.py`
- `llu/models/gdn2/` — GatedDeltaNet2 (CPU chunk kernel + GPU backend); `gdn2.py` excludes from pyrefly analysis
- `llu/models/engram/` — DeepSeek Engram conditional N-gram memory module (`Engram`, `EngramConfig`, `EngramEmbeddingStore`, `NgramHashMapping`, `CompressedTokenizer`)
- `llu/models/liquid_model.py` — `LiquidTransformer` + `build_model` + `ARCH_FACTORIES`
- `llu/models/liquid_llm.py` — LLM-scale models (SLM)
- `llu/models/mlp_model.py` — `LiquidMLP` + `IO_LLN_REGISTRY`
- `llu/__init__.py` — public API exports

Public API from top-level `llu`: all 6 LLN variants (`StableLiquidLN`, `FactorizedLiquidLN`, `RankRLiquidLN`, `GDNLiquidLN`, `CrossAttnLoraLN`, `EngramRetrievedLoraLN`), `GatedDeltaNet2`, `Engram`, `EngramConfig`, `LiquidTransformer`, `build_model`, `ARCH_FACTORIES`, `LiquidMLP`, `IO_LLN_REGISTRY`.

## Important constraints

- **PyTorch nightly CPU wheels** (not stable) — `uv.lock` pins `torch>=2.14.0.dev20260707` from `https://download.pytorch.org/whl/nightly/cpu`. Do not switch to stable PyPItorch without updating the index.
- **Linux x86_64 only** (`uv.required-environments` in pyproject.toml).
- **No GPU CI** — no `.github/` directory exists. Benchmarks run CPU-only on i5-8250U.
- **`parameterization` flag** (`svd` vs `lora`, `alpha`, `rank`, `scale`) is a key axis on most variants. `CrossAttnLoraLN` always uses LoRA.
- **`cond` port** exists on `StableLiquidLN`, `FactorizedLiquidLN`, `GDNLiquidLN`, `CrossAttnLoraLN`, and `EngramRetrievedLoraLN`; `RankRLiquidLN` has none, which is why it is excluded from LLM intermediary comparisons.
- **Engram offloading & routing**: `Engram` supports `"auto"`, `"cpu"`, `"disk"` (`np.memmap` binary file on SSD in 64k chunks), and `"cuda"` storage backends; routing supports `"cond"`, `"additive"`, and `"both"`. Flags `--use_engram`, `--engram_mode`, and `--engram_storage` are exposed in CLI scripts.
- **Zero-init** on adaptive path: at step 1 the adaptive output is identity (HypeLoRA calibration insight).
- **2026-07 pruning pass**: removed `LiquidLinear`, `Rank1LiquidLN`, the momentum family (`SharedMomentumLiquidLN`, `BatchMomentumLiquidLN`, `FactorizedBatchMomentumLiquidLN`, `MomentumGDNLiquidLN`), and the GDN-cond aliases (`StableGDNCondLiquidLN`, `FactorizedGDNCondLiquidLN`, `FactBatchMomGDNCondLiquidLN`). Rationale in README "Removed architectures".
- **`_freeze` mixin** from `utils.py` supports `freeze` param on all LLUs.

## Testing quirks

- `pytest.ini_options` sets `pythonpath = ["."]` — imports work from repo root.
- `testpaths = ["tests"]` — test discovery is limited to `tests/`.
- 148 tests collected across ~16s (includes `tests/test_engram.py` for tokenizer compression, hash mapping, CPU/disk offloading, and model conditioning).
- `test_gdn2_cpu_parity.py` requires CPU-only environment.
- `test_compatibility.py` may need extra dependencies not in base `uv.lock`.

## Style / workflow

- **pyrefly** is the active LSP/typechecker (not pyright). `.lsp.json` disables pyright.
- pyrefly `preset = "strict"` but with `bad-assignment = false`, `implicit-any = false`, `missing-override-decorator = false` — research-code-permissive on those, strict on instantiation/keyword errors.
- `search-path = ["llu"]` in pyrefly — type checking starts from `llu/`.
- ruff line-length = 100, target py311.
- No pre-commit config, no `Makefile`, no `setup.py` — all config is in `pyproject.toml` + `pyrefly.toml`.

## Key insight for agents

The repo is a research project, not a library with stable API. New LLU variants are added by creating a module in `llu/models/llns/`, exporting from `llu/models/llns/__init__.py`, and optionally from `llu/__init__.py`. The `parameterization` flag (svd/lora) is the primary variation axis — do not add new variants without supporting both parametrizations.