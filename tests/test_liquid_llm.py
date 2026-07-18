r"""Tests for the LLM-scale models in ``llu.models.liquid_llm``.

These ground the intermediary-LLN comparison: every LLN in ``LLN_REGISTRY``
must build an ``LiquidGDNCondLLM`` and run a forward pass at the tiny CPU
preset, and the parameter budgets of ``ours`` vs ``baseline`` must be in the
same ballpark.
"""

import torch

from llu.models.liquid_llm import (
    build_llm,
    num_params,
    LLN_REGISTRY,
    LiquidGDNCondLLM,
    GDN2BaselineLLM,
)


def test_when_build_ours_with_each_registry_lln_then_forward_shape_and_finite():
    torch.manual_seed(0)
    B, T = 2, 16
    idx = torch.randint(0, 50257, (B, T))
    for name in LLN_REGISTRY:
        model = build_llm("ours", "tiny", lln=name, parameterization="svd")
        assert isinstance(model, LiquidGDNCondLLM)
        out = model(idx)
        assert out.shape == (B, T, 50257), name
        assert torch.isfinite(out).all(), f"{name}: non-finite output"


def test_when_build_baseline_then_forward_shape_and_finite():
    torch.manual_seed(0)
    idx = torch.randint(0, 50257, (2, 16))
    model = build_llm("baseline", "tiny", parameterization="svd")
    assert isinstance(model, GDN2BaselineLLM)
    out = model(idx)
    assert out.shape == (2, 16, 50257)
    assert torch.isfinite(out).all()


def test_when_build_ours_init_ce_is_reasonable():
    # Embedding is scaled to 1/sqrt(n_embd); init cross-entropy ~= ln(vocab).
    torch.manual_seed(0)
    idx = torch.randint(0, 50257, (2, 16))
    model = build_llm("ours", "tiny", lln="StableLiquidLN", parameterization="svd")
    with torch.no_grad():
        logits = model(idx)
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), idx.reshape(-1)
    )
    # ln(50257) ~= 10.82; allow generous slack for the tiny init.
    assert ce.item() < 12.0, f"init CE too high: {ce.item()}"


def test_when_compare_presets_then_ours_and_baseline_share_budget():
    # At the tiny preset the two variants are tuned to land near the same
    # parameter count (baseline uses more layers to compensate for no SWA/LLN).
    ours = num_params(build_llm("ours", "tiny", lln="StableLiquidLN"))
    base = num_params(build_llm("baseline", "tiny"))
    ratio = max(ours, base) / min(ours, base)
    assert ratio < 1.6, f"parameter budgets diverge too far: ours={ours}, base={base}"


def test_when_unknown_lln_then_build_raises_keyerror():
    try:
        build_llm("ours", "tiny", lln="DoesNotExist")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown LLN")
