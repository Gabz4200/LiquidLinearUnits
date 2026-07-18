import torch
import pytest

from llu.models.llns.cross_attn_lora import CrossAttnLoraLN


# =============================================================================
# INITIALISATION EQUIVALENCE (adaptive path must be zero at step 1)
# =============================================================================


def test_when_initialization_equivalence_lora_then_output_matches_linear_core():
    torch.manual_seed(42)
    in_features, out_features = 8, 4
    model = CrossAttnLoraLN(in_features, out_features, rank=3, bias=True, dynamic_bias=True)
    device = model.linear_core.weight.device
    x = torch.randn(2, 3, in_features, device=device)
    assert torch.allclose(model(x), model.linear_core(x), atol=1e-6)


def test_when_initialization_equivalence_svd_then_output_matches_linear_core():
    torch.manual_seed(42)
    in_features, out_features = 8, 4
    model = CrossAttnLoraLN(in_features, out_features, rank=3, parameterization="svd")
    device = model.linear_core.weight.device
    x = torch.randn(2, 3, in_features, device=device)
    assert torch.allclose(model(x), model.linear_core(x), atol=1e-6)


# =============================================================================
# SHAPES
# =============================================================================


def test_when_forward_with_various_dimensions_then_shapes_are_correct():
    in_features, out_features = 8, 4
    model = CrossAttnLoraLN(in_features, out_features, rank=3, bias=True, dynamic_bias=True)
    device = model.linear_core.weight.device

    assert model(torch.randn(in_features, device=device)).shape == (out_features,)
    assert model(torch.randn(3, in_features, device=device)).shape == (3, out_features)
    assert model(torch.randn(2, 3, in_features, device=device)).shape == (2, 3, out_features)
    # Sequence source with a different cond feature dim exercises the
    # dimension-bridging projection (cond_dim != in_features).
    model_cd = CrossAttnLoraLN(in_features, out_features, rank=3, bias=True, cond_dim=6)
    x = torch.randn(2, 3, in_features, device=device)
    cond = torch.randn(2, 3, 6, device=device)
    assert model_cd(x, cond=cond).shape == (2, 3, out_features)


# =============================================================================
# CROSS-ATTENTION IS SEQUENCE-CONDITIONED
# =============================================================================


def test_when_factors_are_conditioned_on_sequence_then_cond_changes_output():
    torch.manual_seed(42)
    in_features, out_features, rank = 8, 4, 3
    model = CrossAttnLoraLN(in_features, out_features, rank=rank, bias=True)
    device = model.linear_core.weight.device

    x = torch.randn(2, 3, in_features, device=device)
    cond_a = torch.randn(2, 3, in_features, device=device)
    cond_b = torch.randn(2, 3, in_features, device=device)

    # At step 1 the adaptive path is zero regardless of cond.
    assert torch.allclose(model(x, cond=cond_a), model(x, cond=cond_b), atol=1e-6)

    # Train one step so the adaptive path becomes sequence-dependent.
    # factor_activation="none" keeps the factor magnitude informative so the
    # cond-dependent refinement is observable (a normalising activation would
    # collapse the (still small) factors to a near-constant direction).
    model = CrossAttnLoraLN(in_features, out_features, rank=rank, bias=True, factor_activation="none")
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    target = torch.randn(2, 3, out_features, device=device)
    optim.zero_grad()
    (model(x, cond=cond_a) - target).pow(2).sum().backward()
    optim.step()

    out_a = model(x, cond=cond_a)
    out_b = model(x, cond=cond_b)
    assert not torch.allclose(out_a, out_b, atol=1e-5)


# =============================================================================
# FREEZE SEMANTICS
# =============================================================================


def test_when_core_frozen_then_backward_only_updates_refiner():
    torch.manual_seed(42)
    in_features, out_features = 8, 4
    model = CrossAttnLoraLN(in_features, out_features, rank=3, bias=True, dynamic_bias=True)
    device = model.linear_core.weight.device

    for p in model.parameters():
        p.requires_grad_(True)
    model.freeze_core()

    x = torch.randn(2, in_features, device=device)
    y = torch.randn(2, out_features, device=device)
    (model(x) - y).pow(2).sum().backward()

    core_grads = [p.grad for p in model.linear_core.parameters()]
    assert all(g is None for g in core_grads)
    refiner_grads = [
        p.grad
        for name, p in model.named_parameters()
        if not name.startswith("linear_core")
    ]
    assert any(g is not None and g.abs().sum() > 0 for g in refiner_grads)


def test_when_hypernetwork_frozen_then_backward_only_updates_core():
    torch.manual_seed(42)
    in_features, out_features = 8, 4
    model = CrossAttnLoraLN(in_features, out_features, rank=3, bias=True, dynamic_bias=True)
    device = model.linear_core.weight.device

    for p in model.parameters():
        p.requires_grad_(True)
    model.freeze_hypernetwork()

    x = torch.randn(2, in_features, device=device)
    y = torch.randn(2, out_features, device=device)
    (model(x) - y).pow(2).sum().backward()

    core_grads = [p.grad for p in model.linear_core.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in core_grads)
    refiner_grads = [
        p.grad
        for name, p in model.named_parameters()
        if not name.startswith("linear_core")
    ]
    assert all(g is None for g in refiner_grads)


# =============================================================================
# ERROR PATHS / STABILITY
# =============================================================================


def test_when_invalid_rank_then_raises_value_error():
    with pytest.raises(ValueError, match="rank must be >= 1"):
        CrossAttnLoraLN(8, 4, rank=0)


def test_when_input_shape_mismatched_then_raises_runtime_error():
    in_features, out_features = 8, 4
    model = CrossAttnLoraLN(in_features, out_features, rank=3)
    device = model.linear_core.weight.device
    x_invalid = torch.randn(2, 10, device=device)
    with pytest.raises(RuntimeError):
        model(x_invalid)


def test_when_extreme_input_values_then_numerical_stability_preserved():
    in_features, out_features = 8, 4
    model = CrossAttnLoraLN(in_features, out_features, rank=3, bias=True, dynamic_bias=True)
    device = model.linear_core.weight.device
    x_large = torch.zeros(2, in_features, device=device)
    x_large[0, :] = 1e4
    x_large[1, :] = -1e4
    out = model(x_large)
    assert out.shape == (2, out_features)
    assert not torch.isnan(out).any(), "NaN detected with extreme inputs"


def test_when_svd_mode_then_g_parameter_exists_and_is_learnable():
    model = CrossAttnLoraLN(8, 4, rank=3, parameterization="svd")
    assert isinstance(model.g, torch.nn.Parameter)
    assert model.g.requires_grad is True
    assert model.g.shape == (3,)


# =============================================================================
# GRADIENT CORRECTNESS (double precision, gradcheck)
# =============================================================================


def test_when_gradcheck_then_analytical_matches_numerical():
    torch.manual_seed(42)
    in_features, out_features = 4, 2
    model = CrossAttnLoraLN(
        in_features, out_features, rank=2, bias=True, dynamic_bias=True, attn_dim=8, attn_heads=2
    ).double()
    device = model.linear_core.weight.device
    x = torch.randn(2, in_features, dtype=torch.float64, device=device, requires_grad=True)
    # check_undefined_grad=False: zero-init residual means some grads are undefined at step 1.
    assert torch.autograd.gradcheck(model, (x,), eps=1e-6, atol=1e-4, check_undefined_grad=False)
