import torch
import pytest
from llu.models import RankRLiquidLN

# =============================================================================
# HAPPY PATHS
# =============================================================================


@pytest.mark.parametrize("nonlinear", [True, False])
def test_when_initialization_equivalence_then_output_matches_linear_core(nonlinear):
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    model = RankRLiquidLN(
        in_features,
        out_features,
        rank=3,
        bias=True,
        dynamic_bias=True,
        nonlinear_hypernet=nonlinear,
    )

    device = model.linear_core.weight.device
    x = torch.randn(2, 3, in_features, device=device)

    # At step 1, adaptive path should be zeroed
    out = model(x)
    core_out = model.linear_core(x)
    assert torch.allclose(out, core_out, atol=1e-6)


def test_when_forward_with_various_dimensions_then_shapes_are_correct():
    in_features = 8
    out_features = 4
    model = RankRLiquidLN(in_features, out_features, rank=3, bias=True, dynamic_bias=True)
    device = model.linear_core.weight.device

    # 1D input
    x_1d = torch.randn(in_features, device=device)
    assert model(x_1d).shape == (out_features,)

    # 3D input
    x_3d = torch.randn(2, 3, in_features, device=device)
    assert model(x_3d).shape == (2, 3, out_features)


@pytest.mark.parametrize("nonlinear", [True, False])
def test_when_core_frozen_then_backward_only_updates_hypernetwork(nonlinear):
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    model = RankRLiquidLN(
        in_features,
        out_features,
        rank=3,
        bias=True,
        dynamic_bias=True,
        nonlinear_hypernet=nonlinear,
    )
    device = model.linear_core.weight.device

    # Enable grad for everything
    for p in model.parameters():
        p.requires_grad = True

    model.freeze_core()

    x = torch.randn(2, in_features, device=device)
    y = torch.randn(2, out_features, device=device)
    loss = (model(x) - y).pow(2).sum()
    loss.backward()

    # Core should not have gradients
    assert model.linear_core.weight.grad is None
    if model.linear_core.bias is not None:
        assert model.linear_core.bias.grad is None

    # Adaptive path should have gradients
    hyper_param = next(model.hypernetwork.parameters())
    assert hyper_param.grad is not None
    assert model.scale.grad is not None


@pytest.mark.parametrize("nonlinear", [True, False])
def test_when_hypernetwork_frozen_then_backward_only_updates_core(nonlinear):
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    model = RankRLiquidLN(
        in_features,
        out_features,
        rank=3,
        bias=True,
        dynamic_bias=True,
        nonlinear_hypernet=nonlinear,
    )
    device = model.linear_core.weight.device

    # Enable grad for everything
    for p in model.parameters():
        p.requires_grad = True

    model.freeze_hypernetwork()

    x = torch.randn(2, in_features, device=device)
    y = torch.randn(2, out_features, device=device)
    loss = (model(x) - y).pow(2).sum()
    loss.backward()

    # Core should have gradients
    assert model.linear_core.weight.grad is not None

    # Hypernetwork/scale should not have gradients
    hyper_param = next(model.hypernetwork.parameters())
    assert hyper_param.grad is None
    assert model.scale.grad is None


# =============================================================================
# UNHAPPY PATHS & EDGE CASES
# =============================================================================


def test_when_invalid_rank_then_raises_value_error():
    with pytest.raises(ValueError, match="rank must be >= 1"):
        RankRLiquidLN(8, 4, rank=0)


def test_when_input_shape_mismatched_then_raises_runtime_error():
    in_features = 8
    out_features = 4
    model = RankRLiquidLN(in_features, out_features, rank=3)
    device = model.linear_core.weight.device

    # Input has 10 features instead of 8
    x_invalid = torch.randn(2, 10, device=device)
    with pytest.raises(RuntimeError):
        model(x_invalid)


def test_when_extreme_input_values_then_numerical_stability_preserved():
    in_features = 8
    out_features = 4
    model = RankRLiquidLN(in_features, out_features, rank=3, bias=True, dynamic_bias=True)
    device = model.linear_core.weight.device

    # Large inputs that could cause overflow/NaNs in activations
    x_large = torch.zeros(2, in_features, device=device)
    x_large[0, :] = 1e4
    x_large[1, :] = -1e4

    out = model(x_large)
    assert out.shape == (2, out_features)
    assert not torch.isnan(out).any(), "NaN detected with extreme inputs"
    assert not torch.isinf(out).any(), "Inf detected with extreme inputs"
