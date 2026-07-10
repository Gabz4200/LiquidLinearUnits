import torch
import pytest
from llu.models import SharedMomentumLiquidLN

# =============================================================================
# HAPPY PATHS
# =============================================================================


def test_when_initialization_equivalence_then_output_matches_linear_core():
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    rank = 2
    model = SharedMomentumLiquidLN(in_features, out_features, rank=rank, bias=True)
    device = model.linear_core.weight.device

    # Check that buffers start at zero
    assert torch.all(model.a_raw == 0)
    assert torch.all(model.b_raw == 0)

    x = torch.randn(2, 3, in_features, device=device)
    out = model(x)
    core_out = model.linear_core(x)
    assert torch.allclose(out, core_out, atol=1e-6)


def test_when_forward_with_various_dimensions_then_shapes_are_correct():
    in_features = 8
    out_features = 4
    rank = 2
    model = SharedMomentumLiquidLN(in_features, out_features, rank=rank, bias=True)
    device = model.linear_core.weight.device

    # 1D input
    x_1d = torch.randn(in_features, device=device)
    assert model(x_1d).shape == (out_features,)

    # 3D input
    x_3d = torch.randn(2, 3, in_features, device=device)
    assert model(x_3d).shape == (2, 3, out_features)


def test_when_forward_pass_then_buffers_accumulate_momentum():
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    rank = 2
    model = SharedMomentumLiquidLN(in_features, out_features, rank=rank, bias=True)
    device = model.linear_core.weight.device

    x = torch.randn(2, 3, in_features, device=device)

    # First forward pass
    _ = model(x)
    a_raw_1 = model.a_raw.clone()
    assert not torch.all(a_raw_1 == 0)
    assert torch.all(
        model.b_raw == 0
    )  # b_raw stays zero because hypernetwork final layer has zeroed b-section

    # Second forward pass
    _ = model(x)
    a_raw_2 = model.a_raw.clone()
    assert not torch.allclose(a_raw_1, a_raw_2)


def test_when_optimizing_then_updates_core_hypernet_and_buffers():
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    rank = 2
    model = SharedMomentumLiquidLN(in_features, out_features, rank=rank, bias=True)
    device = model.linear_core.weight.device

    x = torch.randn(2, 3, in_features, device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    y = torch.randn(2, 3, out_features, device=device)

    loss = (model(x) - y).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Buffer b_raw should now update after forward pass (since weights have been updated and are no longer zero)
    _ = model(x)
    assert not torch.all(model.b_raw == 0)


def test_when_core_frozen_then_backward_only_updates_hypernetwork():
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    model = SharedMomentumLiquidLN(in_features, out_features, rank=2, bias=True)
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


def test_when_hypernetwork_frozen_then_backward_only_updates_core():
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    model = SharedMomentumLiquidLN(in_features, out_features, rank=2, bias=True)
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
        SharedMomentumLiquidLN(8, 4, rank=0)


def test_when_input_shape_mismatched_then_raises_runtime_error():
    in_features = 8
    out_features = 4
    model = SharedMomentumLiquidLN(in_features, out_features, rank=2)
    device = model.linear_core.weight.device

    # Input has 10 features instead of 8
    x_invalid = torch.randn(2, 10, device=device)
    with pytest.raises(RuntimeError):
        model(x_invalid)


def test_when_cond_shape_mismatched_then_raises_runtime_error():
    in_features = 8
    out_features = 4
    model = SharedMomentumLiquidLN(in_features, out_features, rank=2)
    device = model.linear_core.weight.device

    x = torch.randn(2, in_features, device=device)
    # Conditioning has incompatible feature dimensions
    cond_invalid = torch.randn(2, 10, device=device)
    with pytest.raises(RuntimeError):
        model(x, cond=cond_invalid)


def test_when_extreme_input_values_then_numerical_stability_preserved():
    in_features = 8
    out_features = 4
    model = SharedMomentumLiquidLN(in_features, out_features, rank=2, bias=True)
    device = model.linear_core.weight.device

    # Large inputs that could cause overflow/NaNs in activations
    x_large = torch.zeros(2, in_features, device=device)
    x_large[0, :] = 1e4
    x_large[1, :] = -1e4

    out = model(x_large)
    assert out.shape == (2, out_features)
    assert not torch.isnan(out).any(), "NaN detected with extreme inputs"
    assert not torch.isinf(out).any(), "Inf detected with extreme inputs"
