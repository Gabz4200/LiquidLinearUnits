import torch
import pytest
from llu.models.llns.gdn_liquid import GDNLiquidLN

# =============================================================================
# MOCK CACHE FOR STATEFUL TESTS
# =============================================================================


class MockCache:
    def __init__(self):
        self.caches = {}

    def __len__(self):
        return 999

    def __getitem__(self, idx):
        return self.caches.get(idx, None)

    def update(self, layer_idx, **kwargs):
        self.caches[layer_idx] = kwargs


# =============================================================================
# HAPPY PATHS
# =============================================================================


def test_when_initialization_equivalence_then_output_matches_linear_core():
    torch.manual_seed(42)
    in_features = 16
    out_features = 8
    model = GDNLiquidLN(
        in_features,
        out_features,
        rank=2,
        head_dim=8,
        num_heads=2,
        bias=True,
        dynamic_bias=True,
    )

    device = model.linear_core.weight.device
    x = torch.randn(2, 3, in_features, device=device)

    # At initialization (step 1), the adaptive path should be zeroed
    out = model(x)
    core_out = model.linear_core(x)

    assert torch.allclose(out, core_out, atol=1e-6)


def test_when_forward_with_various_dimensions_then_shapes_are_correct():
    in_features = 16
    out_features = 8
    model = GDNLiquidLN(
        in_features,
        out_features,
        rank=2,
        head_dim=8,
        num_heads=2,
        bias=True,
    )

    # 1D Input
    x_1d = torch.randn(in_features)
    assert model(x_1d).shape == (out_features,)

    # 2D Input
    x_2d = torch.randn(3, in_features)
    assert model(x_2d).shape == (3, out_features)

    # 3D Input
    x_3d = torch.randn(2, 3, in_features)
    assert model(x_3d).shape == (2, 3, out_features)


def test_when_conditioning_provided_then_uses_cond_features():
    torch.manual_seed(42)
    in_features = 16
    out_features = 8
    model = GDNLiquidLN(
        in_features,
        out_features,
        rank=2,
        head_dim=8,
        num_heads=2,
        bias=True,
    )
    device = model.linear_core.weight.device

    x = torch.randn(2, 3, in_features, device=device)
    cond1 = torch.randn(2, 3, in_features, device=device)
    cond2 = torch.randn(2, 3, in_features, device=device)

    # Run optimizer step so the adaptive path is non-zero
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    y = torch.randn(2, 3, out_features, device=device)
    loss = (model(x, cond=cond1) - y).pow(2).sum()
    loss.backward()
    optimizer.step()

    out_cond1 = model(x, cond=cond1)
    out_cond2 = model(x, cond=cond2)

    # Outputs should differ because they are conditioned differently
    assert not torch.allclose(out_cond1, out_cond2)


# =============================================================================
# CACHE & MASK HANDLING
# =============================================================================


def test_when_incremental_decoding_with_cache_then_updates_cache_correctly():
    in_features = 16
    out_features = 8
    model = GDNLiquidLN(
        in_features,
        out_features,
        rank=2,
        head_dim=8,
        num_heads=2,
        use_short_conv=True,
        layer_idx=0,
    )

    cache = MockCache()

    # Step 1: Forward pass with sequence length 1
    x1 = torch.randn(2, 1, in_features)
    out1, cache = model(x1, past_key_values=cache, use_cache=True)

    assert out1.shape == (2, 1, out_features)
    assert 0 in cache.caches

    state1 = cache.caches[0]
    assert "recurrent_state" in state1
    assert "conv_state" in state1
    assert state1["recurrent_state"] is not None

    # Step 2: Forward pass with next token
    x2 = torch.randn(2, 1, in_features)
    out2, cache = model(x2, past_key_values=cache, use_cache=True)

    assert out2.shape == (2, 1, out_features)
    assert cache.caches[0]["recurrent_state"] is not None


def test_when_attention_mask_provided_then_succeeds():
    in_features = 16
    out_features = 8
    model = GDNLiquidLN(
        in_features,
        out_features,
        rank=2,
        head_dim=8,
        num_heads=2,
        use_short_conv=False,
    )

    x = torch.randn(2, 5, in_features)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[0, 3:] = 0  # Pad elements

    out = model(x, attention_mask=mask)
    assert out.shape == (2, 5, out_features)


# =============================================================================
# FREEZE VERIFICATION
# =============================================================================


def test_when_core_frozen_then_backward_only_updates_hypernetwork():
    torch.manual_seed(42)
    in_features = 16
    out_features = 8
    model = GDNLiquidLN(
        in_features,
        out_features,
        rank=2,
        head_dim=8,
        num_heads=2,
        bias=True,
        dynamic_bias=True,
    )

    model.freeze_core()

    # Core should not require grad, but adaptive path parameters should
    assert not model.linear_core.weight.requires_grad
    assert model.proj_out.weight.requires_grad

    x = torch.randn(2, 3, in_features)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert model.linear_core.weight.grad is None
    assert model.proj_out.weight.grad is not None


def test_when_hypernetwork_frozen_then_backward_only_updates_core():
    torch.manual_seed(42)
    in_features = 16
    out_features = 8
    model = GDNLiquidLN(
        in_features,
        out_features,
        rank=2,
        head_dim=8,
        num_heads=2,
        bias=True,
        dynamic_bias=True,
    )

    model.freeze_hypernetwork()

    # Core should require grad, but hypernetwork should not
    assert model.linear_core.weight.requires_grad
    assert not model.proj_out.weight.requires_grad

    x = torch.randn(2, 3, in_features)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert model.linear_core.weight.grad is not None
    assert model.proj_out.weight.grad is None


# =============================================================================
# UNHAPPY PATHS & EDGE CASES
# =============================================================================


def test_when_invalid_rank_then_raises_value_error():
    with pytest.raises(ValueError, match="rank must be >= 1"):
        GDNLiquidLN(16, 8, rank=0)


def test_when_input_shape_mismatched_then_raises_runtime_error():
    in_features = 16
    out_features = 8
    model = GDNLiquidLN(
        in_features,
        out_features,
        rank=2,
        head_dim=8,
        num_heads=2,
    )

    # Input has 10 features, but model expects 16
    x_invalid = torch.randn(2, 3, 10)
    with pytest.raises(RuntimeError):
        model(x_invalid)


def test_when_extreme_input_values_then_numerical_stability_preserved():
    in_features = 16
    out_features = 8
    model = GDNLiquidLN(
        in_features,
        out_features,
        rank=2,
        head_dim=8,
        num_heads=2,
    )

    # Test with very large inputs
    x_large = torch.randn(2, 3, in_features) * 1e4
    out = model(x_large)

    assert not torch.isnan(out).any(), "NaN detected in output with extreme inputs"
    assert not torch.isinf(out).any(), "Inf detected in output with extreme inputs"
