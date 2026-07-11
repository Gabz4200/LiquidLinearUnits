import torch
import pytest
from llu.models import MomentumGDNLiquidLN

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
    model = MomentumGDNLiquidLN(
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

    # Step 1 forward pass
    out = model(x)

    # Expected output: linear_core(x) + dynamic_bias(gdn_out)
    # Since b_new is zero at step 1 and b_raw is zero, adaptive path must contribute nothing
    core_out = model.linear_core(x)

    # Run gdn2 manually to verify dynamic bias contribution
    h_in = torch.nn.functional.rms_norm(x, (in_features,)) if model.normalize_input else x
    h_in_3d = h_in.unsqueeze(1) if len(h_in.shape) == 2 else h_in.flatten(0, -3)
    gdn_out, _, _ = model.gdn2(h_in_3d)
    bias_out = model.bias_dynamic(gdn_out).view(*x.shape[:-1], out_features)
    expected_out = core_out + bias_out

    assert torch.allclose(out, expected_out, atol=1e-6)


def test_when_forward_with_various_dimensions_then_shapes_are_correct():
    in_features = 16
    out_features = 8
    model = MomentumGDNLiquidLN(
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
    model = MomentumGDNLiquidLN(
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

    assert not torch.allclose(out_cond1, out_cond2)


def test_when_forward_pass_then_buffers_accumulate_momentum():
    torch.manual_seed(42)
    in_features = 16
    out_features = 8
    rank = 2
    model = MomentumGDNLiquidLN(
        in_features,
        out_features,
        rank=rank,
        head_dim=8,
        num_heads=2,
        bias=True,
    )
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
    in_features = 16
    out_features = 8
    rank = 2
    model = MomentumGDNLiquidLN(
        in_features,
        out_features,
        rank=rank,
        head_dim=8,
        num_heads=2,
        bias=True,
        factor_activation="tanh",
        initial_decay_rate=0.8,
    )
    device = model.linear_core.weight.device

    # Manually populate buffers so that the adaptive path is active and scale-sensitive
    model.a_raw.data.normal_()
    model.b_raw.data.normal_()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 3, in_features, device=device)
    target = torch.randn(2, 3, out_features, device=device)

    model.set_decay_rate_learnable(True)
    decay_rate_init = model.decay_rate.item()

    # Step 1: Forward & Backward
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    optimizer.step()

    # Decay rate parameter should be updated by optimizer
    assert model.decay_rate.item() != decay_rate_init

    # We run another forward pass to populate b_raw
    _ = model(x)
    assert not torch.all(model.b_raw == 0)


# =============================================================================
# CACHE & MASK HANDLING
# =============================================================================


def test_when_incremental_decoding_with_cache_then_updates_cache_correctly():
    in_features = 16
    out_features = 8
    model = MomentumGDNLiquidLN(
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
    model = MomentumGDNLiquidLN(
        in_features,
        out_features,
        rank=2,
        head_dim=8,
        num_heads=2,
        use_short_conv=False,
    )

    x = torch.randn(2, 5, in_features)
    mask = torch.ones(2, 5, dtype=torch.bool)
    out = model(x, attention_mask=mask)

    assert out.shape == (2, 5, out_features)


# =============================================================================
# FREEZE VERIFICATION
# =============================================================================


def test_when_core_frozen_then_backward_only_updates_hypernetwork():
    torch.manual_seed(42)
    in_features = 16
    out_features = 8
    model = MomentumGDNLiquidLN(
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
    assert model.scale.grad is not None


def test_when_hypernetwork_frozen_then_backward_only_updates_core():
    torch.manual_seed(42)
    in_features = 16
    out_features = 8
    model = MomentumGDNLiquidLN(
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
    assert model.scale.grad is None


# =============================================================================
# UNHAPPY PATHS & EDGE CASES
# =============================================================================


def test_when_invalid_rank_then_raises_value_error():
    with pytest.raises(ValueError, match="rank must be >= 1"):
        MomentumGDNLiquidLN(16, 8, rank=0)


def test_when_input_shape_mismatched_then_raises_runtime_error():
    in_features = 16
    out_features = 8
    model = MomentumGDNLiquidLN(
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
    model = MomentumGDNLiquidLN(
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
