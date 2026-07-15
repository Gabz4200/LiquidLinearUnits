import torch
import pytest
from typing import Any, cast
from llu.models.gdn2.gdn2 import GatedDeltaNet2

# =============================================================================
# HAPPY PATHS
# =============================================================================


def test_when_valid_forward_without_mask_then_succeeds():
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, use_short_conv=True)
    x = torch.randn(2, 5, 64)
    out, _, _ = layer(x)
    assert out.shape == (2, 5, 64)

    # Verify gradient flow
    loss = out.sum()
    loss.backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"


def test_when_valid_forward_with_mask_then_succeeds():
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, use_short_conv=True)
    x = torch.randn(2, 5, 64)
    mask = torch.ones(2, 5)
    mask[0, -2:] = 0  # pad last two positions of first sequence

    out, _, _ = layer(x, attention_mask=mask)
    assert out.shape == (2, 5, 64)

    loss = out.sum()
    loss.backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"


def test_when_forward_without_short_conv_then_succeeds():
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, use_short_conv=False)
    x = torch.randn(2, 5, 64)
    out, _, _ = layer(x)
    assert out.shape == (2, 5, 64)

    loss = out.sum()
    loss.backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, f"Gradient for {name} is None"


class MockCache:
    def __init__(self):
        self.caches = {}

    def __len__(self):
        return 999

    def __getitem__(self, idx):
        return self.caches.get(idx, None)

    def update(self, layer_idx, **kwargs):
        self.caches[layer_idx] = kwargs


def test_when_incremental_decoding_with_cache_then_updates_cache_and_state():
    layer = GatedDeltaNet2(
        hidden_size=64, num_heads=2, head_dim=32, use_short_conv=True, layer_idx=0
    )
    cache = MockCache()

    # Step 1: Forward pass with sequence length 1
    x1 = torch.randn(2, 1, 64)
    out1, _, _ = layer(x1, past_key_values=cache, use_cache=True)
    assert out1.shape == (2, 1, 64)
    assert 0 in cache.caches

    state1 = cache.caches[0]
    assert "recurrent_state" in state1
    assert "conv_state" in state1
    assert state1["recurrent_state"] is not None

    # Step 2: Forward pass with next token
    x2 = torch.randn(2, 1, 64)
    out2, _, _ = layer(x2, past_key_values=cache, use_cache=True)
    assert out2.shape == (2, 1, 64)
    assert cache.caches[0]["recurrent_state"] is not None


# =============================================================================
# UNHAPPY PATHS & EDGE CASES
# =============================================================================


def test_when_invalid_expand_v_then_raises_value_error():
    # expand_v=0.15 on head_dim=32 gives head_v_dim = 4.8 (non-integer)
    with pytest.raises(ValueError, match="does not produce an integer value"):
        GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, expand_v=0.15)


def test_when_invalid_num_v_heads_then_raises_value_error():
    # num_v_heads must be divisible by num_heads
    with pytest.raises(ValueError, match="must be divisible by num_heads"):
        GatedDeltaNet2(hidden_size=64, num_heads=3, num_v_heads=5, head_dim=32)


def test_when_invalid_mode_then_raises_assertion_error():
    with pytest.raises(AssertionError, match="Not supported mode"):
        GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, mode=cast(Any, "invalid_mode"))


def test_when_invalid_attention_mask_dimension_then_raises_assertion_error():
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32)
    x = torch.randn(2, 5, 64)
    # Mask is 3D, which is not supported
    mask_3d = torch.ones(2, 5, 5)
    with pytest.raises(AssertionError, match="Expected attention_mask as a 0-1 matrix"):
        layer(x, attention_mask=mask_3d)


def test_when_training_with_fused_recurrent_mode_then_raises_assertion_error():
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, mode="fused_recurrent")
    layer.train()
    x = torch.randn(2, 5, 64)
    with pytest.raises(AssertionError, match="Only chunk mode is supported in training"):
        layer(x)


def test_when_past_key_values_provided_without_layer_idx_then_raises_value_error():
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, layer_idx=None)
    cache = MockCache()
    x = torch.randn(2, 5, 64)
    with pytest.raises(ValueError, match="requires `layer_idx` when `past_key_values` is provided"):
        layer(x, past_key_values=cache)


def test_when_sequence_is_fully_padded_then_handles_successfully():
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, use_short_conv=True)
    x = torch.randn(2, 5, 64)
    mask = torch.ones(2, 5)
    mask[0, :] = 0  # First sequence completely padded (seq_len = 0 for it)

    out, _, _ = layer(x, attention_mask=mask)
    assert out.shape == (2, 5, 64)


def test_when_extreme_input_values_then_numerical_stability_preserved():
    layer = GatedDeltaNet2(hidden_size=64, num_heads=2, head_dim=32, use_short_conv=True)
    # Inputs containing large positive and negative values
    x = torch.zeros(2, 5, 64)
    x[0, :, :] = 1e4
    x[1, :, :] = -1e4

    out, _, _ = layer(x)
    assert out.shape == (2, 5, 64)
    assert not torch.isnan(out).any(), "NaN detected in output with extreme inputs"
    assert not torch.isinf(out).any(), "Inf detected in output with extreme inputs"
