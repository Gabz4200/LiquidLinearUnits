import pytest
import torch

from llu.models.liquid_model import LiquidTransformer, build_model
from llu.models.llns.engram_retrieved_lora import EngramRetrievedLoraLN
from llu.models.mlp_model import LiquidMLP

# =============================================================================
# INITIALISATION EQUIVALENCE (adaptive path must be zero at step 1)
# =============================================================================


def test_when_initialization_equivalence_lora_then_output_matches_linear_core():
    torch.manual_seed(42)
    x = torch.randn(2, 5, 8)
    model = EngramRetrievedLoraLN(8, 4, rank=4, num_experts=4, parameterization="lora")
    assert torch.allclose(model(x), model.linear_core(x), atol=1e-6)


def test_when_initialization_equivalence_svd_then_output_matches_linear_core():
    torch.manual_seed(42)
    x = torch.randn(2, 5, 8)
    model = EngramRetrievedLoraLN(8, 4, rank=4, num_experts=4, parameterization="svd")
    assert torch.allclose(model(x), model.linear_core(x), atol=1e-6)


# =============================================================================
# SHAPES & DIMENSIONS
# =============================================================================


def test_when_forward_with_various_dimensions_then_shapes_are_correct():
    in_features, out_features = 8, 4
    model_2d = EngramRetrievedLoraLN(in_features, out_features, rank=2, num_experts=3)
    model_3d = EngramRetrievedLoraLN(in_features, out_features, rank=2, num_experts=3)
    model_cd = EngramRetrievedLoraLN(
        in_features, out_features, rank=2, num_experts=3, cond_dim=12
    )

    x2 = torch.randn(5, in_features)
    x3 = torch.randn(2, 3, in_features)
    cond = torch.randn(2, 3, 12)

    assert model_2d(x2).shape == (5, out_features)
    assert model_3d(x3).shape == (2, 3, out_features)
    assert model_cd(x3, cond=cond).shape == (2, 3, out_features)


# =============================================================================
# RETRIEVAL AND DIAGONAL GATING BEHAVIOR
# =============================================================================


def test_when_conditioned_on_sequence_then_cond_changes_output():
    torch.manual_seed(42)
    in_features, out_features = 8, 4
    model = EngramRetrievedLoraLN(
        in_features, out_features, rank=2, num_experts=4, cond_dim=8
    )

    # Break zero-init state by populating B_bank and gate_net
    with torch.no_grad():
        model.B_bank.normal_(std=0.1)
        model.gate_net.weight.normal_(std=0.1)

    x = torch.randn(2, 5, in_features)
    cond_a = torch.randn(2, 5, in_features)
    cond_b = torch.randn(2, 5, in_features)

    out_a = model(x, cond=cond_a)
    out_b = model(x, cond=cond_b)

    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_when_num_experts_changes_then_routing_weights_and_output_valid():
    torch.manual_seed(42)
    in_features, out_features = 16, 8
    model_k2 = EngramRetrievedLoraLN(in_features, out_features, num_experts=2)
    model_k8 = EngramRetrievedLoraLN(in_features, out_features, num_experts=8)

    x = torch.randn(3, 10, in_features)
    out2 = model_k2(x)
    out8 = model_k8(x)

    assert out2.shape == (3, 10, out_features)
    assert out8.shape == (3, 10, out_features)


# =============================================================================
# FREEZE SEMANTICS
# =============================================================================


def test_when_core_frozen_then_backward_only_updates_hypernetwork():
    torch.manual_seed(42)
    model = EngramRetrievedLoraLN(8, 4, rank=2, num_experts=3)
    model.freeze("core")

    x = torch.randn(2, 5, 8, requires_grad=True)
    out = model(x)
    out.sum().backward()

    assert model.linear_core.weight.grad is None
    assert model.gate_net.weight.grad is not None


def test_when_hypernetwork_frozen_then_backward_only_updates_core():
    torch.manual_seed(42)
    model = EngramRetrievedLoraLN(8, 4, rank=2, num_experts=3)
    model.freeze("hyper")

    x = torch.randn(2, 5, 8, requires_grad=True)
    out = model(x)
    out.sum().backward()

    assert model.linear_core.weight.grad is not None
    assert model.gate_net.weight.grad is None
    assert model.A_bank.grad is None
    assert model.B_bank.grad is None


# =============================================================================
# ERROR PATHS AND STABILITY
# =============================================================================


def test_when_invalid_rank_then_raises_value_error():
    with pytest.raises(ValueError, match="rank must be >= 1"):
        EngramRetrievedLoraLN(8, 4, rank=0)


def test_when_invalid_num_experts_then_raises_value_error():
    with pytest.raises(ValueError, match="num_experts must be >= 1"):
        EngramRetrievedLoraLN(8, 4, num_experts=0)


def test_when_invalid_parameterization_then_raises_value_error():
    with pytest.raises(ValueError, match="parameterization must be"):
        EngramRetrievedLoraLN(8, 4, parameterization="unknown")


def test_when_input_shape_mismatched_then_raises_runtime_error():
    model = EngramRetrievedLoraLN(8, 4)
    x_invalid = torch.randn(2, 5, 12)
    with pytest.raises(RuntimeError):
        model(x_invalid)


def test_when_extreme_input_values_then_numerical_stability_preserved():
    model = EngramRetrievedLoraLN(8, 4, rank=2, num_experts=4)
    x = torch.full((2, 5, 8), 1e4)
    out = model(x)
    assert not torch.isnan(out).any(), "NaN detected with extreme inputs"


def test_when_gradcheck_then_analytical_matches_numerical():
    torch.manual_seed(42)
    model = EngramRetrievedLoraLN(4, 2, rank=2, num_experts=3, dtype=torch.float64)
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(model, (x,), eps=1e-6, atol=1e-4, check_undefined_grad=False)


# =============================================================================
# INTEGRATION WITH MODEL FACTORIES
# =============================================================================


def test_when_liquid_transformer_built_with_engram_retrieved_lora_then_forward_passes():
    torch.manual_seed(42)
    model = build_model("EngramRetrievedLoraLN", d_model=16, out_dim=8, rank=2, num_experts=3)
    assert isinstance(model, LiquidTransformer)
    x = torch.randn(2, 5, 16)
    out = model(x)
    assert out.shape == (2, 5, 8)
    assert torch.isfinite(out).all()


def test_when_liquid_mlp_built_with_engram_retrieved_lora_then_forward_passes():
    torch.manual_seed(42)
    model = LiquidMLP(in_dim=16, hidden=32, n_layers=2, out_dim=4, lln_cls=EngramRetrievedLoraLN)
    x = torch.randn(5, 16)
    out = model(x)
    assert out.shape == (5, 4)
    assert torch.isfinite(out).all()

