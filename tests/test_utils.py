import torch
import pytest
from llu.models import (
    LiquidLinear,
    Rank1LiquidLN,
    RankRLiquidLN,
    StableLiquidLN,
    SharedMomentumLiquidLN,
    BatchMomentumLiquidLN,
)

def _get_device(model):
    return model.linear_core.weight.device

@pytest.mark.parametrize("init_method", ["hyperfan_in", "hyperfan_out", "xavier", "small"])
def test_all_initialization_methods(init_method):
    in_features = 8
    out_features = 4
    # Ensure all initialization schemes run without error
    for model_cls in [
        LiquidLinear,
        Rank1LiquidLN,
        RankRLiquidLN,
        StableLiquidLN,
        SharedMomentumLiquidLN,
        BatchMomentumLiquidLN,
    ]:
        model = model_cls(in_features, out_features, init_method=init_method)
        # Smoke test forward pass
        x = torch.randn(2, in_features, device=_get_device(model))
        out = model(x)
        assert out.shape == (2, out_features)

@pytest.mark.parametrize("factor_activation", ["tanh", "norm", "rmsnorm", "none"])
def test_all_activations(factor_activation):
    in_features = 8
    out_features = 4
    for model_cls in [
        LiquidLinear,
        Rank1LiquidLN,
        RankRLiquidLN,
        StableLiquidLN,
        SharedMomentumLiquidLN,
        BatchMomentumLiquidLN,
    ]:
        model = model_cls(in_features, out_features, factor_activation=factor_activation)
        # Smoke test forward pass
        x = torch.randn(2, in_features, device=_get_device(model))
        out = model(x)
        assert out.shape == (2, out_features)

def test_invalid_activation():
    in_features = 8
    out_features = 4
    for model_cls in [
        LiquidLinear,
        Rank1LiquidLN,
        RankRLiquidLN,
        StableLiquidLN,
        SharedMomentumLiquidLN,
        BatchMomentumLiquidLN,
    ]:
        model = model_cls(in_features, out_features, factor_activation="invalid_act")
        x = torch.randn(2, in_features, device=_get_device(model))
        with pytest.raises(ValueError, match="Unknown factor_activation"):
            _ = model(x)
