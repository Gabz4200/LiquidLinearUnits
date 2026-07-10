import torch
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


def test_gradcheck():
    """Verify that analytical gradients match finite differences via autograd.gradcheck."""
    torch.manual_seed(42)
    in_features = 4
    out_features = 2
    for model_cls in [
        LiquidLinear,
        Rank1LiquidLN,
        RankRLiquidLN,
        StableLiquidLN,
        SharedMomentumLiquidLN,
        BatchMomentumLiquidLN,
    ]:
        # Using double precision (float64) is required for gradcheck stability
        model = model_cls(in_features, out_features, bias=True, dynamic_bias=True).double()
        x = torch.randn(
            2, in_features, dtype=torch.float64, device=_get_device(model), requires_grad=True
        )
        # check_undefined_grad=False is needed for momentum layers due to stateful buffer updates
        res = torch.autograd.gradcheck(model, (x,), eps=1e-6, atol=1e-4, check_undefined_grad=False)
        assert res is True


def test_dtype_and_compile_compatibility():
    """Verify half-precision casting (bfloat16) and torch.compile compatibility."""
    torch.manual_seed(42)
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
        model = model_cls(in_features, out_features)
        device = _get_device(model)

        # 1. bfloat16 casting & forward/backward verification
        model_bf16 = model_cls(in_features, out_features).to(dtype=torch.bfloat16)
        x_bf16 = torch.randn(2, in_features, dtype=torch.bfloat16, device=device)
        out_bf16 = model_bf16(x_bf16)
        assert out_bf16.dtype == torch.bfloat16
        assert out_bf16.shape == (2, out_features)

        loss_bf16 = out_bf16.pow(2).sum()
        loss_bf16.backward()

        # Verify gradient dtype is bfloat16
        assert model_bf16.linear_core.weight.grad is not None
        assert model_bf16.linear_core.weight.grad.dtype == torch.bfloat16

        # 2. torch.compile compatibility
        compiled_model = torch.compile(model)
        x_compile = torch.randn(2, in_features, device=device)
        out_compile = compiled_model(x_compile)
        assert out_compile.shape == (2, out_features)
