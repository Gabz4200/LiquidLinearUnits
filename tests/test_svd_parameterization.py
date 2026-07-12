import torch
import pytest
from llu.models.llns import (
    RankRLiquidLN,
    GDNLiquidLN,
    SharedMomentumLiquidLN,
    BatchMomentumLiquidLN,
    MomentumGDNLiquidLN,
)


@pytest.mark.parametrize(
    "model_cls,extra_kwargs",
    [
        (RankRLiquidLN, {}),
        (GDNLiquidLN, {"mode": "chunk"}),
        (SharedMomentumLiquidLN, {}),
        (BatchMomentumLiquidLN, {}),
        (MomentumGDNLiquidLN, {"mode": "chunk"}),
    ],
)
def test_svd_parameterization_happy_paths(model_cls, extra_kwargs):
    torch.manual_seed(42)
    in_features = 8
    out_features = 4
    rank = 3

    # Instantiate model in SVD mode
    model = model_cls(
        in_features=in_features,
        out_features=out_features,
        rank=rank,
        parameterization="svd",
        bias=True,
        dynamic_bias=True,
        **extra_kwargs
    )
    device = model.linear_core.weight.device
    x = torch.randn(2, 3, in_features, device=device)

    # 1. Step 1 zero init verification (adaptive path contributes nothing)
    out = model(x)
    core_out = model.linear_core(x)
    assert torch.allclose(out, core_out, atol=1e-6)

    # 2. Forward execution check
    assert out.shape == (2, 3, out_features)

    # 3. Parameter attributes presence and gradients
    assert hasattr(model, "U") and model.U is not None
    assert hasattr(model, "V") and model.V is not None

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    y = torch.randn(2, 3, out_features, device=device)
    loss = (model(x) - y).pow(2).sum()
    loss.backward()

    # U and V must have gradients
    assert model.U.grad is not None
    assert model.V.grad is not None

    # Step optimizer and ensure no NaNs
    optimizer.step()
    assert not torch.isnan(model.U).any()
    assert not torch.isnan(model.V).any()
