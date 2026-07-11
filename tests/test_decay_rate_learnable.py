import torch
import pytest

from llu.models.llns import (
    SharedMomentumLiquidLN,
    BatchMomentumLiquidLN,
    MomentumGDNLiquidLN,
)

MOMENTUM_CLASSES = [
    SharedMomentumLiquidLN,
    BatchMomentumLiquidLN,
    MomentumGDNLiquidLN,
]


@pytest.mark.parametrize("cls", MOMENTUM_CLASSES)
def test_when_default_constructed_then_decay_rate_is_frozen(cls):
    model = cls(8, 8)
    assert isinstance(model.decay_rate, torch.nn.Parameter)
    assert model.decay_rate.requires_grad is False


@pytest.mark.parametrize("cls", MOMENTUM_CLASSES)
def test_when_learnable_flag_true_then_decay_rate_requires_grad(cls):
    model = cls(8, 8, learnable_decay_rate=True)
    assert model.decay_rate.requires_grad is True


@pytest.mark.parametrize("cls", MOMENTUM_CLASSES)
def test_when_learnable_flag_false_then_decay_rate_frozen(cls):
    model = cls(8, 8, learnable_decay_rate=False)
    assert model.decay_rate.requires_grad is False


@pytest.mark.parametrize("cls", MOMENTUM_CLASSES)
def test_local_decay_rate_is_sigmoid_bounded(cls):
    model = cls(8, 8)
    value = model.local_decay_rate
    assert torch.all(0.0 < value) and torch.all(value < 1.0)
    assert torch.allclose(value, torch.sigmoid(model.decay_rate))


@pytest.mark.parametrize("cls", MOMENTUM_CLASSES)
def test_set_decay_rate_learnable_toggles_requires_grad(cls):
    model = cls(8, 8)
    assert model.decay_rate.requires_grad is False
    model.set_decay_rate_learnable(True)
    assert model.decay_rate.requires_grad is True
    model.set_decay_rate_learnable(False)
    assert model.decay_rate.requires_grad is False
