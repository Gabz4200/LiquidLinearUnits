r"""Base classes for Liquid Linear Units."""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .utils import (
    DEVICE,
    _activate,
    _small_init,
    _zero_out_last,
    _ensure_buffer_shape,
    _FreezeMixin,
)


class BaseLLU(_FreezeMixin, nn.Module):
    r"""Abstract base class for all Liquid Linear Units.

    Handles initialization of features, device/dtype routing, the core linear
    layer, and the per-channel adaptive scale. Also provides helpers for dynamic
    bias initialization/application and low-rank factor operations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale_init: float = 0.9,
        factor_activation: str = "norm",
        init_method: str = "hyperfan_in",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factor_activation = factor_activation
        self.init_method = init_method

        dev = device if device is not None else DEVICE
        self.linear_core = nn.Linear(
            in_features, out_features, bias=bias, device=dev, dtype=dtype
        )
        self.scale = nn.Parameter(
            torch.full((out_features,), scale_init, device=dev, dtype=dtype)
        )
        self.bias_dynamic: Optional[nn.Module] = None

    def _init_bias_dynamic(self) -> None:
        if self.bias_dynamic is not None:
            _small_init(self.bias_dynamic)
            _zero_out_last(self.bias_dynamic)

    def _apply_dynamic_bias(self, out: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.bias_dynamic is not None:
            out = out + self.bias_dynamic(cond)
        return out

    def _compute_low_rank_adaptive(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        dot = torch.matmul(b, x.unsqueeze(-1)).squeeze(-1)  # (..., rank)
        if hasattr(self, "rank_scale"):
            dot = dot * self.rank_scale
        adaptive = torch.matmul(dot.unsqueeze(-2), a).squeeze(-2)  # (..., O)
        return adaptive * self.scale


class BaseMomentumLLU(BaseLLU):
    r"""Base class for Liquid Linear Units with momentum buffers."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        decay_rate: float = 0.4,
        rank: int = 4,
        bias: bool = True,
        scale_init: float = 0.01,
        factor_activation: str = "norm",
        init_method: str = "hyperfan_in",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            scale_init=scale_init,
            factor_activation=factor_activation,
            init_method=init_method,
            device=device,
            dtype=dtype,
        )
        self.rank = rank
        dev = device if device is not None else DEVICE
        self.rank_scale = nn.Parameter(
            torch.full((rank,), 1.0, device=dev, dtype=dtype)
        )

    @property
    def local_decay_rate(self) -> torch.Tensor:
        if isinstance(self.decay_rate, nn.Parameter):
            return torch.sigmoid(self.decay_rate)
        return self.decay_rate

    def _update_shared_momentum(
        self,
        a_new: torch.Tensor,
        b_new: torch.Tensor,
        detach: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if detach:
            self.a_raw = self.a_raw.detach()
            self.b_raw = self.b_raw.detach()

        dims = tuple(range(a_new.ndim - 2))
        if dims:
            a_mean = a_new.mean(dim=dims)
            b_mean = b_new.mean(dim=dims)
        else:
            a_mean = a_new
            b_mean = b_new

        self.a_raw = self.a_raw * self.local_decay_rate + a_mean
        self.b_raw = self.b_raw * self.local_decay_rate + b_mean

        a = _activate(self.a_raw.expand_as(a_new), self.factor_activation)
        b = _activate(self.b_raw.expand_as(b_new), self.factor_activation)
        return a, b

    def _update_batch_momentum(
        self,
        a_new: torch.Tensor,
        b_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.a_raw = _ensure_buffer_shape(self.a_raw, a_new)
        self.b_raw = _ensure_buffer_shape(self.b_raw, b_new)

        self.a_raw = self.a_raw * self.local_decay_rate + a_new
        self.b_raw = self.b_raw * self.local_decay_rate + b_new

        a = _activate(self.a_raw, self.factor_activation)
        b = _activate(self.b_raw, self.factor_activation)
        return a, b
