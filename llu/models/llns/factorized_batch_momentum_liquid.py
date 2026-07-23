r"""FactorizedBatchMomentumLiquidLN: factorized A/B generation with per-batch-element momentum.

Combines the factorized A/B projection pattern from :mod:`FactorizedLiquidLN`
with the per-batch-element EMA momentum from :class:`BatchMomentumLiquidLN`.
Two separate MLP projections generate A and B factors independently, then
momentum smoothing is applied per batch element before computing the
low-rank adaptive update.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseMomentumLLU
from .utils import (
    DEVICE,
    _activate,
    _factorized_hyperfan_init,
    _small_init,
    _validate_parameterization,
    _zero_out_last,
)


class FactorizedBatchMomentumLiquidLN(BaseMomentumLLU):
    """Input-conditioned rank-R update with factorized A/B generation and per-batch-element momentum.

    Two independent MLP projections generate the A and B factor matrices
    separately, then momentum smoothing is applied per batch element before
    the low-rank adaptive product.  This combines the factorised
    initialisation benefits of FactorizedLiquidLN with the temporal
    smoothing of BatchMomentumLiquidLN.

    Supports both LoRA and SVD parameterizations.  In SVD mode, a single
    projection generates the dynamic scaling factor g.

    Zero-initialised so the adaptive path contributes nothing at step 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cond_dim: Optional[int] = None,
        decay_rate: float = 0.4,
        rank: int = 4,
        hyper_hidden_dim: Optional[int] = None,
        bias: bool = True,
        dynamic_bias: bool = False,
        factor_activation: str = "norm",
        scale_init: float = 0.01,
        normalize_input: bool = True,
        init_method: str = "hyperfan_in",
        learnable_decay_rate: bool = False,
        parameterization: str = "lora",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        _validate_parameterization(parameterization)

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            decay_rate=decay_rate,
            rank=rank,
            bias=bias,
            scale_init=scale_init,
            factor_activation=factor_activation,
            init_method=init_method,
            learnable_decay_rate=learnable_decay_rate,
            device=device,
            dtype=dtype,
        )
        dev = device if device is not None else DEVICE

        self.normalize_input = normalize_input
        self.parameterization = parameterization
        self.cond_dim = cond_dim if cond_dim is not None else in_features

        self._register_momentum_buffers(dev, dtype, batch=True)

        hidden_dim = hyper_hidden_dim or max(self.cond_dim // 4, rank * 16)

        if self.parameterization == "lora":
            self.proj_a = nn.Sequential(
                nn.Linear(self.cond_dim, hidden_dim, device=dev, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, rank * out_features, device=dev, dtype=dtype),
            )
            self.proj_b = nn.Sequential(
                nn.Linear(self.cond_dim, hidden_dim, device=dev, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, rank * in_features, device=dev, dtype=dtype),
            )
        else:
            self.proj_a = nn.Sequential(
                nn.Linear(self.cond_dim, hidden_dim, device=dev, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, rank, device=dev, dtype=dtype),
            )
            self.proj_b = None

        self.bias_dynamic: Optional[nn.Sequential] = (
            nn.Sequential(
                nn.Linear(self.cond_dim, hidden_dim, device=dev, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_features, device=dev, dtype=dtype),
            )
            if dynamic_bias
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        if self.parameterization == "lora":
            _factorized_hyperfan_init(
                self.proj_a,
                self.proj_b,
                self.in_features,
                self.out_features,
                self.rank,
            )
            last_b = self.proj_b[-1] if isinstance(self.proj_b, nn.Sequential) else self.proj_b
            with torch.no_grad():
                last_b.weight.data.zero_()
                if last_b.bias is not None:
                    last_b.bias.data.zero_()
        else:
            self._init_svd_projection(self.proj_a)
        if self.bias_dynamic is not None:
            _small_init(self.bias_dynamic)
            _zero_out_last(self.bias_dynamic)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond = cond if cond is not None else x
        h_in = F.rms_norm(cond, (self.cond_dim,)) if self.normalize_input else cond

        core_out = self.linear_core(x)

        if self.parameterization == "lora":
            a_new = self.proj_a(h_in).reshape(*h_in.shape[:-1], self.rank, self.out_features)
            b_new = self.proj_b(h_in).reshape(*h_in.shape[:-1], self.rank, self.in_features)

            a, b = self._update_batch_momentum(a_new, b_new)
            adaptive = self._compute_low_rank_adaptive(a, b, x)
        else:
            g = self._update_g_batch_momentum(self.proj_a(h_in))
            adaptive = self._compute_svd_adaptive(x, g)

        out = core_out + adaptive
        out = self._apply_dynamic_bias(out, cond)

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"act={self.factor_activation}, norm_input={self.normalize_input}, "
            f"mode={self.parameterization}"
        )
