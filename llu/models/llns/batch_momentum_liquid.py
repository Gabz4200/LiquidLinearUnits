r"""BatchMomentumLiquidLN: rank-R adaptive factors with per-batch-element momentum."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseMomentumLLU
from .utils import (
    DEVICE,
    _validate_parameterization,
)


class BatchMomentumLiquidLN(BaseMomentumLLU):
    """Input-conditioned rank-R update with per-batch-element momentum.

    Supports both LoRA and SVD parameterizations. In SVD mode, applies momentum to
    the dynamic scaling factor g.
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
        r"""__init__(in_features, out_features, cond_dim=None, decay_rate=0.4, rank=4, hyper_hidden_dim=None, bias=True, dynamic_bias=False, factor_activation="norm", scale_init=0.01, normalize_input=True, init_method="hyperfan_in", learnable_decay_rate=False, parameterization="lora", device=None, dtype=torch.float32) -> None
        """
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
        # Separate conditioning-stream dimension. Defaults to in_features so the
        # default (cond == x) behaviour is unchanged; the LLM intermediary passes
        # a GDN-2 stream of a different size via cond_dim.
        self.cond_dim = cond_dim if cond_dim is not None else in_features

        # Placeholder buffers; real shape is set on first forward via ensure_buffer_shape
        self._register_momentum_buffers(dev, dtype, batch=True)

        # MLP hypernetwork
        hidden_dim = hyper_hidden_dim or max(self.cond_dim // 4, rank * 16)
        hyper_out_dim = rank if self.parameterization == "svd" else rank * (out_features + in_features)

        self.hypernetwork = nn.Sequential(
            nn.Linear(self.cond_dim, hidden_dim, device=dev, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, hyper_out_dim, device=dev, dtype=dtype),
        )

        # Dynamic bias with MLP
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
        r"""_init_weights() -> None
        """
        if self.parameterization == "lora":
            self._init_low_rank_adaptive(self.hypernetwork, self.rank * self.out_features, rank=self.rank)
        else:
            self._init_svd_projection(self.hypernetwork)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""forward(x, cond=None) -> Tensor
        """
        cond = cond if cond is not None else x

        # RMSNorm for magnitude invariance (over the conditioning dim)
        h_in = F.rms_norm(cond, (self.cond_dim,)) if self.normalize_input else cond

        core_out = self.linear_core(x)

        raw = self.hypernetwork(h_in)

        if self.parameterization == "lora":
            split = self.rank * self.out_features
            a_new = raw[..., :split].reshape(*h_in.shape[:-1], self.rank, self.out_features)
            b_new = raw[..., split:].reshape(*h_in.shape[:-1], self.rank, self.in_features)

            a, b = self._update_batch_momentum(a_new, b_new)
            adaptive = self._compute_low_rank_adaptive(a, b, x)
        else:
            # SVD Mode
            g = self._update_g_batch_momentum(raw)
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
