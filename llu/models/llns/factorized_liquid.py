r"""FactorizedLiquidLN: Zhyper-inspired factorized A/B generation.

Instead of a single linear that outputs ``rank * (out + in)``, uses two
separate projections — one per factor — each with its own variance-scaled
initialisation.  This yields better gradient flow and ~2x fewer hypernetwork
parameters compared to the monolithic approach in StableLiquidLN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseLLU
from .utils import (
    DEVICE,
    _activate,
    _factorized_hyperfan_init,
    _small_init,
    _validate_parameterization,
    _zero_out_last,
)


class FactorizedLiquidLN(BaseLLU):
    """Input-conditioned rank-R update with factorized A/B generation.

    Two independent MLP projections generate the A and B factor matrices
    separately, each initialized with its own variance scaling.  This
    follows the factorized hypernetwork pattern from Zhyper (Abdalla et al.,
    2025) and achieves competitive performance with ~2x fewer hypernetwork
    parameters compared to monolithic generation.

    Supports both LoRA and SVD parameterizations.  In SVD mode, a single
    projection generates the diagonal scaling vector g, and learned static
    U/V factors are used (identical to StableLiquidLN SVD mode).

    Zero-initialised so the adaptive path contributes nothing at step 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        hyper_hidden_dim: Optional[int] = None,
        bias: bool = True,
        dynamic_bias: bool = False,
        factor_activation: str = "norm",
        scale_init: float = 0.01,
        normalize_input: bool = True,
        cond_dim: Optional[int] = None,
        init_method: str = "hyperfan_in",
        parameterization: str = "lora",
        lora_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        r"""__init__(in_features, out_features, rank=4, ...) -> None

        Args:
            in_features: size of each input sample.
            out_features: size of each output sample.
            rank: number of factor pairs (a_r, b_r).  Default: ``4``.
            hyper_hidden_dim: hidden dimension of each MLP projection.
                Default: ``None`` (``max(cond_dim // 4, rank * 16)``).
            bias: whether the core Linear has a learnable bias.
            dynamic_bias: if True, an input-dependent bias is added.
            factor_activation: activation for the factor vectors.
            scale_init: initial scale multiplier.
            normalize_input: if True, RMSNorm the conditioning input.
            cond_dim: dimension of the conditioning tensor.  Defaults to
                ``in_features``.
            init_method: weight init method.
            parameterization: ``"lora"`` or ``"svd"``.
            lora_alpha: LoRA alpha for scaling (alpha / rank).
            device: device of parameters.
            dtype: data type of parameters.
        """
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        _validate_parameterization(parameterization)

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
        dev = device if device is not None else DEVICE

        self.rank = rank
        self.normalize_input = normalize_input
        self.cond_dim = cond_dim if cond_dim is not None else in_features
        self.parameterization = parameterization
        self.lora_alpha = lora_alpha

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
            self._create_svd_factors(dev, dtype)

        self.bias_dynamic: Optional[nn.Module] = (
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
            # Zero the B-factors so adaptive path is zero at step 1.
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
            a_raw = self.proj_a(h_in).reshape(*h_in.shape[:-1], self.rank, self.out_features)
            b_raw = self.proj_b(h_in).reshape(*h_in.shape[:-1], self.rank, self.in_features)

            a = _activate(a_raw, self.factor_activation)
            b = _activate(b_raw, self.factor_activation)

            adaptive = self._compute_low_rank_adaptive(a, b, x)
        else:
            g_raw = self.proj_a(h_in)
            g = _activate(g_raw, self.factor_activation)
            adaptive = self._compute_svd_adaptive(x, g)

        lora_scale = self.lora_alpha / self.rank
        out = core_out + lora_scale * adaptive
        out = self._apply_dynamic_bias(out, cond)
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"act={self.factor_activation}, norm_input={self.normalize_input}, "
            f"mode={self.parameterization}, lora_alpha={self.lora_alpha}"
        )
