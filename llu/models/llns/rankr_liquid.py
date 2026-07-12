r"""RankRLiquidLN: rank-R adaptive factors."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseLLU
from .utils import (
    DEVICE,
    _activate,
    _validate_parameterization,
)


class RankRLiquidLN(BaseLLU):
    """Input‑conditioned rank‑R update.

    Generates :math:`R` pairs of factors :math:`\\{a_r, b_r\\}` under "lora",
    or a scaling vector :math:`g` under "svd" parameterization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 1,
        bias: bool = True,
        dynamic_bias: bool = False,
        factor_activation: str = "norm",
        scale_init: float = 0.5,
        normalize_input: bool = False,
        nonlinear_hypernet: bool = False,
        hyper_hidden_dim: Optional[int] = None,
        init_method: str = "hyperfan_in",
        parameterization: str = "lora",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        r"""__init__(in_features, out_features, rank=1, bias=True, dynamic_bias=False, factor_activation="norm", scale_init=0.5, normalize_input=False, nonlinear_hypernet=False, hyper_hidden_dim=None, init_method="hyperfan_in", parameterization="lora", device=None, dtype=torch.float32) -> None

        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            rank (int): number of factor pairs :math:`(a_r, b_r)`.  Default: ``1``.
            bias (bool): whether the core :class:`~nn.Linear` has a learnable bias.
                Default: ``True``.
            dynamic_bias (bool): if ``True``, an input-dependent bias from an
                auxiliary linear layer is added to the output.
                Default: ``False``.
            factor_activation (str): activation for the factor vectors.
                One of ``"tanh"``, ``"norm"``, ``"rmsnorm"``, or ``"none"``.
                Default: ``"norm"``.
            scale_init (float): initial value of the per-channel scalar multiplier
                on the adaptive path.  Default: ``0.5``.
            normalize_input (bool): if ``True``, apply RMSNorm to the input before
                feeding it to the hypernetwork.  Default: ``False``.
            nonlinear_hypernet (bool): if ``True``, use a 2-layer MLP (SiLU)
                instead of a single linear layer for the hypernetwork.
                Default: ``False``.
            hyper_hidden_dim (int, optional): hidden dimension of the MLP
                hypernetwork.  Ignored when *nonlinear_hypernet* is ``False``.
                Default: ``None`` (``max(in_features // 4, rank * 16)``).
            init_method (str): weight initialisation method for the hypernetwork.
                One of ``"hyperfan_in"``, ``"hyperfan_out"``, ``"xavier"``,
                or ``"small"``.  Default: ``"hyperfan_in"``.
            parameterization (str): parameterization mode for the update, either
                ``"lora"`` or ``"svd"``. Default: ``"lora"``.
            device (torch.device, optional): the desired device of the parameters.
                Default: ``None``.
            dtype (torch.dtype): the desired data type of the parameters.
                Default: ``torch.float32``.
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
        self.parameterization = parameterization

        # Hypernetwork output dimension
        hyper_out_dim = rank if self.parameterization == "svd" else rank * (out_features + in_features)

        if nonlinear_hypernet:
            hidden = hyper_hidden_dim or max(in_features // 4, rank * 16)
            self.hypernetwork: nn.Module = nn.Sequential(
                nn.Linear(in_features, hidden, device=dev, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden, hyper_out_dim, device=dev, dtype=dtype),
            )
        else:
            self.hypernetwork = nn.Linear(
                in_features, hyper_out_dim, bias=True, device=dev, dtype=dtype
            )

        if self.parameterization == "svd":
            self._create_svd_factors(dev, dtype)

        self.bias_dynamic: Optional[nn.Linear] = (
            nn.Linear(in_features, out_features, bias=True, device=dev, dtype=dtype)
            if dynamic_bias
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        r"""_init_weights() -> None

        Initialise hypernetwork layers with the chosen init method.
        """
        if self.parameterization == "lora":
            self._init_low_rank_adaptive(self.hypernetwork, self.rank * self.out_features, rank=self.rank)
        else:
            self._init_svd_projection(self.hypernetwork)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Args:
            x (Tensor): input tensor of shape ``(..., in_features)``.

        Returns:
            Tensor: output tensor of shape ``(..., out_features)``.
        """
        h_in = F.rms_norm(x, (self.in_features,)) if self.normalize_input else x
        raw = self.hypernetwork(h_in)

        core_out = self.linear_core(x)

        if self.parameterization == "lora":
            split = self.rank * self.out_features
            a_raw = raw[..., :split].reshape(*x.shape[:-1], self.rank, self.out_features)
            b_raw = raw[..., split:].reshape(*x.shape[:-1], self.rank, self.in_features)

            a = _activate(a_raw, self.factor_activation)
            b = _activate(b_raw, self.factor_activation)

            adaptive = self._compute_low_rank_adaptive(a, b, x)
        else:
            # SVD mode
            g = _activate(raw, self.factor_activation)
            adaptive = self._compute_svd_adaptive(x, g)

        out = core_out + adaptive
        out = self._apply_dynamic_bias(out, x)

        return out

    def extra_repr(self) -> str:
        nonlinear = isinstance(self.hypernetwork, nn.Sequential)
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"act={self.factor_activation}, norm_input={self.normalize_input}, "
            f"nonlinear_hypernet={nonlinear}, mode={self.parameterization}"
        )
