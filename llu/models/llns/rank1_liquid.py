r"""Rank1LiquidLN: rank-1 adaptive factors."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseLLU
from .utils import (
    DEVICE,
    _activate,
    _init_hypernetwork,
    _zero_b_section,
)


class Rank1LiquidLN(BaseLLU):
    """Input‑conditioned rank‑1 update (lightweight).

    The hypernetwork outputs a single pair of vectors :math:`a \\in \\mathbb{R}^O`,
    :math:`b \\in \\mathbb{R}^I` which are then activated (default ``"norm"``).
    The effective dynamic update is :math:`\\Delta W = a \\otimes b`.

    Zero-initialised so :math:`\\Delta W = 0` at step 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dynamic_bias: bool = False,
        factor_activation: str = "norm",
        scale_init: float = 0.9,
        normalize_input: bool = False,
        init_method: str = "hyperfan_in",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        r"""__init__(in_features, out_features, bias=True, dynamic_bias=False, factor_activation="norm", scale_init=0.5, normalize_input=False, init_method="hyperfan_in", device=None, dtype=torch.float32) -> None

        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            bias (bool): whether the core :class:`~nn.Linear` has a learnable bias.
                Default: ``True``.
            dynamic_bias (bool): if ``True``, an input-dependent bias from an
                auxiliary linear layer is added to the output.
                Default: ``False``.
            factor_activation (str): activation for the factor vectors :math:`a, b`.
                One of ``"tanh"``, ``"norm"``, ``"rmsnorm"``, or ``"none"``.
                ``"norm"`` is recommended.  Default: ``"norm"``.
            scale_init (float): initial value of the per-channel scalar multiplier
                on the adaptive path.  Default: ``0.9``.
            normalize_input (bool): if ``True``, apply RMSNorm to the input before
                feeding it to the hypernetwork.  Default: ``False``.
            init_method (str): weight initialisation method for the hypernetwork.
                One of ``"hyperfan_in"``, ``"hyperfan_out"``, ``"xavier"``,
                or ``"small"``.  Default: ``"hyperfan_in"``.
            device (torch.device, optional): the desired device of the parameters.
                Default: ``None``.
            dtype (torch.dtype): the desired data type of the parameters.
                Default: ``torch.float32``.
        """
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
        self.normalize_input = normalize_input

        # Single hypernetwork -> a (O) + b (I)
        self.hypernetwork = nn.Linear(
            in_features,
            out_features + in_features,
            bias=True,
            device=dev,
            dtype=dtype,
        )

        self.bias_dynamic: Optional[nn.Linear] = (
            nn.Linear(in_features, out_features, bias=True, device=dev, dtype=dtype)
            if dynamic_bias
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        r"""_init_weights() -> None

        Initialise the hypernetwork with the chosen init method, then zero the
        b‑section of its output layer so that :math:`\Delta W = 0` at
        step 1 while gradient still flows through the a‑factors.
        """
        _init_hypernetwork(
            self.hypernetwork, self.init_method, self.in_features, self.out_features, rank=1
        )

        # Zero b-section only; a keeps gradient flowing
        _zero_b_section(self.hypernetwork, self.out_features)
        self._init_bias_dynamic()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Args:
            x (Tensor): input tensor of shape ``(..., in_features)``.

        Returns:
            Tensor: output tensor of shape ``(..., out_features)`` computed as
            :math:`\text{core}(x) + \text{scale} \odot a \, (b \cdot x)`, where
            :math:`a \in \mathbb{R}^O, b \in \mathbb{R}^I` are the
            input-conditioned rank-1 factors.
        """

        # RMSNorm for magnitude invariance
        h_in = F.rms_norm(x, (self.in_features,)) if self.normalize_input else x

        raw = self.hypernetwork(h_in)  # (..., O + I)

        a = _activate(raw[..., :self.out_features].reshape(*h_in.shape[:-1], 1, self.out_features), self.factor_activation)
        b = _activate(raw[..., self.out_features:].reshape(*h_in.shape[:-1], 1, self.in_features), self.factor_activation)

        adaptive = self._compute_low_rank_adaptive(a, b, x)
        out = self.linear_core(x) + adaptive

        out = self._apply_dynamic_bias(out, x)

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"act={self.factor_activation}, norm_input={self.normalize_input}"
        )
