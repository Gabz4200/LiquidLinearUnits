r"""LiquidLinear: input-conditioned full weight matrix."""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseLLU
from .utils import (
    DEVICE,
    _activate,
    _init_hypernetwork,
    _zero_out_last,
)


class LiquidLinear(BaseLLU):
    """Input‑conditioned full weight matrix.

    Memory **O(B · O · I)**: use only for small dimensions.  The hypernetwork
    is a single linear layer; the generated full matrix is squashed with
    *factor_activation* (only ``"tanh"`` or ``"none"`` are well‑motivated here).

    Zero-initialised so the layer behaves as a plain ``nn.Linear`` at step 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dynamic_bias: bool = True,
        factor_activation: str = "tanh",
        scale_init: float = 0.9,
        init_method: str = "hyperfan_in",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        r"""__init__(in_features, out_features, bias=True, dynamic_bias=True, factor_activation="tanh", scale_init=0.9, init_method="hyperfan_in", device=None, dtype=torch.float32) -> None

        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            bias (bool): whether the core :class:`~nn.Linear` has a learnable bias.
                Default: ``True``.
            dynamic_bias (bool): if ``True``, an input-dependent bias from an
                auxiliary linear layer is added to the output.
                Default: ``True``.
            factor_activation (str): activation for the generated weight matrix.
                One of ``"tanh"``, ``"norm"``, ``"rmsnorm"``, or ``"none"``.
                Default: ``"tanh"``.
            scale_init (float): initial value of the per-channel scalar multiplier
                on the adaptive path.  Default: ``0.9``.
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

        # Hypernetwork -> full out x in matrix
        self.hypernetwork = nn.Linear(
            in_features,
            out_features * in_features,
            bias=True,
            device=dev,
            dtype=dtype,
        )

        # Optional input‑dependent bias
        self.bias_dynamic: Optional[nn.Linear] = (
            nn.Linear(in_features, out_features, bias=True, device=dev, dtype=dtype)
            if dynamic_bias
            else None
        )

        self._init_weights()

    # Init

    def _init_weights(self) -> None:
        r"""_init_weights() -> None

        Initialise the hypernetwork and dynamic bias (if present) with the
        chosen init method, then zero the output layer so the adaptive path
        contributes nothing at step 1.
        """
        _init_hypernetwork(self.hypernetwork, self.init_method, self.in_features, self.out_features)
        _zero_out_last(self.hypernetwork)
        self._init_bias_dynamic()

    # Forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""forward(x) -> Tensor

        Args:
            x (Tensor): input tensor of shape ``(..., in_features)``.

        Returns:
            Tensor: output tensor of shape ``(..., out_features)`` computed as
            :math:`\text{core}(x) + \text{scale} \odot (W(x) \cdot x)`, where
            :math:`W(x)` is the input-conditioned full weight matrix.
        """
        core_out = self.linear_core(x)  # (..., O)

        # Dynamic matrix, activated
        W_raw = self.hypernetwork(x)  # (..., O*I)
        W = _activate(W_raw, self.factor_activation)
        W = W.reshape(*x.shape[:-1], self.out_features, self.in_features)  # (..., O, I)

        # delta_W = W @ x
        adaptive = torch.matmul(W, x.unsqueeze(-1)).squeeze(-1)  # (..., O)

        # Scale adaptive path, add bias
        out = core_out + adaptive * self.scale
        out = self._apply_dynamic_bias(out, x)

        return out

    # Utilities

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, act={self.factor_activation}"
