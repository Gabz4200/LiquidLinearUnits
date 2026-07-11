r"""Learnable BatchMomentumLiquidLN: per-batch-element momentum."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import BaseMomentumLLU
from .utils import (
    DEVICE,
    _init_hypernetwork,
    _zero_b_section,
)


class LearnableBatchMomentumLiquidLN(BaseMomentumLLU):
    r"""Input-conditioned rank-:math:`R` update with per-batch-element momentum.

    Like :class:`SharedMomentumLiquidLN` but the momentum is tracked separately
    for each element in the batch dimension.  The raw buffers have shape
    ``(*batch, rank, feat)`` -- the batch dimensions are included and each
    :math:`(b_1, \dots, b_n)` slice has its own state.

    Because the buffer shape depends on the input, a runtime guard detects
    shape/device/dtype mismatches and re-initialises the buffer when needed
    (e.g. on batch-size changes).  This makes it slightly heavier than
    :class:`SharedMomentumLiquidLN` but necessary when per-sample momentum
    dynamics are required.

    Zero-initialised so the adaptive path contributes nothing at step 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        decay_rate=0.4,
        rank: int = 4,
        hyper_hidden_dim: Optional[int] = None,
        bias: bool = True,
        dynamic_bias: bool = False,
        factor_activation: str = "norm",
        scale_init: float = 0.01,
        normalize_input: bool = True,
        init_method: str = "hyperfan_in",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        r"""__init__(in_features, out_features, decay_rate=0.4, rank=4, hyper_hidden_dim=None, bias=True, dynamic_bias=False, factor_activation="norm", scale_init=0.01, normalize_input=True, init_method="hyperfan_in", device=None, dtype=torch.float32) -> None

        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            decay_rate (float): decay factor for the momentum buffers.
                Values close to ``1`` give longer memory.  Default: ``0.4``.
            rank (int): number of factor pairs :math:`(a_r, b_r)`.  Default: ``4``.
            hyper_hidden_dim (int, optional): hidden dimension of the MLP
                hypernetwork.  Default: ``None`` (``max(in_features // 4, rank * 16)``).
            bias (bool): whether the core :class:`~nn.Linear` has a learnable bias.
                Default: ``True``.
            dynamic_bias (bool): if ``True``, an input-dependent bias from an
                MLP is added to the output.  Default: ``False``.
            factor_activation (str): activation for the factor vectors.
                One of ``"tanh"``, ``"norm"``, ``"rmsnorm"``, or ``"none"``.
                Default: ``"norm"``.
            scale_init (float): initial value of the per-channel scalar multiplier
                on the adaptive path.  Default: ``0.01``.
            normalize_input (bool): if ``True``, apply RMSNorm to the conditioning
                input before the hypernetwork.  Default: ``True``.
            init_method (str): weight initialisation method for the hypernetwork.
                One of ``"hyperfan_in"``, ``"hyperfan_out"``, ``"xavier"``,
                or ``"small"``.  Default: ``"hyperfan_in"``.
            device (torch.device, optional): the desired device of the parameters.
                Default: ``None``.
            dtype (torch.dtype): the desired data type of the parameters.
                Default: ``torch.float32``.
        """
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            decay_rate=decay_rate,
            rank=rank,
            bias=bias,
            scale_init=scale_init,
            factor_activation=factor_activation,
            init_method=init_method,
            device=device,
            dtype=dtype,
        )
        dev = device if device is not None else DEVICE

        self.normalize_input = normalize_input
        self.decay_rate = nn.Parameter(torch.tensor(decay_rate, device=dev, dtype=dtype))

        # Placeholder buffer; real shape is set on first forward via the guard
        self.register_buffer(
            "a_raw", torch.zeros(1, rank, out_features, device=dev, dtype=dtype), persistent=True
        )
        self.register_buffer(
            "b_raw", torch.zeros(1, rank, in_features, device=dev, dtype=dtype), persistent=True
        )

        # MLP hypernetwork
        hidden_dim = hyper_hidden_dim or max(in_features // 4, rank * 16)
        self.hypernetwork = nn.Sequential(
            nn.Linear(in_features, hidden_dim, device=dev, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_dim, rank * (out_features + in_features), device=dev, dtype=dtype),
        )

        # Dynamic bias with MLP
        self.bias_dynamic: Optional[nn.Sequential] = (
            nn.Sequential(
                nn.Linear(in_features, hidden_dim, device=dev, dtype=dtype),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_features, device=dev, dtype=dtype),
            )
            if dynamic_bias
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        r"""_init_weights() -> None

        Initialise hypernetwork layers with the chosen init method, then zero the
        b-section of the output layer so the adaptive path produces zero at step 1
        while a-factors keep gradient flowing.

        The dynamic bias MLP (if present) is small-initialised and its
        final layer zeroed.
        """
        _init_hypernetwork(
            self.hypernetwork,
            self.init_method,
            self.in_features,
            self.out_features,
            rank=self.rank,
        )

        # Zero b-section; a-factors keep gradient flowing
        _zero_b_section(self.hypernetwork, self.rank * self.out_features)
        self._init_bias_dynamic()

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""forward(x, cond=None) -> Tensor

        Args:
            x (Tensor): input tensor of shape ``(..., in_features)``.
            cond (Tensor, optional): optional conditioning tensor.  When
                ``None``, *x* is used.  The conditioning drives the hypernetwork
                while *x* always goes through the core linear path.

        Returns:
            Tensor: output tensor of shape ``(..., out_features)`` with
            momentum-smoothed rank-:math:`R` adaptive update.
        """

        cond = cond if cond is not None else x

        # RMSNorm for magnitude invariance
        h_in = F.rms_norm(cond, (self.in_features,)) if self.normalize_input else cond

        core_out = self.linear_core(x)

        raw = self.hypernetwork(h_in)

        split = self.rank * self.out_features

        a_new = raw[..., :split].reshape(*h_in.shape[:-1], self.rank, self.out_features)
        b_new = raw[..., split:].reshape(*h_in.shape[:-1], self.rank, self.in_features)

        a, b = self._update_batch_momentum(a_new, b_new)

        adaptive = self._compute_low_rank_adaptive(a, b, x)
        out = core_out + adaptive

        out = self._apply_dynamic_bias(out, cond)

        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"act={self.factor_activation}, norm_input={self.normalize_input}"
        )
