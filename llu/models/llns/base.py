r"""Base classes for Liquid Linear Units."""


import torch
from torch import nn

from .utils import (
    _FreezeMixin,
    _init_hypernetwork,
    _small_init,
    _zero_b_section,
    _zero_out_last,
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
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factor_activation = factor_activation
        self.init_method = init_method

        self.linear_core = nn.Linear(
            in_features, out_features, bias=bias, device=device, dtype=dtype
        )
        nn.init.xavier_uniform_(self.linear_core.weight)
        if self.linear_core.bias is not None:
            nn.init.zeros_(self.linear_core.bias)
        self.bias_dynamic: nn.Module | None = None
        self.U: nn.Parameter | None = None
        self.V: nn.Parameter | None = None
        # `rank` and `parameterization` are set by concrete subclasses before
        # any SVD/low-rank init runs. They are NOT declared as None here
        # so that pyrefly sees them as their concrete types in subclasses.
        self.rank: int = 1
        self.parameterization: str = "lora"

    def _init_bias_dynamic(self) -> None:
        if self.bias_dynamic is not None:
            _small_init(self.bias_dynamic)
            _zero_out_last(self.bias_dynamic)

    def _init_low_rank_adaptive(
        self,
        target: nn.Module,
        b_start: int,
        rank: int | None = None,
    ) -> None:
        r"""_init_low_rank_adaptive(target, b_start, rank=None) -> None

        Initialise a low-rank hypernetwork / projection module so the adaptive
        path contributes zero at step 1 while gradients still flow through the
        a-factors.  Shared by every rank-based LLU variant; the GDN variants
        prepend the GDN-2 internal init before calling this.

        Args:
            target (nn.Module): the hypernetwork or projection module whose
                last linear layer is initialised.
            b_start (int): row index where the b-section begins
                (``rank * out_features``).
            rank (int, optional): rank passed to the hyperfan variance scaling.
        """
        _init_hypernetwork(
            target, self.init_method, self.in_features, self.out_features, rank=rank
        )
        _zero_b_section(target, b_start)
        self._init_bias_dynamic()

    def _apply_dynamic_bias(self, out: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.bias_dynamic is not None:
            out = out + self.bias_dynamic(cond)
        return out

    def _compute_svd_adaptive(
        self,
        x: torch.Tensor,
        g: torch.Tensor,
    ) -> torch.Tensor:
        r"""_compute_svd_adaptive(x, g) -> Tensor

        SVD-parameterized adaptive path: ``((x @ V) * g) @ U``.
        """
        assert self.U is not None and self.V is not None, "U and V must be initialized for SVD"
        v = torch.matmul(x, self.V)
        modulated = v * g
        return torch.matmul(modulated, self.U)

    def _create_svd_factors(
        self,
        device: torch.device | None,
        dtype: torch.dtype,
    ) -> None:
        r"""_create_svd_factors(device, dtype) -> None

        Create the static U/V factors for SVD parameterization (uninitialised;
        :meth:`_init_svd_projection` fills them with xavier)."""
        assert self.rank is not None, "rank must be set before creating SVD factors"
        self.U = nn.Parameter(
            torch.empty(self.rank, self.out_features, device=device, dtype=dtype)
        )
        self.V = nn.Parameter(
            torch.empty(self.in_features, self.rank, device=device, dtype=dtype)
        )

    def _init_svd_projection(self, proj: nn.Module) -> None:
        r"""_init_svd_projection(proj) -> None

        Initialise the SVD projection module and U/V: hyperfan-init *proj*, zero
        its last layer (zero adaptive path at step 1), then xavier-init U/V and
        the optional dynamic bias."""
        assert self.U is not None and self.V is not None, "U and V must be set"
        _init_hypernetwork(
            proj, self.init_method, self.in_features, self.out_features, rank=self.rank
        )
        _zero_out_last(proj)
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        self._init_bias_dynamic()

    def _compute_low_rank_adaptive(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        dot = torch.matmul(b, x.unsqueeze(-1)).squeeze(-1)  # (..., rank)
        adaptive = torch.matmul(dot.unsqueeze(-2), a).squeeze(-2)  # (..., O)
        return adaptive


