r"""MomentumGDNLiquidLN: Liquid Linear Unit utilizing a Gated DeltaNet 2 (GDN-2) block with momentum."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Literal

from llu.models.gdn2 import GatedDeltaNet2
from .base import BaseMomentumLLU
from .utils import (
    DEVICE,
    _validate_parameterization,
    _run_gdn2_to_factors,
)


class MomentumGDNLiquidLN(BaseMomentumLLU):
    """Liquid Linear Unit utilizing a Gated DeltaNet 2 (GDN-2) block with momentum.

    Supports both LoRA and SVD parameterizations. In SVD mode, applies momentum to
    the dynamic scaling factor g.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        # GDN-2 specific parameters
        initial_decay_rate: float = 0.8,
        expand_v: float = 1.0,
        head_dim: int = 16,
        num_heads: int = 4,
        num_v_heads: Optional[int] = None,
        mode: Literal["chunk", "fused_recurrent"] = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        # standard LLU parameters
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
        r"""__init__(in_features, out_features, rank=4, initial_decay_rate=0.8, expand_v=1.0, head_dim=16, num_heads=4, num_v_heads=None, mode="chunk", use_short_conv=True, allow_neg_eigval=False, conv_size=4, conv_bias=False, layer_idx=None, norm_eps=1e-5, bias=True, dynamic_bias=False, factor_activation="norm", scale_init=0.01, normalize_input=True, init_method="hyperfan_in", learnable_decay_rate=False, parameterization="lora", device=None, dtype=torch.float32) -> None
        """
        _validate_parameterization(parameterization)

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            decay_rate=initial_decay_rate,
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

        # GDN-2 block as the sequence processor
        self.gdn2 = GatedDeltaNet2(
            hidden_size=in_features,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=num_heads,
            num_v_heads=num_v_heads,
            mode=mode,
            use_short_conv=use_short_conv,
            allow_neg_eigval=allow_neg_eigval,
            conv_size=conv_size,
            conv_bias=conv_bias,
            layer_idx=layer_idx,
            norm_eps=norm_eps,
            device=dev,
            dtype=dtype,
        )

        # Output projection from GDN-2 features to low-rank factors or scale
        proj_out_dim = rank if self.parameterization == "svd" else rank * (out_features + in_features)
        self.proj_out = nn.Linear(
            in_features,
            proj_out_dim,
            bias=True,
            device=dev,
            dtype=dtype,
        )

        # Registers dynamic factor buffers based on mode
        self._register_momentum_buffers(dev, dtype)

        # Dynamic bias projected from GDN-2 features
        self.bias_dynamic = (
            nn.Linear(in_features, out_features, bias=True, device=dev, dtype=dtype)
            if dynamic_bias
            else None
        )

        self._init_weights()

    def _init_weights(self) -> None:
        r"""_init_weights() -> None

        Initialise GDN-2 internal parameters, then project layer using the
        chosen init method, then zero the b-section of proj_out so the adaptive
        path produces zero at step 1.
        """
        # Initialize GDN-2 internal weights
        self.gdn2.apply(self.gdn2._initialize_weights)

        if self.parameterization == "lora":
            self._init_low_rank_adaptive(self.proj_out, self.rank * self.out_features, rank=self.rank)
        else:
            self._init_svd_projection(self.proj_out)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        r"""forward(x, cond=None, attention_mask=None, past_key_values=None, use_cache=False) -> Tensor or (Tensor, Cache)
        """
        cond = cond if cond is not None else x

        # RMSNorm for magnitude invariance
        h_in = F.rms_norm(cond, (self.in_features,)) if self.normalize_input else cond

        # Run GDN-2 and project its output to dynamic factors, preserving the
        # original leading dimensions. Momentum buffers are detached inside the
        # update calls below (detach=True), so no manual detach is needed here.
        orig_shape, gdn_out, raw, past_key_values = _run_gdn2_to_factors(
            self.gdn2,
            h_in,
            self.proj_out,
            rank=self.rank,
            in_features=self.in_features,
            out_features=self.out_features,
            parameterization=self.parameterization,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        core_out = self.linear_core(x)

        if self.parameterization == "lora":
            split = self.rank * self.out_features
            a_new = raw[..., :split].reshape(*orig_shape[:-1], self.rank, self.out_features)
            b_new = raw[..., split:].reshape(*orig_shape[:-1], self.rank, self.in_features)

            a, b = self._update_shared_momentum(a_new, b_new, detach=True)
            adaptive = self._compute_low_rank_adaptive(a, b, x)
        else:
            g = self._update_g_shared_momentum(raw, detach=True)
            adaptive = self._compute_svd_adaptive(x, g)

        out = core_out + adaptive

        if self.bias_dynamic is not None:
            bias_out = self.bias_dynamic(gdn_out).view(*orig_shape[:-1], self.out_features)
            out = out + bias_out

        if use_cache:
            return out, past_key_values
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, rank={self.rank}, "
            f"act={self.factor_activation}, norm_input={self.normalize_input}, "
            f"gdn_mode={self.gdn2.mode}, decay_rate={self.decay_rate.item():.4f}, "
            f"mode={self.parameterization}"
        )
