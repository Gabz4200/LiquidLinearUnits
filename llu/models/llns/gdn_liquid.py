r"""GDNLiquidLN: Liquid Linear Unit with a Gated DeltaNet 2 (GDN-2) block."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Any, Literal

from llu.models.gdn2 import GatedDeltaNet2
from .utils import (
    DEVICE,
    _activate,
    _init_hypernetwork,
    _zero_b_section,
    _small_init,
    _zero_out_last,
    _FreezeMixin,
)


class GDNLiquidLN(_FreezeMixin, nn.Module):
    """Liquid Linear Unit utilizing a Gated DeltaNet 2 (GDN-2) block.

    The hypernetwork is a stateful GDN-2 block which processes the conditioning
    input sequence. The accumulated context shapes the low-rank factor matrices,
    forming a dynamic sequence-dependent linear transformation.

    Zero-initialised so the adaptive path contributes nothing at step 1.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        # GDN-2 specific parameters
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
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        r"""__init__(in_features, out_features, rank=4, expand_v=1.0, head_dim=16, num_heads=4, num_v_heads=None, mode="chunk", use_short_conv=True, allow_neg_eigval=False, conv_size=4, conv_bias=False, layer_idx=None, norm_eps=1e-5, bias=True, dynamic_bias=False, factor_activation="norm", scale_init=0.01, normalize_input=True, init_method="hyperfan_in", device=None, dtype=torch.float32) -> None

        Args:
            in_features (int): size of each input sample.
            out_features (int): size of each output sample.
            rank (int): number of factor pairs :math:`(a_r, b_r)`. Default: ``4``.
            expand_v (float): expansion ratio for value dimension in GDN-2. Default: ``1.0``.
            head_dim (int): dimension of each GDN-2 head. Default: ``16``.
            num_heads (int): number of GDN-2 heads. Default: ``4``.
            num_v_heads (int, optional): number of heads for value projection in GDN-2. Default: ``None``.
            mode (str): GDN-2 mode ("chunk" or "fused_recurrent"). Default: ``"chunk"``.
            use_short_conv (bool): whether GDN-2 uses short convolutions. Default: ``True``.
            allow_neg_eigval (bool): allow negative eigenvalues in GDN-2. Default: ``False``.
            conv_size (int): kernel size of short convolution. Default: ``4``.
            conv_bias (bool): whether short convolution has bias. Default: ``False``.
            layer_idx (int, optional): layer index for caching. Default: ``None``.
            norm_eps (float): epsilon for GDN-2 normalization. Default: ``1e-5``.
            bias (bool): whether the core :class:`~nn.Linear` has a learnable bias. Default: ``True``.
            dynamic_bias (bool): if ``True``, a sequence-dependent bias from GDN-2 is added. Default: ``False``.
            factor_activation (str): activation for the factor vectors. Default: ``"norm"``.
            scale_init (float): initial value of scale multiplier. Default: ``0.01``.
            normalize_input (bool): if ``True``, apply RMSNorm to the conditioning input before GDN-2. Default: ``True``.
            init_method (str): weight initialisation method for proj_out. Default: ``"hyperfan_in"``.
            device (torch.device, optional): device of parameters. Default: ``None``.
            dtype (torch.dtype): data type of parameters. Default: ``torch.float32``.
        """
        super().__init__()
        if rank < 1:
            raise ValueError(f"rank must be >= 1, got {rank}")
        dev = device if device is not None else DEVICE

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.factor_activation = factor_activation
        self.normalize_input = normalize_input
        self.init_method = init_method

        self.linear_core = nn.Linear(in_features, out_features, bias=bias, device=dev, dtype=dtype)

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

        # Output projection from GDN-2 features to low-rank factors
        self.proj_out = nn.Linear(
            in_features,
            rank * (out_features + in_features),
            bias=True,
            device=dev,
            dtype=dtype,
        )

        # Scaling dial
        self.scale = nn.Parameter(torch.full((out_features,), scale_init, device=dev, dtype=dtype))
        self.rank_scale = nn.Parameter(torch.full((rank,), 1.0, device=dev, dtype=dtype))

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

        # Initialize the projection layer
        _init_hypernetwork(
            self.proj_out,
            self.init_method,
            self.in_features,
            self.out_features,
            rank=self.rank,
        )

        # Zero b-section; a-factors keep gradient flowing
        _zero_b_section(self.proj_out, self.rank * self.out_features)

        if self.bias_dynamic is not None:
            _small_init(self.bias_dynamic)
            _zero_out_last(self.bias_dynamic)

    def forward(
        self,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        r"""forward(x, cond=None, attention_mask=None, past_key_values=None, use_cache=False) -> Tensor or (Tensor, Cache)

        Args:
            x (Tensor): input tensor of shape ``(..., in_features)``.
            cond (Tensor, optional): optional conditioning tensor. When
                ``None``, *x* is used.
            attention_mask (Tensor, optional): optional padding mask of shape ``(B, T)``.
            past_key_values (Any, optional): optional key-value cache for state tracking.
            use_cache (bool): whether to return the updated cache. Default: ``False``.

        Returns:
            Tensor or (Tensor, Cache): sequence output of shape ``(..., out_features)``,
            optionally with the updated state cache if ``use_cache`` is ``True``.
        """
        cond = cond if cond is not None else x

        # RMSNorm for magnitude invariance
        h_in = F.rms_norm(cond, (self.in_features,)) if self.normalize_input else cond

        # Save original shape to reconstruct output shape
        orig_shape = h_in.shape

        # Prepare 3D shape for GatedDeltaNet2 (batch, seq_len, dim)
        if len(orig_shape) == 1:
            h_in_3d = h_in.unsqueeze(0).unsqueeze(0)
        elif len(orig_shape) == 2:
            h_in_3d = h_in.unsqueeze(1)
        else:
            h_in_3d = h_in.flatten(0, -3)

        # Run GatedDeltaNet2 sequence model
        gdn_out, _, past_key_values = self.gdn2(
            h_in_3d,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Project GDN-2 output to the low-rank factors
        raw = self.proj_out(gdn_out)

        # Reshape raw back to original leading dimensions
        raw = raw.view(*orig_shape[:-1], self.rank * (self.out_features + self.in_features))

        core_out = self.linear_core(x)

        split = self.rank * self.out_features
        a_raw = raw[..., :split].reshape(*orig_shape[:-1], self.rank, self.out_features)
        b_raw = raw[..., split:].reshape(*orig_shape[:-1], self.rank, self.in_features)

        a = _activate(a_raw, self.factor_activation)
        b = _activate(b_raw, self.factor_activation)

        dot = torch.matmul(b, x.unsqueeze(-1)).squeeze(-1)  # (..., rank)
        dot = dot * self.rank_scale

        adaptive = torch.matmul(dot.unsqueeze(-2), a).squeeze(-2)  # (..., O)

        out = core_out + adaptive * self.scale

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
            f"gdn_mode={self.gdn2.mode}"
        )
