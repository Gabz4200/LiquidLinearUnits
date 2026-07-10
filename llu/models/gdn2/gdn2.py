# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

r"""
GDN-2 (Gated DeltaNet 2) token-mixing layer.

This module defines `GatedDeltaNet2`, the `nn.Module` that wraps the GDN-2
recurrence into a drop-in token mixer for a Transformer-style block. It
handles projections, short convolutions, gate construction, kernel dispatch,
caching for incremental decoding, and the gated output normalization.

GDN-2 replaces the scalar write-strength gate of the gated delta rule with two
independent channel-wise gates: an erase gate `b` on the key axis and a write
gate `w` on the value axis.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal, cast, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

if TYPE_CHECKING:
    from typing_extensions import Unpack

    # Mock Cache type for type checking without import
    Cache = Any


# =============================================================================
# LOCAL CACHE & PADDING UTILITIES (replacing fla imports)
# =============================================================================
def require_cache_layer_idx(module: nn.Module, past_key_values: Any) -> int | None:
    layer_idx = getattr(module, "layer_idx", None)
    if past_key_values is not None and layer_idx is None:
        raise ValueError(
            f"{module.__class__.__name__} requires `layer_idx` when `past_key_values` is provided."
        )
    return layer_idx


def get_layer_cache(module: nn.Module, past_key_values: Any) -> Any:
    layer_idx = require_cache_layer_idx(module, past_key_values)
    if past_key_values is not None and layer_idx is not None and len(past_key_values) > layer_idx:
        return past_key_values[layer_idx]
    return None


def update_layer_cache(module: nn.Module, past_key_values: Any, **kwargs: Any) -> Any:
    layer_idx = require_cache_layer_idx(module, past_key_values)
    if past_key_values is not None:
        return past_key_values.update(layer_idx=layer_idx, **kwargs)
    return None


def get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    lens = attention_mask.sum(-1)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(lens.max().item())
    cu_seqlens = F.pad(torch.cumsum(lens, dim=0), (1, 0)).to(torch.int32)
    return indices, cu_seqlens, max_seqlen_in_batch


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return x[indices]


def pad_input(
    hidden_states: torch.Tensor, indices: torch.Tensor, batch_size: int, seq_len: int
) -> torch.Tensor:
    D = hidden_states.shape[-1]
    output = torch.zeros(
        batch_size * seq_len, D, device=hidden_states.device, dtype=hidden_states.dtype
    )
    output[indices] = hidden_states
    return output.view(batch_size, seq_len, D)


# =============================================================================
# CPU-COMPATIBLE MODULE FALLBACKS (replacing Triton/CUDA modules)
# =============================================================================
class FusedRMSNormSwishGate(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x_normed = F.rms_norm(x, (self.weight.shape[0],), self.weight, self.eps)
        return x_normed * F.silu(g)


class ShortConvolution(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = "silu",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = (kernel_size,)
        self.activation = activation
        self.groups = hidden_size
        self.padding = (kernel_size - 1,)
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, D = x.shape
        W = self.kernel_size[0]

        if cu_seqlens is not None:
            N = len(cu_seqlens) - 1
            y_list = []
            new_caches = [] if (output_final_state or cache is not None) else None
            for i in range(N):
                start = int(cu_seqlens[i].item())
                end = int(cu_seqlens[i + 1].item())
                seq_len = end - start
                if seq_len == 0:
                    if new_caches is not None:
                        new_caches.append(x.new_zeros(1, D, W))
                    continue
                x_i = x[:, start:end]
                cache_i = cache[i : i + 1] if cache is not None else None
                y_i, new_cache_i = self._forward_single_sequence(
                    x_i, cache=cache_i, output_final_state=output_final_state
                )
                y_list.append(y_i)
                if new_caches is not None:
                    assert new_cache_i is not None
                    new_caches.append(new_cache_i.squeeze(0))
            y = torch.cat(y_list, dim=1)
            final_cache = None
            if new_caches is not None:
                final_cache = torch.stack(new_caches, dim=0)
            return y, final_cache
        else:
            return self._forward_single_sequence(
                x, cache=cache, output_final_state=output_final_state
            )

    def _forward_single_sequence(
        self,
        x: torch.Tensor,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, D = x.shape
        W = self.kernel_size[0]

        x_t = x.transpose(1, 2)
        if cache is not None:
            history = cache[:, :, -(W - 1) :]
        else:
            history = x.new_zeros(B, D, W - 1)

        x_padded = torch.cat([history, x_t], dim=-1)
        y_conv = F.conv1d(
            x_padded,
            self.weight,
            bias=self.bias,
            stride=1,
            padding=0,
            dilation=1,
            groups=self.groups,
        )
        y = y_conv.transpose(1, 2)
        if self.activation == "silu":
            y = F.silu(y)

        new_cache = None
        if output_final_state or cache is not None:
            new_cache = x_padded[:, :, -W:]
        return y, new_cache

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size[0]


# =============================================================================
# RECURRENT & CHUNK ATTENTION FALLBACKS
# =============================================================================
def _prepare_initial_state(
    initial_state: torch.Tensor | None,
    transpose: bool,
    default_shape: tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if initial_state is not None:
        if transpose:
            return initial_state.transpose(-1, -2).clone().float()
        return initial_state.clone().float()
    return torch.zeros(default_shape, device=device, dtype=torch.float32)


def chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if scale is None:
        scale = q.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = q / torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    q_scaled = q * scale

    device = q.device
    dtype = q.dtype
    q_f = q_scaled.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    b_f = b.float()
    w_f = w.float()

    B, T, H, K = q_f.shape
    V = v_f.shape[-1]

    if cu_seqlens is not None:
        N_seq = int(len(cu_seqlens) - 1)
        o_f = torch.zeros(B, T, H, V, device=device, dtype=torch.float32)
        S = _prepare_initial_state(initial_state, transpose_state_layout, (N_seq, H, K, V), device)
        final_states_list = [] if output_final_state else None
        for i in range(N_seq):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            seq_len = end - start
            if seq_len == 0:
                if output_final_state:
                    if final_states_list is not None:
                        final_states_list.append(S[i])
                continue
            S_i = S[i]
            q_i = q_f[0, start:end]
            k_i = k_f[0, start:end]
            v_i = v_f[0, start:end]
            g_i = g_f[0, start:end]
            b_i = b_f[0, start:end]
            w_i = w_f[0, start:end]
            o_seq = torch.zeros(seq_len, H, V, device=device, dtype=torch.float32)
            for t in range(seq_len):
                q_t = q_i[t]
                k_t = k_i[t]
                v_t = v_i[t]
                g_t = g_i[t]
                b_t = b_i[t]
                w_t = w_i[t]
                decay = torch.exp(g_t).unsqueeze(-1)
                S_i = S_i * decay
                bk_t = b_t * k_t
                erase_d = torch.sum(S_i * bk_t.unsqueeze(-1), dim=1)
                v_new = w_t * v_t - erase_d
                S_i = S_i + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
                o_t = torch.sum(S_i * q_t.unsqueeze(-1), dim=1)
                o_seq[t] = o_t
            o_f[0, start:end] = o_seq
            if output_final_state:
                if final_states_list is not None:
                    final_states_list.append(S_i)
        o = o_f.to(dtype)
        final_state = None
        if output_final_state:
            assert final_states_list is not None
            final_state = torch.stack(final_states_list, dim=0)
            if transpose_state_layout:
                final_state = final_state.transpose(-1, -2)
            final_state = final_state.to(dtype)
        return o, final_state
    else:
        o_f = torch.zeros(B, T, H, V, device=device, dtype=torch.float32)
        S = _prepare_initial_state(initial_state, transpose_state_layout, (B, H, K, V), device)
        for t in range(T):
            q_t = q_f[:, t]
            k_t = k_f[:, t]
            v_t = v_f[:, t]
            g_t = g_f[:, t]
            b_t = b_f[:, t]
            w_t = w_f[:, t]
            decay = torch.exp(g_t).unsqueeze(-1)
            S = S * decay
            bk_t = b_t * k_t
            erase_d = torch.sum(S * bk_t.unsqueeze(-1), dim=2)
            v_new = w_t * v_t - erase_d
            S = S + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
            o_t = torch.sum(S * q_t.unsqueeze(-1), dim=2)
            o_f[:, t] = o_t
        o = o_f.to(dtype)
        final_state = None
        if output_final_state:
            final_state = S
            if transpose_state_layout:
                final_state = final_state.transpose(-1, -2)
            final_state = final_state.to(dtype)
        return o, final_state


def fused_recurrent_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return chunk_gdn2(
        q=q,
        k=k,
        v=v,
        g=g,
        b=b,
        w=w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        transpose_state_layout=transpose_state_layout,
    )


# =============================================================================
# GatedDeltaNet2 MODULE
# =============================================================================
class GatedDeltaNet2(nn.Module):
    """
    Gated DeltaNet 2 (GDN-2) layer implementation.

    GDN-2 extends KDA's scalar-beta erase gate to channel-wise erase and write
    gates:

        S_t = (I - k_t (b_t ⊙ k_t)^T) Diag(exp(g_t)) S_{t-1}
              + k_t (w_t ⊙ v_t)^T

    Here b_t ∈ R^{d_k} is the channel-wise erase gate (replacing KDA's scalar
    beta_t) and w_t ∈ R^{d_v} is the channel-wise write gate (new in GDN-2).
    Setting b_t = beta_t · 1 and w_t = beta_t · 1 recovers KDA exactly.

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dimension. Default: 1.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 128.
        num_heads (int, Optional):
            The number of heads. Default: 16.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA (Grouped Value Attention) is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which GDN-2 kernel to use. Available: `chunk` (training + long inference)
            and `fused_recurrent` (token-by-token decode, inference only).
            The layer automatically falls back to `fused_recurrent` for short
            inference sequences (q_len <= 64); otherwise `self.mode` is used.
            Default: `chunk`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the
            erase gate `b` will be multiplied by 2.
            See reference:
            [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 1,
        head_dim: int = 128,
        num_heads: int = 16,
        num_v_heads: int | None = None,
        mode: Literal["chunk", "fused_recurrent"] = "chunk",
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int | None = None,
        norm_eps: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(
            self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5
        ):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormSwishGate.",
            )
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        # Query / key / value projections.
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Optional depthwise short convolutions on q, k, v. These give the
        # model a small local receptive field before the recurrence and are
        # standard in the gated delta rule family.
        if use_short_conv:
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )

        # Decay-gate projection. Produces the pre-activation that, combined
        # with A_log and dt_bias below, yields the channel-wise log-decay g.
        self.f_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.key_dim, bias=False),
        )

        # GDN-2 channel-wise gates. b_proj produces the erase gate on the key
        # axis; w_proj produces the write gate on the value axis. Together
        # they replace the single scalar write-strength gate of KDA.
        self.b_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.w_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        # Decay-gate parameters. A_log is a per-head log-rate; dt_bias is a
        # per-channel bias initialized so the softplus step-size starts in a
        # small range. Both are excluded from weight decay.
        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16))
        )
        setattr(self.A_log, "_no_weight_decay", True)
        dt = torch.exp(
            torch.rand(self.key_dim, dtype=torch.float32) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        setattr(self.dt_bias, "_no_weight_decay", True)

        # Output path: SiLU-gated RMS norm followed by the output projection.
        self.g_proj = nn.Sequential(
            nn.Linear(hidden_size, self.head_v_dim, bias=False),
            nn.Linear(self.head_v_dim, self.value_dim, bias=True),
        )
        self.o_norm = FusedRMSNormSwishGate(self.head_v_dim, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module) -> None:
        """Xavier-uniform init for all linear layers, applied via `self.apply`.

        The `_is_hf_initialized` guard makes this idempotent so that weights
        loaded by HuggingFace `from_pretrained` are not overwritten.
        """
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2**-2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        module._is_hf_initialized = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        """Run the GDN-2 token mixer.

        Projects the input to q/k/v and the three gates, dispatches to the
        chunkwise or recurrent kernel, updates the incremental-decoding cache,
        and applies the gated output normalization and projection.

        Args:
            hidden_states: input of shape `[B, T, hidden_size]`.
            attention_mask: optional `[B, T]` 0/1 padding mask. When given,
                the batch is unpadded into a single packed sequence and
                repadded on the way out.
            past_key_values: optional cache holding the recurrent state and
                short-convolution state from previous steps.
            use_cache: whether to write the updated state back into the cache.
            output_attentions: unused; kept for interface compatibility.

        Returns:
            A tuple `(o, None, past_key_values)` where `o` has shape
            `[B, T, hidden_size]`. The second element is always `None`
            (GDN-2 has no attention map to return).
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # Short non-training sequences use the lower-latency recurrent kernel;
        # training and long sequences use the chunkwise kernel.
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = get_layer_cache(self, past_key_values)

        indices = None
        cu_seqlens = cast(Any, kwargs.get("cu_seqlens"))
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if self.use_short_conv:
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=bool(use_cache),
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=bool(use_cache),
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=bool(use_cache),
                cu_seqlens=cu_seqlens,
            )
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        # Channel-wise log-decay, computed in fp32 for numerical stability of
        # the downstream cumulative sum. A_log is per-head and broadcast over
        # the head's key channels; dt_bias is per-channel.
        g = -self.A_log.float().exp().repeat_interleave(self.head_k_dim) * F.softplus(
            self.f_proj(hidden_states).float() + self.dt_bias
        )

        # GDN-2 gates, both squashed to [0, 1] by a sigmoid. b is the
        # channel-wise erase gate (key axis); w is the channel-wise write
        # gate (value axis).
        b = self.b_proj(hidden_states).sigmoid()
        w = self.w_proj(hidden_states).sigmoid()

        # Split the flat projection outputs into per-head tensors. Key-side
        # tensors (q, k, g, b) use head_k_dim; value-side (v, w) use head_v_dim.
        q, k, g = (rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim) for x in (q, k, g))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
        b = rearrange(b, "... (h d) -> ... h d", d=self.head_k_dim)
        w = rearrange(w, "... (h d) -> ... h d", d=self.head_v_dim)

        # Grouped value attention: when there are more value heads than key
        # heads, replicate the key-side tensors across each value-head group.
        if self.num_v_heads > self.num_heads:
            q, k, g, b = (
                repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads)
                for x in (q, k, g, b)
            )

        # Optionally lift the erase gate from [0, 1] into [0, 2], which allows
        # negative eigenvalues in the state transition (extra state-tracking
        # capacity). The write gate w is left in [0, 1].
        if self.allow_neg_eigval:
            b = b * 2.0

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if mode == "chunk":
            res_chunk = chunk_gdn2(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b,
                w=w,
                initial_state=recurrent_state,
                output_final_state=bool(use_cache),
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
            o, recurrent_state = res_chunk[0], res_chunk[1]
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gdn2(
                q=q,
                k=k,
                v=v,
                g=g,
                b=b,
                w=w,
                initial_state=recurrent_state,
                output_final_state=bool(use_cache),
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # Persist the recurrent state and short-conv state for the next
        # incremental-decoding step.
        update_layer_cache(
            self,
            past_key_values,
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
            offset=q_len,
        )

        # SiLU-gated RMS norm on the recurrent output, then project back to
        # the model dimension. Repad if the input batch was unpadded above.
        o = self.o_norm(
            o, rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
        )
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            assert indices is not None
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values
