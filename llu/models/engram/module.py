r"""Engram layer implementation for DeepSeek Engram conditional memory."""

from __future__ import annotations

import math
from typing import Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import EngramConfig
from .hash import NgramHashMapping
from .storage import EngramEmbeddingStore


class ShortConv(nn.Module):
    """Short 1D depthwise convolution with per-group RMSNorm and optional SiLU."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        self.hidden_size = hidden_size

        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)])

        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input:  (B, L, HC_MULT, D)

        Output: (B, L, HC_MULT, D)
        """
        b, t, g, c = x.shape
        if g != self.hc_mult:
            raise ValueError(f"Input groups {g} != hc_mult {self.hc_mult}")

        normed_chunks = []
        for i in range(g):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))

        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :t]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(b, t, g, c).contiguous()
        return y


class MultiHeadEmbedding(nn.Module):
    """Multi-head embedding wrapper using EngramEmbeddingStore."""

    def __init__(
        self,
        list_of_N: list[int],
        D: int,
        storage_backend: Literal["auto", "cpu", "disk", "cuda"] = "auto",
        disk_dir: str | None = None,
        target_device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.store = EngramEmbeddingStore(
            list_of_N=list_of_N,
            D=D,
            storage_backend=storage_backend,
            disk_dir=disk_dir,
            target_device=target_device,
            dtype=dtype,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.store(input_ids)


class Engram(nn.Module):
    """DeepSeek Engram memory module for static N-gram lookup and context gating.

    Computes O(1) N-gram hashes, retrieves offloaded multi-head embeddings, gates
    retrievals using the dynamic hidden state, and returns both additive output
    and conditioning representation (``engram_cond``).
    """

    def __init__(
        self,
        layer_id: int,
        cfg: EngramConfig | None = None,
        hidden_size: int = 1024,
        hc_mult: int = 4,
        kernel_size: int = 4,
        storage_backend: Literal["auto", "cpu", "disk", "cuda"] = "auto",
        disk_dir: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.cfg = cfg or EngramConfig(layer_ids=[layer_id])
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        self.hash_mapping = NgramHashMapping(cfg=self.cfg)

        vocab_across = self.hash_mapping.vocab_size_across_layers.get(layer_id, [])
        list_of_N = [x for head_list in vocab_across for x in head_list]
        if not list_of_N:
            # Fallback default if layer_id not in initial vocab map
            list_of_N = [1000] * (self.cfg.n_head_per_ngram * (self.cfg.max_ngram_size - 1))

        d_head = max(1, self.cfg.n_embed_per_ngram // max(1, self.cfg.n_head_per_ngram))
        backend_choice = cast(
            Literal["auto", "cpu", "disk", "cuda"],
            storage_backend
            if storage_backend != "auto"
            else getattr(self.cfg, "storage_backend", "auto"),
        )

        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=list_of_N,
            D=d_head,
            storage_backend=backend_choice,
            disk_dir=disk_dir or self.cfg.disk_dir,
            target_device=self.device,
            dtype=self.dtype,
        )

        self.short_conv = ShortConv(
            hidden_size=self.hidden_size,
            kernel_size=kernel_size,
            dilation=self.cfg.max_ngram_size,
            hc_mult=self.hc_mult,
        )

        engram_hidden_size = (self.cfg.max_ngram_size - 1) * self.cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(
            engram_hidden_size, self.hidden_size, device=self.device, dtype=self.dtype
        )
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(
                    engram_hidden_size,
                    self.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.hc_mult)
            ]
        )
        self.norm1 = nn.ModuleList(
            [
                nn.RMSNorm(self.hidden_size, device=self.device, dtype=self.dtype)
                for _ in range(self.hc_mult)
            ]
        )
        self.norm2 = nn.ModuleList(
            [
                nn.RMSNorm(self.hidden_size, device=self.device, dtype=self.dtype)
                for _ in range(self.hc_mult)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for Engram module.

        Args:
            hidden_states: (B, D), (B, T, D) or (B, T, HC_MULT, D)
            input_ids: (B, T) token IDs

        Returns:
            output: (B, D), (B, T, HC_MULT, D) or (B, T, D) additive tensor
            engram_cond: (B, D) or (B, T, D) conditioning vector for LLU modules
        """
        was_2d = hidden_states.ndim == 2
        if was_2d:
            hidden_states = hidden_states.unsqueeze(1)

        was_3d = hidden_states.ndim == 3
        if was_3d:
            h_4d = hidden_states.unsqueeze(2).expand(-1, -1, self.hc_mult, -1)
        else:
            h_4d = hidden_states

        _b, _t, _g, _d = h_4d.shape
        if isinstance(input_ids, torch.Tensor):
            ids_np = input_ids.detach().cpu().numpy()
        else:
            ids_np = input_ids

        layer_hashes = self.hash_mapping.hash(ids_np).get(self.layer_id)
        if layer_hashes is None:
            hashes_tensor = torch.zeros(
                (_b, _t, len(self.multi_head_embedding.store.list_of_N)),
                dtype=torch.long,
                device=self.device,
            )
        else:
            hashes_tensor = torch.from_numpy(layer_hashes).to(self.device, dtype=torch.long)

        embeddings = self.multi_head_embedding(hashes_tensor).flatten(start_dim=-2)
        if embeddings.shape[-1] != self.value_proj.in_features:
            embeddings = F.pad(
                embeddings, (0, max(0, self.value_proj.in_features - embeddings.shape[-1]))
            )[..., : self.value_proj.in_features]

        # Match time dimension if input_ids sequence differs from hidden_states sequence length
        if embeddings.shape[1] != _t:
            embeddings = embeddings[:, -_t:, :]

        gates = []
        for hc_idx in range(self.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = h_4d[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(self.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)

        gates_tensor = torch.stack(gates, dim=2)  # (B, T, HC_MULT, 1)
        value = gates_tensor * self.value_proj(embeddings).unsqueeze(2)
        output = value + self.short_conv(value)

        engram_cond = output.mean(dim=2)  # (B, T, D)
        if was_3d:
            output_ret = output.mean(dim=2)
        else:
            output_ret = output

        if was_2d:
            output_ret = output_ret.squeeze(1)
            engram_cond = engram_cond.squeeze(1)

        return output_ret, engram_cond
