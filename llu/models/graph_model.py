r"""Graph model built entirely from LLU layers (``llu.models.llns``).

``GraphLiquidNet`` is a graph Transformer whose every linear map -- the
self-attention Q/K/V/O projections, the SwiGLU FFN maps, and the
global readout projections -- is an LLU from :mod:`llu.models.llns`
instead of ``nn.Linear``. This lets us benchmark the *same* LLU family
used in the sequence models on non-sequence (graph / set) tasks, isolating
the same mechanism questions (recurrent state capacity, gating, input-
adaptive weights) in a structural setting.

Design (graph-agnostic, CPU-friendly):

* Each graph is padded to ``N_max`` nodes with an adjacency mask. There is no
  canonical order: attention is computed over *all* nodes with a graph bias
  that blocks disconnected pairs (and padded nodes), so the model is
  permutation-agnostic up to the node features.
* Message aggregation is dense attention (cheap at the tens-of-nodes scale
  these synthetic probes use), which subsumes the GAT-style "attend to the
  neighbour whose key matches my query" test (the GATv2 dictionary task).
* The global readout is a **Pixel-Wise Attention** summary: a learned query
  attends over the node states via LLU projections and pools them into a
  single graph embedding -- the direct graph analog of needle-in-a-haystack
  for the pooling operation.

Only the plain LLU variants (``StableLiquidLN`` and friends) are offered as
the graph linear maps; ``CrossAttnLoraLN`` is intentionally excluded
because its cross-attention source would degenerate to ``cond = x`` here.
"""

from __future__ import annotations

import inspect
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .llns import (
    StableLiquidLN,
    RankRLiquidLN,
    SharedMomentumLiquidLN,
    BatchMomentumLiquidLN,
)

# Plain LLU variants that are sensible as graph linear maps.
GRAPH_LLN_REGISTRY = {
    "StableLiquidLN": StableLiquidLN,
    "RankRLiquidLN": RankRLiquidLN,
    "SharedMomentumLiquidLN": SharedMomentumLiquidLN,
    "BatchMomentumLiquidLN": BatchMomentumLiquidLN,
}


def _lln_kwargs(cls: type, *, in_f: int, out_f: int, rank: int,
                parameterization: str, normalize_input: bool = True) -> dict:
    """Build kwargs for an LLU projection, forwarding only accepted params."""
    params = set(inspect.signature(cls.__init__).parameters)
    kw: dict = {"in_features": in_f, "out_features": out_f}
    if "rank" in params:
        kw["rank"] = rank
    if "parameterization" in params:
        kw["parameterization"] = parameterization
    if "normalize_input" in params:
        kw["normalize_input"] = normalize_input
    if "cond_dim" in params:
        # cond defaults to x; keep the cond dimension explicit (= in_features).
        kw["cond_dim"] = in_f
    return kw


class GraphAttnBlock(nn.Module):
    """One pre-norm graph-attention block: LLU self-attn + LLU SwiGLU FFN."""

    def __init__(self, hidden: int, n_heads: int, lln_cls: type, rank: int,
                 parameterization: str, swiglu_mult: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = hidden // n_heads
        self.norm1 = nn.RMSNorm(hidden)
        self.q_proj = lln_cls(**_lln_kwargs(lln_cls, in_f=hidden, out_f=hidden, rank=rank, parameterization=parameterization))
        self.k_proj = lln_cls(**_lln_kwargs(lln_cls, in_f=hidden, out_f=hidden, rank=rank, parameterization=parameterization))
        self.v_proj = lln_cls(**_lln_kwargs(lln_cls, in_f=hidden, out_f=hidden, rank=rank, parameterization=parameterization))
        self.o_proj = lln_cls(**_lln_kwargs(lln_cls, in_f=hidden, out_f=hidden, rank=rank, parameterization=parameterization))
        self.norm2 = nn.RMSNorm(hidden)
        self.ffn_gate = lln_cls(**_lln_kwargs(lln_cls, in_f=hidden, out_f=hidden * swiglu_mult, rank=rank, parameterization=parameterization))
        self.ffn_up = lln_cls(**_lln_kwargs(lln_cls, in_f=hidden, out_f=hidden * swiglu_mult, rank=rank, parameterization=parameterization))
        self.ffn_down = lln_cls(**_lln_kwargs(lln_cls, in_f=hidden * swiglu_mult, out_f=hidden, rank=rank, parameterization=parameterization))

    def _attn(self, h: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        # h: (B, N, H) ; attn_bias: (B, N, N) with -inf blocking edges.
        B, N, H = h.shape
        q = self.q_proj(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)  # (B,H,N,d)
        k = self.k_proj(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / (self.d_head ** 0.5) + attn_bias.unsqueeze(1)
        attn = scores.softmax(dim=-1)
        ctx = (attn @ v).transpose(1, 2).reshape(B, N, H)
        return self.o_proj(ctx)

    def forward(self, h: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        h = h + self._attn(self.norm1(h), attn_bias)
        f = self.ffn_gate(self.norm2(h))
        u = self.ffn_up(self.norm2(h))
        h = h + self.ffn_down(F.silu(f) * u)
        return h


class GraphLiquidNet(nn.Module):
    """Graph Transformer with LLU linear maps throughout.

    Args:
        node_dim: input node-feature dimension.
        hidden: model width.
        n_layers: number of graph-attention blocks.
        n_heads: attention heads.
        lln_cls: plain LLU class used for every linear map.
        out_dim: readout width.
        readout: ``"graph"`` (pool to one embedding) or ``"nodes"``
            (per-node output). Per-node is used by node-level tasks.
    """

    def __init__(self, node_dim: int, hidden: int = 64, n_layers: int = 3,
                 n_heads: int = 4, lln_cls: type = StableLiquidLN,
                 parameterization: str = "lora", rank: int = 4,
                 out_dim: int = 1, readout: str = "graph",
                 swiglu_mult: int = 4) -> None:
        super().__init__()
        self.readout_mode = readout
        self.embed = lln_cls(**_lln_kwargs(lln_cls, in_f=node_dim, out_f=hidden, rank=rank, parameterization=parameterization))
        self.blocks = nn.ModuleList([
            GraphAttnBlock(hidden, n_heads, lln_cls, rank, parameterization, swiglu_mult)
            for _ in range(n_layers)
        ])
        # Pixel-Wise Attention global readout.
        self.read_query = nn.Parameter(torch.zeros(hidden))
        self.read_k = lln_cls(**_lln_kwargs(lln_cls, in_f=hidden, out_f=hidden, rank=rank, parameterization=parameterization))
        self.read_v = lln_cls(**_lln_kwargs(lln_cls, in_f=hidden, out_f=hidden, rank=rank, parameterization=parameterization))
        self.graph_head = lln_cls(**_lln_kwargs(lln_cls, in_f=hidden, out_f=out_dim, rank=rank, parameterization=parameterization))
        self.node_head = lln_cls(**_lln_kwargs(lln_cls, in_f=hidden, out_f=out_dim, rank=rank, parameterization=parameterization))

    def _attn_bias(self, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        # adj: (B, N, N) float 1=edge, 0=no edge. node_mask: (B, N) real nodes.
        # Block disconnected pairs; exclude padded nodes as keys.
        B, N, _ = adj.shape
        bias = torch.where(adj > 0.5, torch.zeros_like(adj), torch.full_like(adj, float("-inf")))
        key_mask = node_mask.unsqueeze(1).expand(B, N, N)  # (B, N, N) True for real keys
        bias = bias.masked_fill(~key_mask, float("-inf"))
        return bias

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor,
                readout: Optional[str] = None) -> torch.Tensor:
        # x: (B, N, node_dim) ; adj: (B, N, N) ; node_mask: (B, N)
        mode = readout or self.readout_mode
        h = self.embed(x)
        bias = self._attn_bias(adj, node_mask)
        for blk in self.blocks:
            h = blk(h, bias)
        if mode == "nodes":
            out = self.node_head(h)                       # (B, N, out_dim)
            return out
        # Graph-level Pixel-Wise Attention pooling.
        B, N, H = h.shape
        Q = self.read_query.view(1, 1, H).expand(B, 1, H)  # (B, 1, H)
        K = self.read_k(h)                              # (B, N, H)
        V = self.read_v(h)                             # (B, N, H)
        scores = (Q @ K.transpose(-1, -2)) / (H ** 0.5)  # (B, 1, N)
        key_mask = node_mask.unsqueeze(1)             # (B, 1, N)
        scores = scores.masked_fill(~key_mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        pooled = (attn @ V).squeeze(1)             # (B, H)
        # Zero out pooled embedding for fully-padded (shouldn't happen) safety.
        out = self.graph_head(pooled)                 # (B, out_dim)
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["GraphLiquidNet", "GRAPH_LLN_REGISTRY"]
