"""Model 5: GAT Layout Predictor Baseline (纯 PyTorch, 稠密邻接)。

与 GCN 类似, 但使用 graph attention 学习不同邻居零件对当前零件的重要程度。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .base import (BaseLayoutModel, ScorerHead, masked_mean, masked_max,
                   PART_DIM, GLOBAL_DIM)


class _GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, out_dim * heads)
        self.attn_src = nn.Parameter(torch.randn(heads, out_dim) * 0.1)
        self.attn_dst = nn.Parameter(torch.randn(heads, out_dim) * 0.1)
        self.leaky = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        H = self.heads
        wh = self.lin(h).view(B, N, H, self.out_dim)               # [B,N,H,D]
        e_src = (wh * self.attn_src.view(1, 1, H, -1)).sum(-1)     # [B,N,H]
        e_dst = (wh * self.attn_dst.view(1, 1, H, -1)).sum(-1)     # [B,N,H]
        # e[b,i,j,h] = src_i + dst_j
        e = self.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))    # [B,N,N,H]
        eye = torch.eye(N, device=h.device).view(1, N, N, 1)
        conn = ((adj.unsqueeze(-1) > 0) | (eye > 0))
        valid = (mask.view(B, 1, N, 1) > 0) & conn
        e = e.masked_fill(~valid, float("-inf"))
        alpha = torch.softmax(e, dim=2)                            # 归一化 over j
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.drop(alpha)
        # out[b,i,h,d] = sum_j alpha[b,i,j,h] * wh[b,j,h,d]
        out = torch.einsum("bijh,bjhd->bihd", alpha, wh)
        return out.reshape(B, N, H * self.out_dim)


class GATPredictor(BaseLayoutModel):
    is_generator = False

    def __init__(self, hidden: int = 64, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, **_):
        super().__init__()
        self.in_proj = nn.Linear(PART_DIM, hidden * heads)
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(_GATLayer(hidden * heads, hidden, heads, dropout))
        self.act = nn.ELU()
        emb = hidden * heads
        self.head = ScorerHead(2 * emb + GLOBAL_DIM, emb)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        mask = batch["node_mask"]
        adj = batch["adj"]
        h = self.act(self.in_proj(batch["node_feat"]))
        for layer in self.layers:
            h = self.act(layer(h, adj, mask))
        pooled = torch.cat([masked_mean(h, mask), masked_max(h, mask)], dim=-1)
        z = torch.cat([pooled, batch["global_feat"]], dim=-1)
        return self.head(z)
