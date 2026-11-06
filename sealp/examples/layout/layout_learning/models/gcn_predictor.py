"""Model 4: GCN Layout Predictor Baseline (纯 PyTorch, 稠密邻接)。

装配任务建成图 (node=part, edge=order/parent/spatial), 用 GCN 卷积聚合邻居,
输出 feasibility 与 layout_score。为避免 torch_geometric 依赖, 采用稠密邻接矩阵
实现 (零件数很小, 完全可行)。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import (BaseLayoutModel, ScorerHead, masked_mean, masked_max,
                   PART_DIM, GLOBAL_DIM)


def _norm_adj(adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对称归一化稠密邻接 (含自环)。adj[B,N,N], mask[B,N]。"""
    m = mask.unsqueeze(1) * mask.unsqueeze(2)   # [B,N,N] 有效节点对
    a = adj * m
    eye = torch.eye(a.shape[-1], device=a.device).unsqueeze(0)
    a = a + eye * mask.unsqueeze(-1)            # 自环
    deg = a.sum(-1).clamp_min(1e-6)
    dinv = deg.pow(-0.5)
    return dinv.unsqueeze(-1) * a * dinv.unsqueeze(1)


class _GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return torch.bmm(a_norm, self.lin(h))


class GCNPredictor(BaseLayoutModel):
    is_generator = False

    def __init__(self, hidden: int = 128, layers: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        self.in_proj = nn.Linear(PART_DIM, hidden)
        self.convs = nn.ModuleList([_GCNLayer(hidden, hidden) for _ in range(layers)])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.head = ScorerHead(2 * hidden + GLOBAL_DIM, hidden)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        mask = batch["node_mask"]
        a_norm = _norm_adj(batch["adj"], mask)
        h = self.act(self.in_proj(batch["node_feat"]))
        for conv in self.convs:
            h = self.drop(self.act(conv(h, a_norm)))
        pooled = torch.cat([masked_mean(h, mask), masked_max(h, mask)], dim=-1)
        z = torch.cat([pooled, batch["global_feat"]], dim=-1)
        return self.head(z)
