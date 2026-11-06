"""Model 10: SAGPN (Ours) — Sequence-Aware Graph Proposal Network。

目标: 不是简单预测已有 layout 好不好, 而是**根据装配图结构主动生成**更有希望的
candidate layout。

输入:
    - assembly sequence / part geometry / goal poses / workspace bounds
    - candidate assembly regions
    - grasp-related features
    - (可选) current layout context
图结构 (node=part):
    - edge = assembly order relation
    - edge = parent-child assembly relation
    - edge = spatial relation
输出:
    1. assembly station xy (连续回归)  (station_pred, 归一化 [-1,1])
    2. per-part xy proposal mean      (xy_pred)
    3. per-part xy proposal variance  (xy_logvar)
    4. feasibility probability        (feas_logit)
    5. predicted layout_score         (score_pred)

装配站已从"固定 3x3 网格分类"升级为**连续 xy 回归**: 装配站可落在桌面连续可行域
的任意位置, 比离散网格更精细, 且与 per-part xy 回归口径一致。

推理: 回归装配站 xy -> 设定装配站 -> 按 (mean, sigma) 采样 top-K per-part xy
-> evaluate_layout。纯 PyTorch 稠密消息传递实现 (无 torch_geometric 依赖)。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import (BaseLayoutModel, ScorerHead, mlp, masked_mean, masked_max,
                   static_only, masked_global, PART_DIM, GLOBAL_DIM, EDGE_DIM)


class _EdgeGatedMP(nn.Module):
    """带边特征 (含 order/parent/spatial 类型) 的稠密消息传递层。"""

    def __init__(self, dim: int, edge_dim: int = EDGE_DIM):
        super().__init__()
        self.msg = mlp([dim + edge_dim, dim, dim], last_act=True)
        self.upd = mlp([2 * dim, dim, dim], last_act=True)

    def forward(self, h: torch.Tensor, edge_attr: torch.Tensor,
                adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        h_j = h.unsqueeze(1).expand(B, N, N, D)                     # 源节点 j
        m = self.msg(torch.cat([h_j, edge_attr], dim=-1))          # [B,N,N,D]
        gate = (adj * mask.unsqueeze(1)).unsqueeze(-1)             # 有效邻居
        m = m * gate
        deg = gate.sum(dim=2).clamp_min(1.0)
        agg = m.sum(dim=2) / deg                                    # 邻居均值聚合
        return self.upd(torch.cat([h, agg], dim=-1))


class SAGPN(BaseLayoutModel):
    is_generator = True

    def __init__(self, hidden: int = 128, layers: int = 3, dropout: float = 0.1, **_):
        super().__init__()
        self.in_proj = mlp([PART_DIM, hidden, hidden], last_act=True, dropout=dropout)
        self.mp = nn.ModuleList([_EdgeGatedMP(hidden) for _ in range(layers)])
        self.pool_dim = 2 * hidden + GLOBAL_DIM
        # 连续装配站 xy 回归头 (取代离散 region 分类头)
        self.station_head = mlp([self.pool_dim, hidden, 2], last_act=False)
        # per-node xy proposal 头 (mean + logvar)
        self.xy_mean = mlp([hidden + GLOBAL_DIM, hidden, 2], last_act=False)
        self.xy_logvar = mlp([hidden + GLOBAL_DIM, hidden, 2], last_act=False)
        # feasibility + score 头
        self.head = ScorerHead(self.pool_dim, hidden)

    def _encode(self, batch: Dict, use_static_only: bool):
        node = batch["node_feat"]
        if use_static_only:
            node = static_only(node, batch["static_mask"])
        gfeat = masked_global(batch)   # 屏蔽装配站绝对位置, 避免站位回归泄漏
        h = self.in_proj(node)
        mask = batch["node_mask"]
        for layer in self.mp:
            h = layer(h, batch["edge_attr"], batch["adj"], mask)
        pooled = torch.cat([masked_mean(h, mask), masked_max(h, mask), gfeat], dim=-1)
        return h, mask, pooled

    def _xy_heads(self, h, gfeat):
        B, N, _ = h.shape
        gc = gfeat.unsqueeze(1).expand(B, N, gfeat.shape[-1])
        z = torch.cat([h, gc], dim=-1)
        return torch.tanh(self.xy_mean(z)), self.xy_logvar(z).clamp(-8, 4)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        # 训练时用 static-only 输入 (proposal 生成器不能看 staging), 保证与推理一致。
        h, mask, pooled = self._encode(batch, use_static_only=True)
        out = self.head(pooled)
        out["station_pred"] = torch.tanh(self.station_head(pooled))  # 连续装配站 xy [-1,1]
        xy_mean, xy_logvar = self._xy_heads(h, masked_global(batch))
        out["xy_pred"] = xy_mean
        out["xy_logvar"] = xy_logvar
        return out

    @torch.no_grad()
    def propose(self, batch: Dict, k: int) -> torch.Tensor:
        h, mask, pooled = self._encode(batch, use_static_only=True)
        xy_mean, xy_logvar = self._xy_heads(h, masked_global(batch))
        std = torch.exp(0.5 * xy_logvar)
        outs = [xy_mean]  # 第一个用均值 (最可能)
        for _ in range(max(0, k - 1)):
            outs.append(torch.clamp(xy_mean + std * torch.randn_like(std), -1.0, 1.0))
        return torch.stack(outs, dim=0).squeeze(1)  # [k, N, 2]

    @torch.no_grad()
    def predict_station(self, batch: Dict) -> torch.Tensor:
        """回归装配站 xy (归一化 [-1,1]), [B, 2]。"""
        _, _, pooled = self._encode(batch, use_static_only=True)
        return torch.tanh(self.station_head(pooled))
