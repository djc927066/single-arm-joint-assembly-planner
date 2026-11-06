"""Model 7: PointNet Geometry Encoder Baseline (轻量版)。

从每个 part 的几何 (这里用 extent 合成 8 个 box 角点作为 sampled point cloud)
提取 geometry embedding, 拼接到 part feature 上, 再用 DeepSets 聚合。用于验证引入
几何特征是否有效。

TODO: 后续可替换为从真实 mesh 采样点云的 PointNetEncoder (接口已就绪:
只需把 ``_synth_points`` 换成真实点云输入即可)。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import (BaseLayoutModel, ScorerHead, mlp, masked_mean, masked_max,
                   PART_DIM, GLOBAL_DIM)

# extent 在节点特征中的位置 (features.build_part_feature: feat[0:3] = extent/LEN_SCALE)
_EXTENT_SLICE = slice(0, 3)


class _PointNetEncoder(nn.Module):
    """对每个零件的合成点云做 shared MLP + max pool。"""

    def __init__(self, emb: int = 32):
        super().__init__()
        self.point_mlp = mlp([3, 32, emb], last_act=True)

    @staticmethod
    def _synth_points(extent_half: torch.Tensor) -> torch.Tensor:
        """extent_half[B,N,3] -> box 8 角点 [B,N,8,3]。"""
        signs = torch.tensor(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=extent_half.dtype, device=extent_half.device)  # [8,3]
        return extent_half.unsqueeze(2) * signs.view(1, 1, 8, 3)

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        extent_half = node_feat[..., _EXTENT_SLICE] * 0.5
        pts = self._synth_points(extent_half)              # [B,N,8,3]
        z = self.point_mlp(pts)                            # [B,N,8,emb]
        return z.max(dim=2).values                         # [B,N,emb]


class PointNetGeometry(BaseLayoutModel):
    is_generator = False

    def __init__(self, hidden: int = 128, geo_emb: int = 32, dropout: float = 0.1, **_):
        super().__init__()
        self.pointnet = _PointNetEncoder(geo_emb)
        self.phi = mlp([PART_DIM + geo_emb, hidden, hidden], last_act=True, dropout=dropout)
        self.head = ScorerHead(2 * hidden + GLOBAL_DIM, hidden)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        geo = self.pointnet(batch["node_feat"])
        x = self.phi(torch.cat([batch["node_feat"], geo], dim=-1))
        mask = batch["node_mask"]
        pooled = torch.cat([masked_mean(x, mask), masked_max(x, mask)], dim=-1)
        h = torch.cat([pooled, batch["global_feat"]], dim=-1)
        return self.head(h)
