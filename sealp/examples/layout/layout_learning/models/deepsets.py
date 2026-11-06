"""Model 2: DeepSets Baseline。

把 layout 看成 part set, 每个 part 用 shared MLP 编码, 再 mean/max pooling
聚合, 输出 feasibility probability 与 predicted layout_score。
"""

from __future__ import annotations

from typing import Dict

import torch

from .base import (BaseLayoutModel, ScorerHead, mlp, masked_mean, masked_max,
                   PART_DIM, GLOBAL_DIM)


class DeepSets(BaseLayoutModel):
    is_generator = False

    def __init__(self, hidden: int = 128, dropout: float = 0.1, **_):
        super().__init__()
        self.phi = mlp([PART_DIM, hidden, hidden], last_act=True, dropout=dropout)
        self.rho_in = 2 * hidden + GLOBAL_DIM
        self.head = ScorerHead(self.rho_in, hidden)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        x = self.phi(batch["node_feat"])            # [B,N,H]
        mask = batch["node_mask"]
        pooled = torch.cat([masked_mean(x, mask), masked_max(x, mask)], dim=-1)
        h = torch.cat([pooled, batch["global_feat"]], dim=-1)
        return self.head(h)
