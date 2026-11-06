"""Model 1: MLP Surrogate Baseline。

输入 flatten layout feature, 输出 feasibility probability 与 predicted
layout_score。最简单的 baseline: 快速判断候选 layout 是否值得 evaluate_layout。
"""

from __future__ import annotations

from typing import Dict

import torch

from .base import BaseLayoutModel, ScorerHead, mlp


class MLPSurrogate(BaseLayoutModel):
    is_generator = False

    def __init__(self, flat_dim: int, hidden: int = 256, dropout: float = 0.1, **_):
        super().__init__()
        self.encoder = mlp([flat_dim, hidden, hidden], last_act=True, dropout=dropout)
        self.head = ScorerHead(hidden, hidden)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        h = self.encoder(batch["flat_feat"])
        return self.head(h)
