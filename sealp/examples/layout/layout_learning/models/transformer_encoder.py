"""Model 6: Transformer Layout Encoder Baseline (轻量版)。

每个零件作为 token, 加入 assembly order positional encoding, 用 Transformer
Encoder 建模全局零件关系, 输出 feasibility 与 layout_score。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import (BaseLayoutModel, ScorerHead, masked_mean, PART_DIM, GLOBAL_DIM)


class TransformerEncoderModel(BaseLayoutModel):
    is_generator = False

    def __init__(self, hidden: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, max_parts: int = 16, **_):
        super().__init__()
        self.proj = nn.Linear(PART_DIM, hidden)
        self.pos = nn.Parameter(torch.randn(max_parts, hidden) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 2,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = ScorerHead(hidden + GLOBAL_DIM, hidden)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        mask = batch["node_mask"]
        x = self.proj(batch["node_feat"])
        n = x.shape[1]
        x = x + self.pos[:n].unsqueeze(0)
        x = self.encoder(x, src_key_padding_mask=(mask == 0))
        pooled = masked_mean(x, mask)
        h = torch.cat([pooled, batch["global_feat"]], dim=-1)
        return self.head(h)
