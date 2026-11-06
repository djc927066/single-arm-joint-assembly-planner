"""Model 3: Set Transformer Baseline (轻量版)。

每个 part 作为 token, 用 self-attention 建模 part 之间交互, 再池化输出
feasibility 与 layout_score。用于对比 attention-based set model 是否优于 DeepSets。

说明: 这里用标准 MultiheadAttention 堆叠实现 SAB (Set Attention Block) 的轻量
等价形式; 保留完整接口, 后续可替换为带 inducing points 的 ISAB。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import (BaseLayoutModel, ScorerHead, mlp, masked_mean,
                   PART_DIM, GLOBAL_DIM)


class _SAB(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = mlp([dim, dim, dim], last_act=False)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff(x))
        return x


class SetTransformer(BaseLayoutModel):
    is_generator = False

    def __init__(self, hidden: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, **_):
        super().__init__()
        self.proj = nn.Linear(PART_DIM, hidden)
        self.blocks = nn.ModuleList([_SAB(hidden, heads, dropout) for _ in range(layers)])
        self.head = ScorerHead(hidden + GLOBAL_DIM, hidden)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        mask = batch["node_mask"]
        kpm = (mask == 0)  # True 表示 padding, 被忽略
        x = self.proj(batch["node_feat"])
        for blk in self.blocks:
            x = blk(x, kpm)
        pooled = masked_mean(x, mask)
        h = torch.cat([pooled, batch["global_feat"]], dim=-1)
        return self.head(h)
