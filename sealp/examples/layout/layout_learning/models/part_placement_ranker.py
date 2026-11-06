"""PartPlacementRanker: per-part staging placement ranker (variable N at inference).

Inputs (per part):
  - bbox extent / footprint, init_pos, goal_pos, pose_candidates (flatsurface)
  - Global table context
  - A pool of candidate placements (xy + pose)

Output:
  - Scores over candidates → top_k ranked (xy, pose_tag, score) for **this part alone**
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .base import BaseLayoutModel, GLOBAL_DIM, mlp
from ..part_placement_dataset import MAX_CANDIDATES, POSE_FEAT_DIM

STATIC_DIM = 18


class PartPlacementRankerNet(BaseLayoutModel):
    """Listwise ranker over table staging candidates for one part."""

    is_generator = False

    def __init__(self,
                 hidden: int = 128,
                 dropout: float = 0.15,
                 max_candidates: int = MAX_CANDIDATES,
                 **kwargs):
        super().__init__()
        self.hidden = int(hidden)
        self.max_candidates = int(max_candidates)
        self.part_enc = mlp([STATIC_DIM, hidden, hidden], last_act=True, dropout=dropout)
        self.global_enc = mlp([GLOBAL_DIM, hidden // 2, hidden // 2], last_act=True, dropout=dropout)
        self.cand_enc = mlp([POSE_FEAT_DIM, hidden // 2, hidden // 2], last_act=True, dropout=dropout)
        self.score_head = nn.Sequential(
            nn.Linear(hidden + hidden // 2 + hidden // 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self,
                part_static: torch.Tensor,
                global_feat: torch.Tensor,
                cand_feat: torch.Tensor,
                cand_mask: Optional[torch.Tensor] = None,
                **_) -> Dict[str, torch.Tensor]:
        """
        part_static: [B, 18]
        global_feat: [B, 7]
        cand_feat:   [B, C, POSE_FEAT_DIM]
        cand_mask:   [B, C] 1=valid
        """
        b, c, _ = cand_feat.shape
        p = self.part_enc(part_static[:, :STATIC_DIM])
        g = self.global_enc(global_feat)
        g_exp = g.unsqueeze(1).expand(b, c, g.shape[-1])
        p_exp = p.unsqueeze(1).expand(b, c, p.shape[-1])
        ce = self.cand_enc(cand_feat.reshape(b * c, -1)).reshape(b, c, -1)
        fused = torch.cat([p_exp, g_exp, ce], dim=-1)
        logits = self.score_head(fused.reshape(b * c, -1)).reshape(b, c)
        if cand_mask is not None:
            logits = logits.masked_fill(cand_mask <= 0, -1e4)
        return {"cand_logits": logits, "cand_scores": Fn.softmax(logits, dim=-1)}

    def predict_top_k(self,
                    part_static: torch.Tensor,
                    global_feat: torch.Tensor,
                    cand_feat: torch.Tensor,
                    cand_mask: torch.Tensor,
                    k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(part_static, global_feat, cand_feat, cand_mask)
        scores = out["cand_scores"]
        k = min(int(k), scores.shape[-1])
        top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
        return top_idx, top_scores
