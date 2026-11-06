"""统一模型接口。

所有模型继承 ``BaseLayoutModel``, forward 返回统一 dict:

    {
        "feas_logit":  [B]          必有 (feasibility 分类 logit)
        "score_pred":  [B]          必有 (layout_score 回归)
        "xy_pred":     [B, N, 2]    可选 (proposal 模型, 归一化 [-1,1])
        "xy_logvar":   [B, N, 2]    可选 (SAGPN)
        "region_logits": [B, R]     可选 (SAGPN)
        "kl":          scalar       可选 (CVAE)
    }

区分:
    is_generator = False -> scorer  (MLP/DeepSets/SetTransformer/GCN/GAT/
                                     Transformer/PointNet)
    is_generator = True  -> proposal(CVAE/Diffusion/SAGPN); 需实现 ``propose``。
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from .. import features as F


class BaseLayoutModel(nn.Module):
    is_generator: bool = False

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    def propose(self, batch: Dict, k: int) -> torch.Tensor:
        """生成式模型: 给定条件 batch, 返回 [k, N, 2] 归一化 xy proposals。

        scorer 模型不实现该方法 (由 infer.py 走"采样+打分"路径)。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 不是生成式模型, 请使用 scorer 推理路径。")

    def propose_structured(
        self,
        batch: Dict,
        k: int,
        seed: int | None = None,
        temperature: float = 1.0,
    ) -> list:
        """Optional structured proposal API.  Legacy generators may omit this."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement propose_structured().")


def mlp(dims: List[int], act=nn.ReLU, last_act=False, dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = (i == len(dims) - 2)
        if not is_last or last_act:
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ScorerHead(nn.Module):
    """共享的 feasibility + score 双头。"""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = mlp([in_dim, hidden, hidden], last_act=True)
        self.feas = nn.Linear(hidden, 1)
        self.score = nn.Linear(hidden, 1)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.trunk(h)
        return {
            "feas_logit": self.feas(z).squeeze(-1),
            "score_pred": torch.sigmoid(self.score(z)).squeeze(-1),
        }


PART_DIM = F.PART_FEATURE_DIM
GLOBAL_DIM = F.GLOBAL_FEATURE_DIM
EDGE_DIM = F.EDGE_FEATURE_DIM

MAX_REGIONS = F.MAX_REGIONS  # region 分类头固定输出维度


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x[B,N,D], mask[B,N] -> [B,D]。"""
    m = mask.unsqueeze(-1)
    s = (x * m).sum(dim=1)
    cnt = m.sum(dim=1).clamp_min(1.0)
    return s / cnt


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1)
    neg = torch.finfo(x.dtype).min
    xm = x.masked_fill(m == 0, neg)
    return xm.max(dim=1).values


def static_only(node_feat: torch.Tensor, static_mask: torch.Tensor) -> torch.Tensor:
    """把依赖 staging 的动态维置零 (proposal 模型输入)。"""
    return node_feat * static_mask.view(1, 1, -1)


def masked_global(batch: Dict) -> torch.Tensor:
    """生成式模型用的全局特征: 屏蔽泄漏装配站绝对位置的维度。

    若 batch 未提供 global_static_mask, 则退回原始 global_feat (向后兼容)。
    """
    g = batch["global_feat"]
    m = batch.get("global_static_mask")
    if m is None:
        return g
    return g * m.view(1, -1)
