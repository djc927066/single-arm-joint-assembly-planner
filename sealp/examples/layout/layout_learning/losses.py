"""复合损失。

L = L_cls + alpha * L_score + beta * L_xy + gamma * L_station  (+ kl_weight * KL)

- L_cls    : feasibility 分类 (BCEWithLogitsLoss)
- L_score  : layout_score 回归 (SmoothL1), 只在 feasible 样本上算
- L_xy     : xy proposal (SmoothL1), 只在"高分 feasible" 样本 & valid 零件上算
- L_station: 连续装配站 xy 回归 (SmoothL1), 只在高分 feasible 样本上算
             (若模型仍输出离散 region_logits, 则回退为 region 分类 CrossEntropy)
- KL       : CVAE 专用
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import torch
import torch.nn.functional as Fn


@dataclass
class LossWeights:
    alpha: float = 1.0     # score
    beta: float = 1.0      # xy proposal
    gamma: float = 0.5     # region
    kl: float = 0.01       # cvae kl
    # 此处始终保存已经解析后的有效浮点值；"auto" 在 train.py 完成 split 后计算。
    pos_weight: float = 3.0  # 可行样本通常偏少, 给正类加权
    score_threshold: float = 0.0  # 只有 score >= 该阈值的 feasible 样本参与 proposal 监督
    # ---- seqrel 专用 (默认 0/False, 不影响其它模型) ----
    rank_weight: float = 0.0        # pair-ranking loss 权重 (beta in spec)
    fail_weight: float = 0.0        # fail 辅助分类权重 (gamma in spec)
    rank_margin: float = 0.05       # margin ranking loss 的 margin
    rank_min_score_gap: float = 0.05  # 真实 score 差小于该值的 pair 不训练
    rank_pairs_per_batch: int = 256   # 每个 batch 最多采样多少个排序对
    use_focal: bool = False         # L_cls 是否用 focal loss
    focal_gamma: float = 2.0        # focal loss 聚焦系数
    focal_alpha: float = 0.25       # focal loss 正类权重 (metadata / 可选扩展)
    score_only: bool = False        # frozen adapter: total 仅包含 score + rank


def _focal_bce(logit: torch.Tensor, target: torch.Tensor,
               pos_weight: torch.Tensor, gamma: float) -> torch.Tensor:
    """带 pos_weight 的 focal BCE (数值稳定, 对已分对样本降权)。"""
    bce = Fn.binary_cross_entropy_with_logits(
        logit, target, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logit)
    pt = p * target + (1.0 - p) * (1.0 - target)     # 命中正确类的概率
    focal = (1.0 - pt).clamp_min(0.0) ** float(gamma) * bce
    return focal.mean()


def _pair_rank_loss(pred: torch.Tensor, score: torch.Tensor,
                    feas_mask: torch.Tensor, group_id: torch.Tensor,
                    weights: LossWeights) -> torch.Tensor:
    """同组 feasible 样本之间的 margin ranking loss。"""

    device = pred.device
    idx = torch.nonzero(feas_mask, as_tuple=False).squeeze(-1)

    if idx.numel() < 2:
        # 数值为0，但保留与pred相连的计算图
        return pred.sum() * 0.0

    gid = group_id[idx]
    sc = score[idx]
    pr = pred[idx]

    ii, jj = torch.combinations(
        torch.arange(idx.numel(), device=device),
        r=2,
    ).unbind(1)

    same_group = gid[ii] == gid[jj]
    gap = sc[ii] - sc[jj]
    keep = same_group & (
        gap.abs() >= float(weights.rank_min_score_gap)
    )

    if not bool(keep.any()):
        # 没有满足条件的排序对时仍允许backward
        return pred.sum() * 0.0

    ii, jj, gap = ii[keep], jj[keep], gap[keep]

    max_pairs = int(weights.rank_pairs_per_batch)

    if ii.numel() > max_pairs:
        sel = torch.randperm(
            ii.numel(),
            device=device,
        )[:max_pairs]

        ii, jj, gap = ii[sel], jj[sel], gap[sel]

    target = torch.sign(gap)

    return Fn.margin_ranking_loss(
        pr[ii],
        pr[jj],
        target,
        margin=float(weights.rank_margin),
    )


def compute_loss(out: Dict[str, torch.Tensor],
                 batch: Dict[str, torch.Tensor],
                 weights: LossWeights,
                 is_generator: bool) -> Dict[str, torch.Tensor]:
    device = out["feas_logit"].device
    feas = batch["feas"]
    score = batch["score"]

    # ---- feasibility ----
    pw_value = float(weights.pos_weight)
    if not math.isfinite(pw_value) or pw_value <= 0:
        raise ValueError(f"pos_weight 必须为正有限值，得到 {pw_value}")
    pw = torch.as_tensor(
        pw_value, device=device, dtype=out["feas_logit"].dtype
    )
    if weights.use_focal:
        l_cls = _focal_bce(out["feas_logit"], feas, pw, weights.focal_gamma)
    else:
        l_cls = Fn.binary_cross_entropy_with_logits(out["feas_logit"], feas, pos_weight=pw)

    # ---- score (仅 feasible) ----
    feas_mask = feas > 0.5
    if feas_mask.any():
        l_score = Fn.smooth_l1_loss(out["score_pred"][feas_mask], score[feas_mask])
    else:
        l_score = out["score_pred"].sum() * 0.0

    total = weights.alpha * l_score if weights.score_only else (
        l_cls + weights.alpha * l_score)
    logs = {"l_cls": l_cls.detach(), "l_score": l_score.detach()}

    # ---- pair ranking (scorer, 需要 group_id) ----
    if weights.rank_weight > 0 and "group_id" in batch:
        l_rank = _pair_rank_loss(out["score_pred"], score, feas_mask,
                                 batch["group_id"], weights)
        total = total + weights.rank_weight * l_rank
        logs["l_rank"] = l_rank.detach()

    # ---- fail 辅助分类 (仅 infeasible, 需要模型输出 fail_logits) ----
    if (not weights.score_only and weights.fail_weight > 0
            and "fail_logits" in out and "fail_class" in batch):
        fail_target = batch["fail_class"].long()
        l_fail = Fn.cross_entropy(out["fail_logits"], fail_target,
                                  ignore_index=-1)
        if not torch.isfinite(l_fail):   # 该 batch 全为 feasible -> CE=nan
            l_fail = torch.zeros((), device=device)
        total = total + weights.fail_weight * l_fail
        logs["l_fail"] = l_fail.detach()

    # ---- proposal 相关 (仅生成式模型) ----
    if is_generator and "xy_pred" in out:
        # 高分 feasible 样本作为 proposal target
        good = feas_mask & (score >= weights.score_threshold)
        xy_pred = out["xy_pred"]
        xy_tgt = batch["xy_target"]
        valid = batch["xy_valid"] * good.float().unsqueeze(1)   # [B,N]
        denom = valid.sum().clamp_min(1.0)
        l_xy = (Fn.smooth_l1_loss(xy_pred, xy_tgt, reduction="none").sum(-1) * valid).sum() / denom
        total = total + weights.beta * l_xy
        logs["l_xy"] = l_xy.detach()

        # 连续装配站回归 (取代固定网格 region 分类)
        if "station_pred" in out and good.any():
            st_valid = batch.get("station_valid")
            gmask = good.float()
            if st_valid is not None:
                gmask = gmask * st_valid
            denom_s = gmask.sum().clamp_min(1.0)
            l_station = (Fn.smooth_l1_loss(out["station_pred"], batch["station_target"],
                                           reduction="none").sum(-1) * gmask).sum() / denom_s
            total = total + weights.gamma * l_station
            logs["l_station"] = l_station.detach()
        elif "region_logits" in out and good.any():
            # 兼容: 仍输出离散 region 的模型
            l_region = Fn.cross_entropy(out["region_logits"][good], batch["region_target"][good])
            total = total + weights.gamma * l_region
            logs["l_region"] = l_region.detach()

        if "kl" in out:
            total = total + weights.kl * out["kl"]
            logs["kl"] = out["kl"].detach()

    logs["total"] = total.detach()
    return {"loss": total, "logs": logs}
