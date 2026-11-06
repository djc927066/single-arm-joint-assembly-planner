"""SeqRel vs DeepSets 定向失败诊断 (纯推理, 不训练, 不改模型)。

在完全相同的 validation samples (共享 split) 上重新推理两个 best checkpoint,
输出预测退化 / 消息传递过平滑 / 多任务损失冲突 / ranking&fail 标签质量诊断。

用法:
    python -m sealp.examples.layout.layout_learning._seqrel_failure_diagnosis

输出:
    sealp/examples/layout/_output/seqrel_diagnostics/
        seqrel_failure_diagnosis.json
        seqrel_failure_diagnosis.csv
        seqrel_failure_samples.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as Fn

_THIS = os.path.dirname(os.path.abspath(__file__))
_LAYOUT = os.path.dirname(_THIS)
if _LAYOUT not in sys.path:
    sys.path.insert(0, _LAYOUT)

from layout_learning import features as F
from layout_learning.dataset import collate_items, load_jsonl, move_batch, sample_to_item
from layout_learning.losses import LossWeights, _pair_rank_loss, _focal_bce
from layout_learning.models import build_model
from layout_learning.models.seqrel_layout_net import _STAGING_SLICE


# ------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------

def _quantiles(x: np.ndarray, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)) -> Dict[str, float]:
    if x.size == 0:
        return {f"q{int(q*100)}": float("nan") for q in qs}
    return {f"q{int(q*100)}": float(np.quantile(x, q)) for q in qs}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 1e-9 else float("nan")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 1e-9 else float("nan")


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n < 3:
        return float("nan")
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    total = conc + disc
    return float((conc - disc) / total) if total > 0 else float("nan")


def _roc_auc(prob: np.ndarray, y: np.ndarray) -> float:
    pos = prob[y > 0.5]; neg = prob[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(prob)
    ranks = np.empty(len(prob), float); ranks[order] = np.arange(1, len(prob) + 1)
    r_pos = ranks[y > 0.5].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _pr_auc(prob: np.ndarray, y: np.ndarray) -> float:
    n_pos = int((y > 0.5).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-prob)
    yy = (y[order] > 0.5).astype(float)
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy)
    prec = tp / np.maximum(tp + fp, 1e-9); rec = tp / n_pos
    ap = 0.0; prev = 0.0
    for p_i, r_i in zip(prec, rec):
        ap += (r_i - prev) * p_i; prev = r_i
    return float(ap)


def _brier(prob: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((prob - y) ** 2)) if prob.size else float("nan")


def _ece(prob: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    if prob.size == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (prob >= lo) & (prob < hi if i < n_bins - 1 else prob <= hi)
        if m.sum() == 0:
            continue
        conf = prob[m].mean(); acc = (y[m] > 0.5).mean()
        ece += (m.sum() / prob.size) * abs(acc - conf)
    return float(ece)


def _hist_overlap(pos: np.ndarray, neg: np.ndarray, n_bins: int = 20) -> float:
    """正负概率分布直方图重叠系数 (0=完全分离, 1=完全重叠)。"""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    hp, _ = np.histogram(pos, bins=edges, density=False)
    hn, _ = np.histogram(neg, bins=edges, density=False)
    hp = hp / max(hp.sum(), 1); hn = hn / max(hn.sum(), 1)
    return float(np.minimum(hp, hn).sum())


def _confusion(prob: np.ndarray, y: np.ndarray, thr: float = 0.5) -> Dict[str, int]:
    pred = prob > thr; pos = y > 0.5
    return {
        "tp": int((pred & pos).sum()),
        "fp": int((pred & ~pos).sum()),
        "tn": int((~pred & ~pos).sum()),
        "fn": int((~pred & pos).sum()),
    }


# ------------------------------------------------------------
# 推理: 得到全 val 集 logit / prob / score
# ------------------------------------------------------------

def _load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], flat_dim=ckpt["flat_dim"],
                        **ckpt.get("model_kwargs", {})).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _infer_all(model, samples: List[Dict], val_idx: List[int], max_parts: int,
               feature_version: str, device: str, batch_size: int = 64):
    logits, scores = [], []
    for start in range(0, len(val_idx), batch_size):
        chunk = val_idx[start:start + batch_size]
        items = [sample_to_item(samples[i], max_parts, feature_version) for i in chunk]
        batch = move_batch(collate_items(items), device)
        with torch.no_grad():
            out = model(batch)
        logits.append(out["feas_logit"].cpu().numpy().ravel())
        scores.append(out["score_pred"].cpu().numpy().ravel())
    return np.concatenate(logits), np.concatenate(scores)


def _feasibility_diag(logit: np.ndarray, score_pred: np.ndarray,
                      y: np.ndarray) -> Dict:
    prob = 1.0 / (1.0 + np.exp(-logit))
    pos_l = logit[y > 0.5]; neg_l = logit[y <= 0.5]
    pos_p = prob[y > 0.5]; neg_p = prob[y <= 0.5]

    def _stats(a):
        return {"mean": float(a.mean()) if a.size else float("nan"),
                "std": float(a.std()) if a.size else float("nan"),
                "min": float(a.min()) if a.size else float("nan"),
                "max": float(a.max()) if a.size else float("nan")}

    return {
        "n_val": int(len(y)),
        "n_pos": int((y > 0.5).sum()),
        "n_neg": int((y <= 0.5).sum()),
        "logit_pos": _stats(pos_l),
        "logit_neg": _stats(neg_l),
        "prob_mean": float(prob.mean()),
        "prob_std": float(prob.std()),
        "prob_quantiles": _quantiles(prob),
        "prob_pos_quantiles": _quantiles(pos_p),
        "prob_neg_quantiles": _quantiles(neg_p),
        "roc_auc": _roc_auc(prob, y),
        "pr_auc": _pr_auc(prob, y),
        "brier": _brier(prob, y),
        "ece": _ece(prob, y),
        "confusion_matrix": _confusion(prob, y),
        "pos_neg_prob_overlap": _hist_overlap(pos_p, neg_p),
    }


def _score_diag(score_pred: np.ndarray, y: np.ndarray,
                true_score: np.ndarray) -> Dict:
    fmask = y > 0.5
    sp = score_pred[fmask]; st = true_score[fmask]
    if sp.size < 2:
        return {"n_feasible": int(fmask.sum()), "note": "feasible < 2"}
    pred_std = float(sp.std()); true_std = float(st.std())
    slope = float(np.polyfit(st, sp, 1)[0]) if true_std > 1e-9 else float("nan")
    collapse = pred_std < 0.3 * true_std
    return {
        "n_feasible": int(fmask.sum()),
        "true_mean": float(st.mean()), "true_std": true_std,
        "pred_mean": float(sp.mean()), "pred_std": pred_std,
        "pred_over_true_std_ratio": float(pred_std / true_std) if true_std > 1e-9 else float("nan"),
        "mae": float(np.abs(sp - st).mean()),
        "rmse": float(np.sqrt(((sp - st) ** 2).mean())),
        "spearman": _spearman(sp, st),
        "pearson": _pearson(sp, st),
        "kendall_tau": _kendall_tau(sp, st),
        "regression_slope": slope,
        "pred_range": [float(sp.min()), float(sp.max())],
        "true_range": [float(st.min()), float(st.max())],
        "mean_collapse": bool(collapse),
    }


def _sample_id(sample: Dict, idx: int) -> int:
    sid = sample.get("sample_id") or sample.get("id")
    if sid is not None:
        return int(sid)
    return int(idx)


def _topk_diag(ds_score: np.ndarray, sq_score: np.ndarray,
               ds_logit: np.ndarray, sq_logit: np.ndarray,
               y: np.ndarray, true_score: np.ndarray,
               val_idx: List[int], samples: List[Dict], k: int = 10) -> Tuple[Dict, List[Dict]]:
    def _topk_by(pred_prob, tag):
        top = np.argsort(-pred_prob)[:k]
        return {
            f"{tag}_top{k}_feasible_count": int((y[top] > 0.5).sum()),
            f"{tag}_top{k}_true_scores": [round(float(true_score[i]), 4) for i in top],
            f"{tag}_top{k}_mean_true_score": float(true_score[top].mean()),
            f"{tag}_top{k}_indices": [int(val_idx[i]) for i in top],
        }
    ds_prob = 1 / (1 + np.exp(-ds_logit)); sq_prob = 1 / (1 + np.exp(-sq_logit))
    ds_top = set(np.argsort(-ds_prob)[:k].tolist())
    sq_top = set(np.argsort(-sq_prob)[:k].tolist())
    out = {}
    out.update(_topk_by(ds_prob, "deepsets"))
    out.update(_topk_by(sq_prob, "seqrel"))
    out["top10_overlap_count"] = int(len(ds_top & sq_top))

    # SeqRel 排错最严重 20 个 (feasible 内, 预测分数秩 vs 真实分数秩)
    fmask = y > 0.5
    fidx = np.nonzero(fmask)[0]
    worst = []
    if fidx.size >= 3:
        sp = sq_score[fidx]; st = true_score[fidx]
        rank_pred = np.argsort(np.argsort(sp))
        rank_true = np.argsort(np.argsort(st))
        err = np.abs(rank_pred - rank_true)
        order = np.argsort(-err)[:20]
        for o in order:
            gi = fidx[o]
            si = val_idx[gi]
            worst.append({
                "sample_id": _sample_id(samples[si], si),
                "val_index": int(si),
                "true_score": round(float(true_score[gi]), 4),
                "seqrel_pred_score": round(float(sq_score[gi]), 4),
                "rank_error": int(err[o]),
            })
    out["seqrel_worst_misrank_count"] = len(worst)
    return out, worst


# ------------------------------------------------------------
# SeqRel 消息传递诊断
# ------------------------------------------------------------

def _seqrel_layer_trace(model, batch: Dict) -> Dict:
    """手动重放 SeqRel encode, 捕获每层 node embedding, 统计 oversmoothing。"""
    node = batch["node_feat"]; mask = batch["node_mask"]
    edge_attr = batch["edge_attr"]; adj = batch["adj"]
    rel_attr, pair_gate = model._pair_context(node, edge_attr, adj, mask)
    h = model.encoder(node) * mask.unsqueeze(-1)
    layers = [("encoder", h.detach().clone())]
    residual_ratios = []
    for li, layer in enumerate(model.mp):
        B, N, D = h.shape
        h_j = h.unsqueeze(1).expand(B, N, N, D)
        m = layer.msg(torch.cat([h_j, rel_attr], dim=-1))
        gate = pair_gate.unsqueeze(-1)
        m = m * gate
        deg = gate.sum(dim=2).clamp_min(1.0)
        agg = m.sum(dim=2) / deg
        upd = layer.upd(torch.cat([h, agg], dim=-1))
        # 消息强度 vs residual
        msg_norm = float(upd.detach().norm(dim=-1).mean().item())
        res_norm = float(h.detach().norm(dim=-1).mean().item())
        pre_var = float((h + upd).detach().var().item())
        h = layer.norm(h + layer.drop(upd)) * mask.unsqueeze(-1)
        post_var = float(h.detach().var().item())
        residual_ratios.append({
            "layer": li,
            "message_norm": msg_norm,
            "residual_norm": res_norm,
            "message_over_residual": float(msg_norm / max(res_norm, 1e-9)),
            "layernorm_pre_var": pre_var,
            "layernorm_post_var": post_var,
        })
        layers.append((f"mp{li}", h.detach().clone()))

    def _emb_stats(h_t, mask_t):
        stats = []
        B, N, D = h_t.shape
        for b in range(B):
            m = mask_t[b] > 0.5
            valid = h_t[b][m]
            if valid.shape[0] < 2:
                continue
            fstd = float(valid.std().item())
            nrm = Fn.normalize(valid, dim=-1)
            cos = nrm @ nrm.t()
            off = cos[~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)]
            dist = torch.cdist(valid.unsqueeze(0), valid.unsqueeze(0)).squeeze(0)
            doff = dist[~torch.eye(dist.shape[0], dtype=torch.bool, device=dist.device)]
            stats.append((fstd, float(off.mean().item()), float(doff.mean().item())))
        if not stats:
            return {"feature_std": float("nan"), "mean_cosine": float("nan"),
                    "mean_pairwise_dist": float("nan")}
        arr = np.array(stats)
        return {"feature_std": float(arr[:, 0].mean()),
                "mean_cosine": float(arr[:, 1].mean()),
                "mean_pairwise_dist": float(arr[:, 2].mean())}

    layer_stats = {name: _emb_stats(h_t, mask) for name, h_t in layers}
    last = list(layer_stats.values())[-1]
    first = layer_stats["encoder"]
    oversmoothing = (last["mean_cosine"] > 0.95) or \
                    (last["feature_std"] < 0.5 * first["feature_std"])
    return {
        "per_layer_embedding": layer_stats,
        "message_strength": residual_ratios,
        "oversmoothing": bool(oversmoothing),
        "cosine_first_to_last": [first["mean_cosine"], last["mean_cosine"]],
        "std_first_to_last": [first["feature_std"], last["feature_std"]],
    }


def _masking_test(model, samples: List[Dict], val_idx: List[int], max_parts: int,
                  feature_version: str, device: str) -> Dict:
    """受控输入遮蔽: 比较 full / relation0 / sequence0 / staging0 的输出。"""
    items = [sample_to_item(samples[i], max_parts, feature_version) for i in val_idx]
    base = move_batch(collate_items(items), device)

    def _run(mod_fn):
        b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in base.items()}
        mod_fn(b)
        with torch.no_grad():
            out = model(b)
        return (out["feas_logit"].cpu().numpy().ravel(),
                out["score_pred"].cpu().numpy().ravel())

    full_l, full_s = _run(lambda b: None)

    def _zero_relation(b):        # 关系边属性全部置零 (edge_attr)
        b["edge_attr"] = torch.zeros_like(b["edge_attr"])

    def _zero_sequence(b):        # 仅顺序/父子边标志置零 (edge_attr dim 3,4) + adj
        ea = b["edge_attr"]
        if ea.shape[-1] >= 5:
            ea[..., 3] = 0.0; ea[..., 4] = 0.0
        b["adj"] = torch.zeros_like(b["adj"])

    def _zero_staging(b):         # staging_xy (node_feat 18:20) 置零
        b["node_feat"][..., _STAGING_SLICE] = 0.0

    rel_l, rel_s = _run(_zero_relation)
    seq_l, seq_s = _run(_zero_sequence)
    stg_l, stg_s = _run(_zero_staging)

    def _delta(a_l, a_s):
        return {"mean_abs_logit_delta": float(np.abs(a_l - full_l).mean()),
                "max_abs_logit_delta": float(np.abs(a_l - full_l).max()),
                "mean_abs_score_delta": float(np.abs(a_s - full_s).mean()),
                "max_abs_score_delta": float(np.abs(a_s - full_s).max())}

    return {
        "relation_zeroed": _delta(rel_l, rel_s),
        "sequence_edges_zeroed": _delta(seq_l, seq_s),
        "staging_xy_zeroed": _delta(stg_l, stg_s),
        "note": "delta 相对 full; 越大表示该输入对输出越重要",
    }


# ------------------------------------------------------------
# 多任务损失 & 梯度冲突
# ------------------------------------------------------------

def _history_loss_diag(history_csv: str, weights: LossWeights) -> Dict:
    rows = []
    with open(history_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return {}

    def col(name):
        return np.array([float(r.get(name, "nan") or "nan") for r in rows])

    cls = col("train_l_cls"); score = col("train_l_score")
    rank = col("train_l_rank"); fail = col("train_l_fail")
    # 加权
    w_cls = cls; w_score = weights.alpha * score
    w_rank = weights.rank_weight * rank; w_fail = weights.fail_weight * fail
    total = w_cls + w_score + w_rank + w_fail
    epochs = col("epoch").astype(int)
    comp = col("composite"); pr = col("pr_auc"); sp = col("score_spearman")

    def _frac(w):
        return float(np.nanmean(w / np.maximum(total, 1e-9)))

    best_comp_ep = int(epochs[np.nanargmax(comp)])
    best_pr_ep = int(epochs[np.nanargmax(pr)])
    best_sp_ep = int(epochs[np.nanargmax(sp)])
    return {
        "weighted_loss_fraction": {
            "cls": _frac(w_cls), "score": _frac(w_score),
            "rank": _frac(w_rank), "fail": _frac(w_fail),
        },
        "rank_gt_main_epochs": int(np.sum(w_rank > w_cls)),
        "fail_gt_main_epochs": int(np.sum(w_fail > w_cls)),
        "rank_gt_main_fraction": float(np.mean(w_rank > w_cls)),
        "fail_gt_main_fraction": float(np.mean(w_fail > w_cls)),
        "best_composite_epoch": best_comp_ep,
        "best_pr_auc_epoch": best_pr_ep,
        "best_spearman_epoch": best_sp_ep,
        "best_comp_eq_best_pr": best_comp_ep == best_pr_ep,
        "best_comp_eq_best_spearman": best_comp_ep == best_sp_ep,
        "mean_weighted_loss": {
            "cls": float(np.nanmean(w_cls)), "score": float(np.nanmean(w_score)),
            "rank": float(np.nanmean(w_rank)), "fail": float(np.nanmean(w_fail)),
        },
    }


def _shared_encoder_params(model):
    ps = []
    for name, p in model.named_parameters():
        if name.startswith("encoder.") or name.startswith("mp."):
            ps.append((name, p))
    return ps


def _grad_conflict(model, samples: List[Dict], val_idx: List[int], max_parts: int,
                   feature_version: str, device: str, weights: LossWeights,
                   batch_size: int = 96) -> Dict:
    """对共享 encoder 参数, 分别计算各 task loss 的梯度并求 pairwise cosine。"""
    chunk = val_idx[:batch_size]
    items = [sample_to_item(samples[i], max_parts, feature_version) for i in chunk]
    batch = move_batch(collate_items(items), device)
    shared = _shared_encoder_params(model)

    def _grad_of(loss):
        model.zero_grad(set_to_none=True)
        g = torch.autograd.grad(loss, [p for _, p in shared],
                                retain_graph=True, allow_unused=True)
        flat = []
        for gi in g:
            flat.append(gi.detach().reshape(-1) if gi is not None
                        else torch.zeros(1, device=device))
        return torch.cat(flat)

    out = model(batch)
    feas = batch["feas"]; score = batch["score"]
    pw = torch.tensor(float(weights.pos_weight), device=device)
    feas_mask = feas > 0.5

    if weights.use_focal:
        l_cls = _focal_bce(out["feas_logit"], feas, pw, weights.focal_gamma)
    else:
        l_cls = Fn.binary_cross_entropy_with_logits(out["feas_logit"], feas, pos_weight=pw)
    l_score = (Fn.smooth_l1_loss(out["score_pred"][feas_mask], score[feas_mask])
               if feas_mask.any() else torch.zeros((), device=device))
    l_rank = _pair_rank_loss(out["score_pred"], score, feas_mask,
                             batch["group_id"], weights)
    ft = batch["fail_class"].long()
    l_fail = Fn.cross_entropy(out["fail_logits"], ft, ignore_index=-1)
    if not torch.isfinite(l_fail):
        l_fail = torch.zeros((), device=device)

    grads = {}
    for name, loss in [("cls", l_cls), ("score", l_score),
                       ("rank", l_rank), ("fail", l_fail)]:
        try:
            grads[name] = _grad_of(loss)
        except RuntimeError:
            grads[name] = torch.zeros(1, device=device)

    def _cos(a, b):
        if a.numel() != b.numel():
            n = min(a.numel(), b.numel()); a = a[:n]; b = b[:n]
        na = a.norm(); nb = b.norm()
        if na < 1e-9 or nb < 1e-9:
            return float("nan")
        return float((a @ b / (na * nb)).item())

    model.zero_grad(set_to_none=True)
    pairs = [("cls", "score"), ("cls", "rank"), ("cls", "fail"),
             ("score", "rank"), ("score", "fail")]
    return {f"cos_{a}_{b}": _cos(grads[a], grads[b]) for a, b in pairs}


# ------------------------------------------------------------
# ranking & fail 标签质量
# ------------------------------------------------------------

def _label_diag(samples: List[Dict], val_idx: List[int], weights: LossWeights) -> Dict:
    val = [samples[i] for i in val_idx]
    groups: Dict[str, List[int]] = {}
    fail_counts = {c: 0 for c in F.FAIL_CLASSES}
    fail_missing = 0
    for j, s in enumerate(val):
        gk = F.ranking_group_key(s)
        groups.setdefault(gk, []).append(j)
        if not s.get("l2_pass", False):
            fc = F.fail_reason_class(s)
            if 0 <= fc < len(F.FAIL_CLASSES):
                fail_counts[F.FAIL_CLASSES[fc]] += 1
            else:
                fail_missing += 1

    # ranking pair 统计 (仅 feasible, 同组, score gap >= min_gap)
    feas_by_group: Dict[str, List[float]] = {}
    for s in val:
        if s.get("l2_pass", False):
            feas_by_group.setdefault(F.ranking_group_key(s), []).append(
                float(s.get("layout_score", 0.0)))
    total_pairs = valid_pairs = near_pairs = 0
    min_gap = float(weights.rank_min_score_gap)
    for scs in feas_by_group.values():
        n = len(scs)
        if n < 2:
            continue
        for a in range(n):
            for b in range(a + 1, n):
                total_pairs += 1
                if abs(scs[a] - scs[b]) >= min_gap:
                    valid_pairs += 1
                else:
                    near_pairs += 1

    group_sizes = [len(v) for v in groups.values()]
    n_fail = sum(fail_counts.values())
    rare = [c for c, n in fail_counts.items() if 0 < n < max(1, int(0.05 * max(n_fail, 1)))]
    return {
        "n_groups": len(groups),
        "mean_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "max_group_size": int(max(group_sizes)) if group_sizes else 0,
        "n_singleton_groups": int(sum(1 for g in group_sizes if g == 1)),
        "feasible_ranking_total_pairs": total_pairs,
        "feasible_ranking_valid_pairs": valid_pairs,
        "feasible_ranking_near_pairs": near_pairs,
        "near_pair_fraction": float(near_pairs / max(total_pairs, 1)),
        "valid_pair_fraction": float(valid_pairs / max(total_pairs, 1)),
        "fail_class_distribution": fail_counts,
        "n_fail_samples": n_fail,
        "rare_fail_classes": rare,
        "fail_label_missing": fail_missing,
        "rank_min_score_gap": min_gap,
    }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    root = os.path.abspath(os.path.join(_LAYOUT, "..", "..", ".."))
    repro = os.path.join(root, "checkpoints", "layout_models_repro")
    out_dir = os.path.join(root, "sealp", "examples", "layout", "_output",
                           "seqrel_diagnostics")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=os.path.join(
        root, "sealp", "examples", "layout", "_output", "layout_dataset_v2.jsonl"))
    p.add_argument("--deepsets-checkpoint", default=os.path.join(
        repro, "deepsets", "stratified", "seed0", "deepsets_best.pt"))
    p.add_argument("--seqrel-checkpoint", default=os.path.join(
        repro, "seqrel", "stratified", "seed0", "seqrel_best.pt"))
    p.add_argument("--split-indices", default=os.path.join(
        repro, "_splits", "stratified", "seed0", "split_indices.json"))
    p.add_argument("--feature-version", default="v2")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for pth in (args.dataset, args.deepsets_checkpoint, args.seqrel_checkpoint,
                args.split_indices):
        if not os.path.isfile(pth):
            raise FileNotFoundError(pth)

    samples = load_jsonl(args.dataset)
    with open(args.split_indices, "r", encoding="utf-8") as f:
        split = json.load(f)
    val_idx = [int(i) for i in split["val_indices"]]

    y = np.array([1.0 if samples[i].get("l2_pass", False) else 0.0 for i in val_idx])
    true_score = np.array([float(samples[i].get("layout_score", 0.0))
                           if samples[i].get("l2_pass", False) else 0.0 for i in val_idx])

    ds_model, ds_ckpt = _load_model(args.deepsets_checkpoint, device)
    sq_model, sq_ckpt = _load_model(args.seqrel_checkpoint, device)
    max_parts = int(sq_ckpt.get("max_parts", F.MAX_PARTS_DEFAULT))

    ds_logit, ds_score = _infer_all(ds_model, samples, val_idx, max_parts,
                                    args.feature_version, device)
    sq_logit, sq_score = _infer_all(sq_model, samples, val_idx, max_parts,
                                    args.feature_version, device)

    # SeqRel 训练 loss 权重 (来自 config)
    weights = LossWeights(
        alpha=1.0,
        rank_weight=float(sq_ckpt.get("rank_weight", 0.5)),
        fail_weight=float(sq_ckpt.get("fail_weight", 0.2)),
        use_focal=bool(sq_ckpt.get("use_focal", True)),
        focal_gamma=float(sq_ckpt.get("focal_gamma", 2.0)),
        pos_weight=float(sq_ckpt.get("effective_pos_weight", 2.4313)),
    )

    topk, worst = _topk_diag(ds_score, sq_score, ds_logit, sq_logit,
                             y, true_score, val_idx, samples)

    # SeqRel 消息传递 (取一个中等 batch)
    trace_idx = val_idx[:64]
    trace_items = [sample_to_item(samples[i], max_parts, args.feature_version)
                   for i in trace_idx]
    trace_batch = move_batch(collate_items(trace_items), device)
    mp_diag = _seqrel_layer_trace(sq_model, trace_batch)
    mask_diag = _masking_test(sq_model, samples, val_idx, max_parts,
                              args.feature_version, device)

    history_csv = os.path.join(os.path.dirname(args.seqrel_checkpoint),
                               "training_history.csv")
    loss_diag = _history_loss_diag(history_csv, weights) if os.path.isfile(history_csv) else {}
    grad_diag = _grad_conflict(sq_model, samples, val_idx, max_parts,
                               args.feature_version, device, weights)
    label_diag = _label_diag(samples, val_idx, weights)

    report = {
        "checkpoints": {
            "deepsets": os.path.abspath(args.deepsets_checkpoint),
            "seqrel": os.path.abspath(args.seqrel_checkpoint),
            "split_indices": os.path.abspath(args.split_indices),
            "n_val": len(val_idx),
        },
        "feasibility": {
            "deepsets": _feasibility_diag(ds_logit, ds_score, y),
            "seqrel": _feasibility_diag(sq_logit, sq_score, y),
        },
        "score_regression": {
            "deepsets": _score_diag(ds_score, y, true_score),
            "seqrel": _score_diag(sq_score, y, true_score),
        },
        "topk": topk,
        "seqrel_message_passing": mp_diag,
        "seqrel_input_masking": mask_diag,
        "seqrel_multitask_loss": loss_diag,
        "seqrel_grad_conflict": grad_diag,
        "seqrel_labels": label_diag,
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "seqrel_failure_diagnosis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 扁平 CSV (关键标量)
    flat = {
        "ds_pr_auc": report["feasibility"]["deepsets"]["pr_auc"],
        "sq_pr_auc": report["feasibility"]["seqrel"]["pr_auc"],
        "ds_ece": report["feasibility"]["deepsets"]["ece"],
        "sq_ece": report["feasibility"]["seqrel"]["ece"],
        "ds_brier": report["feasibility"]["deepsets"]["brier"],
        "sq_brier": report["feasibility"]["seqrel"]["brier"],
        "ds_prob_overlap": report["feasibility"]["deepsets"]["pos_neg_prob_overlap"],
        "sq_prob_overlap": report["feasibility"]["seqrel"]["pos_neg_prob_overlap"],
        "ds_pred_std_ratio": report["score_regression"]["deepsets"].get("pred_over_true_std_ratio"),
        "sq_pred_std_ratio": report["score_regression"]["seqrel"].get("pred_over_true_std_ratio"),
        "ds_score_collapse": report["score_regression"]["deepsets"].get("mean_collapse"),
        "sq_score_collapse": report["score_regression"]["seqrel"].get("mean_collapse"),
        "sq_oversmoothing": mp_diag["oversmoothing"],
        "sq_cos_last_layer": mp_diag["cosine_first_to_last"][1],
        "top10_overlap": topk["top10_overlap_count"],
        **{f"grad_{k}": v for k, v in grad_diag.items()},
        "n_groups": label_diag["n_groups"],
        "valid_pair_fraction": label_diag["valid_pair_fraction"],
        "near_pair_fraction": label_diag["near_pair_fraction"],
    }
    csv_path = os.path.join(out_dir, "seqrel_failure_diagnosis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat.keys()))
        w.writeheader(); w.writerow(flat)

    samples_csv = os.path.join(out_dir, "seqrel_failure_samples.csv")
    with open(samples_csv, "w", newline="", encoding="utf-8") as f:
        cols = ["sample_id", "val_index", "true_score", "seqrel_pred_score", "rank_error"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in worst:
            w.writerow(row)

    print(f"[fail-diag] wrote {json_path}")
    print(f"[fail-diag] wrote {csv_path}")
    print(f"[fail-diag] wrote {samples_csv}")
    print(f"[fail-diag] SeqRel oversmoothing={mp_diag['oversmoothing']} "
          f"score_collapse={report['score_regression']['seqrel'].get('mean_collapse')} "
          f"grad={grad_diag}")


if __name__ == "__main__":
    main()
