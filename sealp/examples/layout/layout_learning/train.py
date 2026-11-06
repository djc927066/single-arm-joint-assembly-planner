"""统一训练循环 + 指标 + best checkpoint。

供 ``train_layout_network.py`` 调用, 也可直接:
    from layout_learning.train import train_model

设计要点 (为修复小数据集上 SAGPN 崩溃):
    - stratified / seed_holdout / random 三种 split;
    - score regression 只在 feasible 样本上算;
    - xy / station proposal loss 只在 "elite feasible" (>= 分位阈值) 上算;
    - 丰富验证指标: ROC-AUC / PR-AUC / recall@K / precision@K /
      top-K avg true score / score MAE·RMSE·Spearman / enrichment factor;
    - best checkpoint 用 composite metric, 不再只看 topk_hit;
    - early stopping + ReduceLROnPlateau。
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import features as F
from .dataset import LayoutDataset, collate_items, move_batch, load_jsonl
from .losses import LossWeights, compute_loss
from .relseqgen_losses import RelSeqGenLossWeights, compute_relseqgen_loss
from .generator_dataset import geometry_holdout_split
from .models import build_model, is_generator
from .repro_data import verify_dataset_manifest


# ------------------------------------------------------------
# 指标 (全部无 sklearn 依赖)
# ------------------------------------------------------------

def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC (Mann-Whitney U)。"""
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    r_pos = ranks[labels > 0.5].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def _pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """PR-AUC = average precision (按预测概率降序累积)。"""
    n_pos = int((labels > 0.5).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores)
    y = (labels[order] > 0.5).astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = tp / n_pos
    # AP = sum over thresholds (R_i - R_{i-1}) * P_i
    prev_r = 0.0
    ap = 0.0
    for p_i, r_i in zip(precision, recall):
        ap += (r_i - prev_r) * p_i
        prev_r = r_i
    return float(ap)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关 = 秩上的 Pearson。"""
    if len(a) < 3:
        return float("nan")

    def _rank(x):
        order = np.argsort(x)
        r = np.empty(len(x), dtype=float)
        r[order] = np.arange(len(x))
        return r

    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    if denom < 1e-9:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    aa, bb = a - a.mean(), b - b.mean()
    denom = np.sqrt((aa ** 2).sum() * (bb ** 2).sum())
    return float((aa * bb).sum() / denom) if denom > 1e-12 else float("nan")


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    da = a[:, None] - a[None, :]
    db = b[:, None] - b[None, :]
    upper = np.triu(np.ones_like(da, dtype=bool), k=1)
    valid = upper & (da != 0) & (db != 0)
    if not valid.any():
        return float("nan")
    return float(np.sign(da[valid] * db[valid]).mean())


@torch.no_grad()
def evaluate(model, loader, device, gen: bool, topk: int = 10) -> Dict[str, float]:
    model.eval()
    feas_logits, feas_labels, score_preds, score_tgts = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        feas_logits.append(out["feas_logit"].cpu().numpy())
        feas_labels.append(batch["feas"].cpu().numpy())
        score_preds.append(out["score_pred"].cpu().numpy())
        score_tgts.append(batch["score"].cpu().numpy())

    fl = np.concatenate(feas_logits)
    fy = np.concatenate(feas_labels)
    sp = np.concatenate(score_preds)
    st = np.concatenate(score_tgts)          # 真 layout_score (infeasible=0)
    prob = 1.0 / (1.0 + np.exp(-fl))
    n = len(prob)
    val_feas_rate = float((fy > 0.5).mean()) if n else float("nan")

    acc = float(((prob > 0.5) == (fy > 0.5)).mean()) if n else float("nan")
    roc = _roc_auc(prob, fy)
    pr = _pr_auc(prob, fy)

    # score 指标: 只在 feasible 上算
    fmask = fy > 0.5
    if fmask.sum() >= 1:
        mae = float(np.abs(sp[fmask] - st[fmask]).mean())
        rmse = float(np.sqrt(((sp[fmask] - st[fmask]) ** 2).mean()))
    else:
        mae = rmse = float("nan")
    spearman = _spearman(sp[fmask], st[fmask]) if fmask.sum() >= 3 else float("nan")
    pearson = _pearson(sp[fmask], st[fmask]) if fmask.sum() >= 2 else float("nan")
    kendall = _kendall_tau(sp[fmask], st[fmask]) if fmask.sum() >= 2 else float("nan")
    pred_std = float(np.std(sp[fmask])) if fmask.sum() >= 1 else float("nan")
    true_std = float(np.std(st[fmask])) if fmask.sum() >= 1 else float("nan")
    std_ratio = (
        float(pred_std / true_std)
        if np.isfinite(true_std) and true_std > 1e-12 else float("nan"))
    if fmask.sum() >= 2 and float(np.var(st[fmask])) > 1e-12:
        regression_slope = float(
            np.cov(st[fmask], sp[fmask], ddof=0)[0, 1]
            / np.var(st[fmask]))
    else:
        regression_slope = float("nan")

    # top-K (按预测可行概率排序)
    k = int(min(topk, n))
    if k > 0:
        top_prob = np.argsort(-prob)[:k]
        topk_hit = float(fy[top_prob].mean())                 # = precision@K
        precision_at_k = topk_hit
        n_pos = float((fy > 0.5).sum())
        recall_at_k = float(fy[top_prob].sum() / n_pos) if n_pos > 0 else float("nan")
        enrichment = (precision_at_k / val_feas_rate) if val_feas_rate > 1e-9 else float("nan")
    else:
        topk_hit = precision_at_k = recall_at_k = enrichment = float("nan")

    # top-K 真分数 (按预测 score 排序) 以及归一化 (相对 oracle top-K)
    if k > 0:
        top_score = np.argsort(-sp)[:k]
        topk_avg_true = float(st[top_score].mean())
        oracle = float(np.sort(st)[::-1][:k].mean())
        topk_avg_norm = float(topk_avg_true / oracle) if oracle > 1e-9 else 0.0
    else:
        topk_avg_true = topk_avg_norm = float("nan")

    def _z(v):  # nan -> 0 (用于 composite)
        return 0.0 if (v is None or np.isnan(v)) else float(v)

    composite = (0.4 * _z(pr) + 0.3 * _z(recall_at_k)
                 + 0.2 * _z(topk_avg_norm) + 0.1 * _z(spearman))

    return {
        "feas_acc": acc,
        "auc": roc,               # 保留旧名 (= ROC-AUC)
        "roc_auc": roc,
        "pr_auc": pr,
        "score_mae": mae,
        "score_rmse": rmse,
        "score_spearman": spearman,
        "score_pearson": pearson,
        "score_kendall": kendall,
        "score_pred_std": pred_std,
        "score_true_std": true_std,
        "score_std_ratio": std_ratio,
        "score_regression_slope": regression_slope,
        "topk_hit": topk_hit,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "topk_avg_true_score": topk_avg_true,
        "topk_avg_score_norm": topk_avg_norm,
        "enrichment": enrichment,
        "composite": composite,
    }


def evaluate_generator_loss(model, loader, device, model_name: str,
                            weights, rq) -> Dict[str, float]:
    """验证集生成损失 (l_xy + l_station)。

    生成器模型的价值在于 proposal 质量, 而 composite 是打分器指标, 对生成毫无意义
    (SAGPN/RelSeqGen 在单任务数据上 composite 恒定, 会把 best 选在几乎没训练的
    epoch 1)。这里用验证集上的生成损失作为选择依据, 返回其负值 ``gen_val_neg``
    以便与"越大越好"的 best 选择逻辑统一。
    """
    from .relseqgen_losses import compute_relseqgen_loss
    model.eval()
    xy_sum = 0.0
    st_sum = 0.0
    nb = 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            out = model(batch)
            if model_name == "relseqgen":
                res = compute_relseqgen_loss(out, batch, weights, rq)
            else:
                res = compute_loss(out, batch, weights, True)
            logs = res["logs"]
            xy_sum += float(logs.get("l_xy", 0.0))
            st_sum += float(logs.get("l_station", 0.0))
            nb += 1
    xy = xy_sum / max(nb, 1)
    st = st_sum / max(nb, 1)
    # 仅用 xy (零件摆放质量) 选择, 避免 station NLL 靠缩方差刷分干扰选择。
    return {
        "gen_val_xy": xy,
        "gen_val_station": st,
        "gen_val_loss": xy + st,
        "gen_val_neg": -xy,
    }


# ------------------------------------------------------------
# train / val split
# ------------------------------------------------------------

def _region_bucket(sample: Dict, mode: str, xy_grid: int,
                   xy_bounds: Tuple[float, float, float, float]) -> str:
    """给一条样本分配 region 桶标签 (供 region_holdout split 使用)。

    mode:
        rc      -> 用 assembly_region_rc (离散 3x3 网格)。
        xy_grid -> 用 assembly_station_pos 落在 xy_grid×xy_grid 网格的格子。
        auto    -> rc 有效 (非 [-1,-1]) 用 rc, 否则退回 xy_grid。
    """
    rc = sample.get("assembly_region_rc", [-1, -1])
    rc_valid = int(rc[0]) >= 0 and int(rc[1]) >= 0
    use = mode
    if mode == "auto":
        use = "rc" if rc_valid else "xy_grid"
    if use == "rc" and rc_valid:
        return f"rc_{int(rc[0])}_{int(rc[1])}"
    # xy_grid
    pos = np.asarray(sample.get("assembly_station_pos", [0.0, 0.0, 0.0]), dtype=float)
    xlo, xhi, ylo, yhi = xy_bounds
    gx = int(np.clip((pos[0] - xlo) / max(xhi - xlo, 1e-6) * xy_grid, 0, xy_grid - 1))
    gy = int(np.clip((pos[1] - ylo) / max(yhi - ylo, 1e-6) * xy_grid, 0, xy_grid - 1))
    return f"xy_{gx}_{gy}"


def _split_indices(samples: List[Dict], val_ratio: float, seed: int,
                   mode: str, region_holdout_mode: str = "auto",
                   xy_grid: int = 3) -> Tuple[List[int], List[int]]:
    rng = np.random.RandomState(seed)
    n = len(samples)
    idx = np.arange(n)

    if mode == "region_holdout":
        st = np.array([[float(v) for v in (s.get("assembly_station_pos") or [0, 0, 0])[:2]]
                       for s in samples], dtype=float)
        xy_bounds = (float(st[:, 0].min()), float(st[:, 0].max()),
                     float(st[:, 1].min()), float(st[:, 1].max())) if len(st) else (0, 1, 0, 1)
        buckets = np.array([_region_bucket(s, region_holdout_mode, xy_grid, xy_bounds)
                            for s in samples])
        uniq = sorted(set(buckets.tolist()))
        if len(uniq) >= 2:
            rng.shuffle(uniq)
            n_hold = max(1, int(round(len(uniq) * val_ratio)))
            hold = set(uniq[:n_hold])
            val = idx[np.isin(buckets, list(hold))].tolist()
            train = idx[~np.isin(buckets, list(hold))].tolist()
            if train and val:
                return sorted(train), sorted(val)
        # 桶不足 -> 退回 stratified
        mode = "stratified"

    if mode == "seed_holdout":
        seeds = np.array([int(s.get("seed", 0)) for s in samples])
        uniq = sorted(set(seeds.tolist()))
        if len(uniq) >= 2:
            n_hold = max(1, int(round(len(uniq) * val_ratio)))
            hold = set(uniq[-n_hold:])
            val = idx[np.isin(seeds, list(hold))].tolist()
            train = idx[~np.isin(seeds, list(hold))].tolist()
            if train and val:
                return train, val
        # 只有 1 个 seed -> 退回 stratified
        mode = "stratified"

    if mode == "stratified":
        feas = np.array([bool(s.get("l2_pass", False)) for s in samples])
        train, val = [], []
        for cls in (True, False):
            cls_idx = idx[feas == cls]
            rng.shuffle(cls_idx)
            n_val = int(round(len(cls_idx) * val_ratio))
            if len(cls_idx) >= 2:
                n_val = max(1, min(n_val, len(cls_idx) - 1))
            val.extend(cls_idx[:n_val].tolist())
            train.extend(cls_idx[n_val:].tolist())
        if train and val:
            return sorted(train), sorted(val)
        mode = "random"

    # random
    rng.shuffle(idx)
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1) if n >= 2 else 0
    val = idx[:n_val].tolist()
    train = idx[n_val:].tolist()
    return sorted(train), sorted(val)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load_split_indices(path: str) -> Tuple[List[int], List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    train_idx = [int(i) for i in data["train_indices"]]
    val_idx = [int(i) for i in data["val_indices"]]
    if not train_idx or not val_idx:
        raise ValueError(f"split_indices 无效 (空 train/val): {path}")
    return train_idx, val_idx


def _save_split_indices(path: str, train_idx: List[int], val_idx: List[int],
                        meta: Optional[Dict[str, Any]] = None) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload: Dict[str, Any] = {
        "train_indices": sorted(int(i) for i in train_idx),
        "val_indices": sorted(int(i) for i in val_idx),
    }
    if meta:
        payload.update(meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_training_history_csv(path: str, history: List[Dict]) -> None:
    if not history:
        return
    keys = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _model_hidden_dropout(model_name: str, model_kwargs: Optional[Dict]) -> Tuple[int, float]:
    mk = model_kwargs or {}
    defaults = {
        "mlp": (128, 0.1),
        "deepsets": (128, 0.1),
        "seqrel": (64, 0.2),
        "dynaseqrel_dynedge": (64, 0.2),
        "gcn": (128, 0.1),
        "gat": (128, 0.1),
        "sagpn": (128, 0.1),
    }
    d_hidden, d_drop = defaults.get(model_name, (128, 0.1))
    return int(mk.get("hidden", d_hidden)), float(mk.get("dropout", d_drop))


def resolve_run_dir(run_root: str, model_name: str, split_mode: str,
                    seed: int) -> str:
    """规范实验目录: {run_root}/{model}/{split_mode}/seed{N}/"""
    return os.path.join(run_root, model_name, split_mode, f"seed{seed}")


def resolve_shared_split_path(run_root: str, split_mode: str, seed: int) -> str:
    """跨模型共享的 split 文件路径。"""
    return os.path.join(run_root, "_splits", split_mode, f"seed{seed}",
                        "split_indices.json")


# ------------------------------------------------------------
# 训练
# ------------------------------------------------------------

def _group_grad_norms(model) -> Dict[str, float]:
    """按参数名首段分组统计 grad L2 范数 (用于 --debug-grad)。"""
    groups: Dict[str, float] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        head = name.split(".")[0]
        groups[head] = groups.get(head, 0.0) + float(p.grad.detach().pow(2).sum().item())
    return {k: float(np.sqrt(v)) for k, v in groups.items()}


def _frozen_output_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-5,
) -> bool:
    """Frozen backbone outputs may differ bitwise on CUDA due to nondeterministic
    sparse aggregation, even when parameters are unchanged."""
    return bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))


def _state_hash(state: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _reset_module_parameters(module: torch.nn.Module) -> None:
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def _build_strict_shared_v3(
    flat_dim: int,
    model_kwargs: Dict[str, Any],
    model_init_seed: int,
    score_trunk_init_seed: int,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if model_kwargs.get("head_mode") != "task_specific_score_v3":
        raise ValueError("--strict-shared-init 要求 task_specific_score_v3")
    reference_kwargs = dict(model_kwargs)
    reference_kwargs["head_mode"] = "shared_v1"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_init_seed)
        reference = build_model(
            "dynaseqrel_dynedge", flat_dim=flat_dim, **reference_kwargs)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_init_seed)
        model = build_model(
            "dynaseqrel_dynedge", flat_dim=flat_dim, **model_kwargs)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(score_trunk_init_seed)
        model.score_trunk.apply(_reset_module_parameters)

    reference_state = reference.state_dict()
    model_state = model.state_dict()
    shared_names = []
    extra_names = []
    for name, tensor in model_state.items():
        if name in reference_state:
            if tensor.shape != reference_state[name].shape:
                raise RuntimeError(
                    f"strict shared init shape mismatch: {name}: "
                    f"{tuple(tensor.shape)} != {tuple(reference_state[name].shape)}")
            tensor.copy_(reference_state[name])
            shared_names.append(name)
        else:
            extra_names.append(name)
    missing = sorted(set(reference_state) - set(model_state))
    if missing:
        raise RuntimeError(f"strict-v3 missing v2 parameters: {missing}")
    if any(not name.startswith("score_trunk.") for name in extra_names):
        raise RuntimeError(
            f"strict-v3 unexpected extra parameters: {extra_names}")
    shared_state = {name: model.state_dict()[name] for name in shared_names}
    return model, {
        "strict_shared_init": True,
        "shared_init_reference": "v2",
        "shared_parameter_count": int(sum(
            model_state[name].numel() for name in shared_names)),
        "shared_parameter_tensor_count": len(shared_names),
        "shared_parameter_hash": _state_hash(shared_state),
        "extra_parameter_count": int(sum(
            model_state[name].numel() for name in extra_names)),
        "shared_parameter_names": sorted(shared_names),
        "extra_parameter_names": sorted(extra_names),
        "model_init_seed": int(model_init_seed),
        "score_trunk_init_seed": int(score_trunk_init_seed),
    }


def _validate_v2_base_checkpoint(
    checkpoint_path: str,
    checkpoint: Dict[str, Any],
    feature_version: str,
    model_kwargs: Dict[str, Any],
) -> None:
    expected = {
        "model_name": "dynaseqrel_dynedge",
        "feature_version": "v2",
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"base checkpoint {key} mismatch: "
                f"expected {value!r}, got {checkpoint.get(key)!r}")
    ck_kwargs = checkpoint.get("model_kwargs", {})
    checks = {
        "relation_mode": "staging_dynedge",
        "edge_encoding": "edge_mlp_v2",
        "dynamic_k_spatial": 2,
        "hidden": 64,
        "dropout": 0.2,
    }
    for key, expected_value in checks.items():
        actual = ck_kwargs.get(key, 64 if key == "hidden" else None)
        if actual != expected_value:
            raise ValueError(
                f"base checkpoint model_kwargs.{key} mismatch: "
                f"expected {expected_value!r}, got {actual!r}")
    base_head_mode = ck_kwargs.get("head_mode", "shared_v1")
    if base_head_mode != "shared_v1":
        raise ValueError(
            f"base checkpoint must be v2 shared_v1, got {base_head_mode!r}")
    requested_checks = {
        "relation_mode": "staging_dynedge",
        "edge_encoding": "edge_mlp_v2",
        "dynamic_k_spatial": 2,
        "hidden": 64,
        "dropout": 0.2,
    }
    for key, expected_value in requested_checks.items():
        if model_kwargs.get(key) != expected_value:
            raise ValueError(
                f"adapter architecture {key} must be {expected_value!r}, "
                f"got {model_kwargs.get(key)!r}")
    if feature_version != "v2":
        raise ValueError("adapter-v4 requires --feature-version v2")
    if int(checkpoint.get("epoch", -1)) != 48:
        raise ValueError(
            f"base checkpoint must be best-PR epoch 48, got "
            f"{checkpoint.get('epoch')!r}: {checkpoint_path}")
    if checkpoint.get("select_metric") != "pr_auc":
        raise ValueError(
            "base checkpoint selection metric must be pr_auc, got "
            f"{checkpoint.get('select_metric')!r}")


def _load_frozen_adapter_base(
    model: torch.nn.Module,
    checkpoint_path: str,
    feature_version: str,
    model_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    if not checkpoint_path:
        raise ValueError("adapter-v4 requires --base-checkpoint")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"base checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    _validate_v2_base_checkpoint(
        checkpoint_path, checkpoint, feature_version, model_kwargs)
    base_state = checkpoint["state_dict"]
    result = model.load_state_dict(base_state, strict=False)
    expected_missing = sorted(
        name for name in model.state_dict() if name.startswith("score_adapter."))
    if sorted(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError(
            "adapter base state mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    model.freeze_base_for_score_adapter()
    frozen_names = [
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad]
    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad]
    if not trainable_names or any(
            not name.startswith("score_adapter.") for name in trainable_names):
        raise RuntimeError(
            f"only score_adapter may remain trainable, got {trainable_names}")
    return {
        "training_mode": "frozen_residual_score_adapter",
        "base_checkpoint_path": os.path.abspath(checkpoint_path),
        "base_checkpoint_epoch": int(checkpoint["epoch"]),
        "base_checkpoint_sha256": _file_sha256(checkpoint_path),
        "frozen_parameter_count": int(sum(
            parameter.numel() for parameter in model.parameters()
            if not parameter.requires_grad)),
        "trainable_parameter_count": int(sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad)),
        "adapter_hidden_dim": int(model.adapter_hidden_dim),
        "adapter_dropout": float(model.adapter_dropout),
        "residual_space": "score_logit",
        "zero_initialized_output": True,
        "frozen_modules": list(model.frozen_base_modules()),
        "frozen_parameter_names": frozen_names,
        "trainable_parameter_names": trainable_names,
        "primary_metric": "score_spearman",
    }


def train_model(dataset_path: str,
                model_name: str,
                save_dir: str,
                epochs: int = 100,
                batch_size: int = 64,
                lr: float = 1e-3,
                val_ratio: float = 0.15,
                weight_decay: float = 1e-5,
                seed: int = 0,
                device: Optional[str] = None,
                loss_weights: Optional[LossWeights] = None,
                pos_weight=None,
                model_kwargs: Optional[Dict] = None,
                max_parts: int = F.MAX_PARTS_DEFAULT,
                topk: int = 10,
                split_mode: str = "stratified",
                region_holdout_mode: str = "auto",
                region_xy_grid: int = 3,
                feature_version: str = "v1",
                elite_quantile: Optional[float] = None,
                early_stop_patience: int = 0,
                early_stop_metric: str = "composite",
                min_delta: float = 1e-4,
                save_metric_checkpoints: bool = False,
                debug_grad: bool = False,
                debug_batches: int = 3,
                limit_samples: int = 0,
                shuffle_labels: bool = False,
                split_indices_path: Optional[str] = None,
                shared_split_path: Optional[str] = None,
                run_root: Optional[str] = None,
                code_version: str = "seqrel-v2-repro",
                strict_shared_init: bool = False,
                shared_init_reference: str = "v2",
                model_init_seed: Optional[int] = None,
                score_trunk_init_seed: Optional[int] = None,
                dataloader_seed: Optional[int] = None,
                base_checkpoint_path: Optional[str] = None,
                dataset_manifest_path: Optional[str] = None,
                geometry_holdout_domains: Optional[List[str]] = None,
                generator_pose_mode: str = "predict",
                relseqgen_loss_weights: Optional[RelSeqGenLossWeights] = None,
                generator_elite_quantile: float = 0.70,
                verbose: bool = True) -> Dict:
    os.makedirs(save_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model_init_seed = seed if model_init_seed is None else int(model_init_seed)
    score_trunk_init_seed = (
        seed + 100003 if score_trunk_init_seed is None
        else int(score_trunk_init_seed))
    dataloader_seed = seed if dataloader_seed is None else int(dataloader_seed)

    manifest_split_path = (
        split_indices_path
        or (shared_split_path if shared_split_path
            and os.path.isfile(shared_split_path) else None))
    dataset_manifest = verify_dataset_manifest(
        dataset_path,
        manifest_path=dataset_manifest_path,
        split_path=manifest_split_path,
    )
    samples = load_jsonl(dataset_path)
    if not samples:
        raise RuntimeError(f"数据集为空: {dataset_path}")
    if limit_samples and limit_samples > 0 and limit_samples < len(samples):
        # 分层截断: 尽量保留正/负样本各一部分 (overfit smoke test 用)。
        rng_lim = np.random.RandomState(seed)
        pos = [s for s in samples if s.get("l2_pass", False)]
        neg = [s for s in samples if not s.get("l2_pass", False)]
        rng_lim.shuffle(pos)
        rng_lim.shuffle(neg)
        n_pos_keep = max(1, min(len(pos), limit_samples // 2))
        n_neg_keep = max(1, limit_samples - n_pos_keep)
        samples = pos[:n_pos_keep] + neg[:n_neg_keep]
        rng_lim.shuffle(samples)
        if verbose:
            print(f"[train] limit_samples={limit_samples} -> 实际 {len(samples)} "
                  f"(pos={n_pos_keep} neg={min(len(neg), n_neg_keep)})")
    if shuffle_labels:
        # 标签置换 sanity check: 打乱 l2_pass / layout_score, 模型不应还能拿到正常指标。
        rng_sh = np.random.RandomState(seed + 12345)
        perm = rng_sh.permutation(len(samples))
        feas_shuf = [bool(samples[i].get("l2_pass", False)) for i in perm]
        score_shuf = [float(samples[i].get("layout_score", 0.0)) for i in perm]
        for s, f_new, sc_new in zip(samples, feas_shuf, score_shuf):
            s["l2_pass"] = f_new
            s["layout_score"] = sc_new if f_new else 0.0
        if verbose:
            print("[train] shuffle_labels=True -> 标签已随机置换 (sanity check)")
    ds = LayoutDataset(samples, max_parts=max_parts, feature_version=feature_version)

    split_meta = {
        "split_mode": split_mode,
        "training_seed": seed,
        "val_ratio": val_ratio,
        "dataset_path": os.path.abspath(dataset_path),
        "dataset_sample_count": len(samples),
        "feature_version": feature_version,
    }
    split_source = "computed"
    if split_indices_path:
        if not os.path.isfile(split_indices_path):
            raise FileNotFoundError(
                f"--split-indices 指定文件不存在: {split_indices_path}")
        train_idx, val_idx = _load_split_indices(split_indices_path)
        split_source = split_indices_path
        if verbose:
            print(f"[train] 使用指定 split: {split_indices_path} "
                  f"(train={len(train_idx)} val={len(val_idx)})")
    elif shared_split_path and os.path.isfile(shared_split_path):
        train_idx, val_idx = _load_split_indices(shared_split_path)
        split_source = shared_split_path
        if verbose:
            print(f"[train] 使用共享 split: {shared_split_path} "
                  f"(train={len(train_idx)} val={len(val_idx)})")
    else:
        if split_mode == "geometry_holdout":
            holdout = set(geometry_holdout_domains or [])
            train_idx, val_idx, geo_meta = geometry_holdout_split(
                samples, val_ratio, seed,
                holdout_domains=holdout if holdout else None)
            split_meta.update(geo_meta)
        else:
            train_idx, val_idx = _split_indices(samples, val_ratio, seed, split_mode,
                                                region_holdout_mode=region_holdout_mode,
                                                xy_grid=region_xy_grid)
        if shared_split_path:
            _save_split_indices(shared_split_path, train_idx, val_idx, split_meta)
            split_source = shared_split_path
            if verbose:
                print(f"[train] 已写入共享 split -> {shared_split_path}")
    _save_split_indices(os.path.join(save_dir, "split_indices.json"),
                        train_idx, val_idx,
                        {**split_meta, "source": split_source})
    train_ds, val_ds = Subset(ds, train_idx), Subset(ds, val_idx)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(dataloader_seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_items, drop_last=False,
        generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_items)

    flat_dim = F.flatten_feature_dim(max_parts)
    gen = is_generator(model_name)
    effective_model_kwargs = dict(model_kwargs or {})
    if model_name == "relseqgen":
        effective_model_kwargs.setdefault("pose_mode", generator_pose_mode)
    head_mode = effective_model_kwargs.get("head_mode", "shared_v1")
    is_adapter = (
        model_name == "dynaseqrel_dynedge"
        and head_mode == "frozen_residual_score_adapter_v4")
    if strict_shared_init:
        if model_name != "dynaseqrel_dynedge":
            raise ValueError("--strict-shared-init only supports dynaseqrel_dynedge")
        if shared_init_reference != "v2":
            raise ValueError("--shared-init-reference currently only supports v2")
        if is_adapter:
            raise ValueError("strict shared init and adapter-v4 are mutually exclusive")
        model, initialization_metadata = _build_strict_shared_v3(
            flat_dim, effective_model_kwargs, model_init_seed,
            score_trunk_init_seed)
    else:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(model_init_seed)
            model = build_model(
                model_name, flat_dim=flat_dim, **effective_model_kwargs)
        initialization_metadata = {
            "strict_shared_init": False,
            "shared_init_reference": None,
            "shared_parameter_count": 0,
            "shared_parameter_hash": None,
            "extra_parameter_count": 0,
            "model_init_seed": int(model_init_seed),
            "score_trunk_init_seed": (
                int(score_trunk_init_seed)
                if head_mode == "task_specific_score_v3" else None),
        }
    adapter_metadata: Dict[str, Any] = {}
    if is_adapter:
        adapter_metadata = _load_frozen_adapter_base(
            model, str(base_checkpoint_path or ""), feature_version,
            effective_model_kwargs)
    model = model.to(device)
    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("model has no trainable parameters")
    opt = torch.optim.Adam(
        trainable_parameters, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=max(3, early_stop_patience // 3 or 5),
        min_lr=1e-6)
    # 使用深拷贝，避免 auto pos_weight 修改调用方复用的 LossWeights 对象。
    weights = copy.deepcopy(loss_weights) if loss_weights is not None else LossWeights()
    if is_adapter:
        weights.score_only = True

    # ---- elite 分位阈值: 用训练集 feasible 分数的分位数 ----
    n_feas = sum(1 for s in samples if s.get("l2_pass", False))
    if elite_quantile is not None:
        train_feas_scores = np.array(
            [float(samples[i].get("layout_score", 0.0)) for i in train_idx
             if samples[i].get("l2_pass", False)])
        if len(train_feas_scores) >= 1:
            weights.score_threshold = float(np.percentile(train_feas_scores,
                                                          elite_quantile * 100.0))
        if verbose:
            print(f"[train] elite quantile={elite_quantile:.2f} "
                  f"-> score_threshold={weights.score_threshold:.4f} "
                  f"(train feasible={len(train_feas_scores)})")

    if model_name == "relseqgen":
        rq = relseqgen_loss_weights or RelSeqGenLossWeights()
        rq.elite_quantile = float(
            elite_quantile if elite_quantile is not None else generator_elite_quantile)
        if weights.score_threshold > 0:
            rq.score_threshold = float(weights.score_threshold)
    else:
        rq = None

    # 生成器模型: composite 对生成质量无意义, 默认改用负验证生成损失选 best checkpoint。
    if gen and early_stop_metric == "composite":
        early_stop_metric = "gen_val_neg"
        if verbose:
            print("[train] generator: 自动改用 gen_val_neg "
                  "(负验证生成损失 = -(l_xy+l_station)) 选择 best checkpoint")

    # ---- pos_weight: 仅根据当前训练 split 自动计算，避免验证集泄漏 ----
    n_pos = sum(1 for i in train_idx if samples[i].get("l2_pass", False))
    n_neg = len(train_idx) - n_pos
    suggested_pos_weight = float(n_neg / max(n_pos, 1))

    requested_pos_weight = weights.pos_weight if pos_weight is None else pos_weight
    if isinstance(requested_pos_weight, str):
        spec = requested_pos_weight.strip().lower()
        if spec == "auto":
            effective_pos_weight = suggested_pos_weight if n_pos > 0 else 1.0
        else:
            try:
                effective_pos_weight = float(spec)
            except ValueError as exc:
                raise ValueError(
                    "pos_weight 必须是 'auto' 或正浮点数，例如 2.5"
                ) from exc
    else:
        effective_pos_weight = float(requested_pos_weight)

    if not np.isfinite(effective_pos_weight) or effective_pos_weight <= 0:
        raise ValueError(f"effective pos_weight 必须为正有限值，得到 {effective_pos_weight}")
    weights.pos_weight = float(effective_pos_weight)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[train] model={model_name} generator={gen} params={n_params:,} "
              f"size={model_kwargs}")
        print(f"[train] samples={len(ds)} (feasible={n_feas}) train={len(train_idx)} "
              f"val={len(val_idx)} split={split_mode} feature={feature_version} "
              f"device={device} flat_dim={flat_dim}")
        if feature_version == "v2":
            print("[train] NOTE: 使用 v2 自适应归一化特征, 与 v1 checkpoint 不兼容。")
        mode_text = "auto" if isinstance(requested_pos_weight, str) and requested_pos_weight.strip().lower() == "auto" else "manual"
        print(f"[train] train pos/neg={n_pos}/{n_neg} "
              f"suggested={suggested_pos_weight:.4f} "
              f"requested={requested_pos_weight} "
              f"effective={weights.pos_weight:.4f} mode={mode_text}")
        # 小数据 + 生成式 -> 警告
        if gen and (len(ds) < 3000 or n_feas < 500):
            print("=" * 70)
            print("WARNING:")
            print("Dataset too small for SAGPN / generator.")
            print("Recommended to train MLP / DeepSets / GCN first, or collect more data.")
            print("如果一定要跑, 建议 --model-size small。")
            print("=" * 70)

    best_metric = -np.inf
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    best_path = os.path.join(save_dir, f"{model_name}_best.pt")
    log_path = os.path.join(save_dir, "train.log")
    legacy_log_path = os.path.join(save_dir, f"{model_name}_train.log")
    history: List[Dict] = []
    since_improve = 0
    auxiliary_best = {
        "pr_auc": {"value": -np.inf, "mode": "max", "epoch": 0, "path": os.path.join(
            save_dir, "best_pr_auc.pt")},
        "score_spearman": {"value": -np.inf, "mode": "max", "epoch": 0, "path": os.path.join(
            save_dir, "best_spearman.pt")},
        "score_rmse": {"value": np.inf, "mode": "min", "epoch": 0, "path": os.path.join(
            save_dir, "best_rmse.pt")},
        "composite": {"value": -np.inf, "mode": "max", "epoch": 0, "path": os.path.join(
            save_dir, "best_legacy_composite.pt")},
    }
    t0 = time.time()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_opt_params = sum(p.numel() for g in opt.param_groups for p in g["params"])
    zero_grad_streak = 0
    if debug_grad and verbose:
        print(f"[debug-grad] trainable_params={n_trainable:,} "
              f"optimizer_params={n_opt_params:,} "
              f"(match={'YES' if n_trainable == n_opt_params else 'NO'})")

    hidden_dim, dropout = _model_hidden_dropout(model_name, model_kwargs)
    git_commit = _git_commit()
    n_params = sum(p.numel() for p in model.parameters())
    frozen_state_reference = None
    frozen_output_reference = None
    frozen_check_batch = None
    if is_adapter:
        frozen_state_reference = _state_hash({
            name: tensor for name, tensor in model.state_dict().items()
            if not name.startswith("score_adapter.")
        })
        frozen_check_batch = move_batch(next(iter(val_loader)), device)
        model.eval()
        with torch.no_grad():
            initial_frozen_output = model(frozen_check_batch)
        frozen_output_reference = {
            key: initial_frozen_output[key].detach().clone()
            for key in ("feas_logit", "fail_logits")
        }
    abs_dataset = os.path.abspath(dataset_path)
    abs_save_dir = os.path.abspath(save_dir)
    abs_best_path = os.path.join(abs_save_dir, f"{model_name}_best.pt")

    def _run_config(best_ep: int = 0, best_val: float = float("nan"),
                    best_m: Optional[Dict] = None,
                    wall_s: float = 0.0) -> Dict[str, Any]:
        w = weights
        disable_relation = bool((model_kwargs or {}).get("disable_relation", False))
        disable_sequence = bool((model_kwargs or {}).get("disable_sequence", False))
        if model_name == "dynaseqrel_dynedge":
            relation_variant = str(
                (model_kwargs or {}).get("relation_mode", "staging_dynedge"))
        else:
            relation_variant = (
                "no_relation" if disable_relation
                else "no_sequence" if disable_sequence
                else "full"
            )
        return {
            "model_name": model_name,
            "feature_version": feature_version,
            "split_mode": split_mode,
            "training_seed": seed,
            "dataloader_seed": int(dataloader_seed),
            "model_init_seed": int(model_init_seed),
            "score_trunk_init_seed": (
                int(score_trunk_init_seed)
                if head_mode == "task_specific_score_v3" else None),
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "disable_relation": disable_relation,
            "disable_sequence": disable_sequence,
            "relation_variant": relation_variant,
            "dynamic_edge_dim": (
                11 if model_name == "dynaseqrel_dynedge" else None),
            "dynamic_k_spatial": (
                int((model_kwargs or {}).get("dynamic_k_spatial", 2))
                if model_name == "dynaseqrel_dynedge" else None),
            "dynedge_edge_encoding": (
                str((model_kwargs or {}).get("edge_encoding", "raw_v1"))
                if model_name == "dynaseqrel_dynedge" else None),
            "dynedge_head_mode": (
                str((model_kwargs or {}).get("head_mode", "shared_v1"))
                if model_name == "dynaseqrel_dynedge" else None),
            "score_weight": float(w.alpha),
            "rank_weight": float(w.rank_weight),
            "fail_weight": float(w.fail_weight),
            "use_focal": bool(w.use_focal),
            "focal_alpha": float(w.focal_alpha),
            "focal_gamma": float(w.focal_gamma),
            "learning_rate": float(lr),
            "weight_decay": float(weight_decay),
            "batch_size": int(batch_size),
            "dataset_path": abs_dataset,
            "dataset_sample_count": len(samples),
            "train_count": len(train_idx),
            "val_count": len(val_idx),
            "best_epoch": int(best_ep),
            "best_metric_name": early_stop_metric,
            "best_metric_value": float(best_val),
            "parameter_count": int(n_params),
            "training_time_seconds": float(wall_s),
            "git_commit": git_commit,
            "code_version": code_version,
            "checkpoint_path": abs_best_path,
            "model_kwargs": model_kwargs or {},
            "topk": int(topk),
            "epochs": int(epochs),
            "early_stop_patience": int(early_stop_patience),
            "save_metric_checkpoints": bool(save_metric_checkpoints),
            "effective_pos_weight": float(w.pos_weight),
            "split_indices_source": split_source,
            "run_root": os.path.abspath(run_root) if run_root else None,
            "best_metrics": best_m or {},
            "dataset_manifest_path": (
                os.path.abspath(dataset_manifest_path)
                if dataset_manifest_path else (
                    os.path.abspath(
                        os.path.splitext(dataset_path)[0] + "_manifest.json")
                    if dataset_manifest else None)),
            "dataset_sha256": (
                dataset_manifest.get("dataset_sha256")
                if dataset_manifest else None),
            "split_sha256": (
                dataset_manifest.get("split_sha256")
                if dataset_manifest else None),
            "training_mode": (
                "frozen_residual_score_adapter" if is_adapter else "standard"),
            **initialization_metadata,
            **{
                key: value for key, value in adapter_metadata.items()
                if not key.endswith("_names")
            },
        }

    def _checkpoint_payload(
        epoch: int,
        sel: float,
        metrics: Dict[str, float],
        select_metric: Optional[str] = None,
    ) -> Dict:
        return {
            "model_name": model_name,
            "model_kwargs": model_kwargs or {},
            "state_dict": model.state_dict(),
            "flat_dim": flat_dim,
            "max_parts": max_parts,
            "feature_version": feature_version,
            "is_generator": gen,
            "epoch": epoch,
            "select_metric": select_metric or early_stop_metric,
            "metric": float(sel),
            "metrics": metrics,
            "requested_pos_weight": requested_pos_weight,
            "effective_pos_weight": float(weights.pos_weight),
            "train_positive_count": int(n_pos),
            "train_negative_count": int(n_neg),
            **_run_config(epoch, sel, metrics, time.time() - t0),
            "select_metric": select_metric or early_stop_metric,
            "best_metric_name": select_metric or early_stop_metric,
            "best_metric_value": float(sel),
        }

    with open(log_path, "w", encoding="utf-8") as logf:
        for epoch in range(1, epochs + 1):
            model.train()
            if is_adapter:
                if any(module.training for module in model.frozen_base_modules().values()):
                    raise RuntimeError(
                        "adapter-v4 frozen module entered train mode")
                if not model.score_adapter.training:
                    raise RuntimeError("adapter-v4 adapter did not enter train mode")
            ep_logs: Dict[str, float] = {}
            nb = 0
            for batch in train_loader:
                batch = move_batch(batch, device)
                out = model(batch)
                if model_name == "relseqgen":
                    res = compute_relseqgen_loss(out, batch, weights, rq)
                else:
                    res = compute_loss(out, batch, weights, gen)
                opt.zero_grad()
                res["loss"].backward()
                if is_adapter and any(
                        parameter.grad is not None
                        for name, parameter in model.named_parameters()
                        if not name.startswith("score_adapter.")):
                    raise RuntimeError(
                        "adapter-v4 produced gradients for frozen parameters")

                if debug_grad and nb < debug_batches:
                    with torch.no_grad():
                        logit = out["feas_logit"].detach()
                        sp = out["score_pred"].detach()
                        vpc = float(batch["node_mask"].sum(dim=1).float().mean().item())
                        gnorms = _group_grad_norms(model)
                        cur_lr = opt.param_groups[0]["lr"]
                        logit_std = float(logit.std().item())
                        gline = " ".join(f"{k}={v:.2e}" for k, v in sorted(gnorms.items()))
                        print(f"[debug-grad] ep{epoch} b{nb} "
                              f"node={tuple(batch['node_feat'].shape)} "
                              f"valid_parts~{vpc:.2f} "
                              f"logit(mean={logit.mean():.3f} std={logit_std:.3e} "
                              f"min={logit.min():.3f} max={logit.max():.3f}) "
                              f"score(mean={sp.mean():.3f} std={sp.std():.3e}) "
                              f"lr={cur_lr:.2e}")
                        print(f"[debug-grad] ep{epoch} b{nb} grad_norms: {gline}")
                        if logit_std < 1e-5:
                            print("[debug-grad] WARN: 同 batch logits std < 1e-5 "
                                  "(输出近常数, 疑似 SAGPN 式退化)")
                        main_keys = [k for k in gnorms
                                     if any(t in k for t in
                                            ("encoder", "trunk", "feas", "score",
                                             "phi", "in_proj", "mp", "convs", "layers"))]
                        main_zero = main_keys and all(gnorms[k] < 1e-12 for k in main_keys)
                        if main_zero:
                            zero_grad_streak += 1
                        else:
                            zero_grad_streak = 0
                        if zero_grad_streak >= 5:
                            raise RuntimeError(
                                "[debug-grad] 主要分支 grad norm 连续 5 次为 0, "
                                "训练链路断裂 (梯度未回传)。")

                torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
                opt.step()
                for k, v in res["logs"].items():
                    ep_logs[k] = ep_logs.get(k, 0.0) + float(v)
                nb += 1
            for k in ep_logs:
                ep_logs[k] /= max(nb, 1)

            if is_adapter:
                current_frozen_hash = _state_hash({
                    name: tensor for name, tensor in model.state_dict().items()
                    if not name.startswith("score_adapter.")
                })
                if current_frozen_hash != frozen_state_reference:
                    raise RuntimeError(
                        "adapter-v4 changed frozen base state")
                model.eval()
                with torch.no_grad():
                    current_frozen_output = model(frozen_check_batch)
                for key, expected in frozen_output_reference.items():
                    actual = current_frozen_output[key]
                    if not _frozen_output_close(actual, expected):
                        max_diff = float((actual - expected).abs().max().item())
                        raise RuntimeError(
                            f"adapter-v4 changed frozen {key} "
                            f"(max_abs_diff={max_diff:.6e})")
            metrics = evaluate(model, val_loader, device, gen, topk=topk)
            if gen:
                metrics.update(evaluate_generator_loss(
                    model, val_loader, device, model_name, weights, rq))
            sel = metrics.get(early_stop_metric, metrics["composite"])
            if sel is None or np.isnan(sel):
                sel = metrics["composite"]
            scheduler.step(sel)

            row = {"epoch": epoch, **{f"train_{k}": v for k, v in ep_logs.items()}, **metrics}
            history.append(row)
            line = (f"epoch {epoch:03d} | loss={ep_logs.get('total', 0):.4f} "
                    f"cls={ep_logs.get('l_cls', 0):.4f} score={ep_logs.get('l_score', 0):.4f} "
                    f"rank={ep_logs.get('l_rank', 0):.4f} fail={ep_logs.get('l_fail', 0):.4f} "
                    f"xy={ep_logs.get('l_xy', 0):.4f} st={ep_logs.get('l_station', 0):.4f} "
                    f"| roc={metrics['roc_auc']:.3f} pr={metrics['pr_auc']:.3f} "
                    f"rec@{topk}={metrics['recall_at_k']:.3f} "
                    f"prec@{topk}={metrics['precision_at_k']:.3f} "
                    f"topkAvg={metrics['topk_avg_score_norm']:.3f} "
                    f"sp={metrics['score_spearman']:.3f} "
                    f"pe={metrics['score_pearson']:.3f} "
                    f"kt={metrics['score_kendall']:.3f} "
                    f"rmse={metrics['score_rmse']:.4f} "
                    f"std={metrics['score_pred_std']:.4f} "
                    f"stdR={metrics['score_std_ratio']:.3f} "
                    f"slope={metrics['score_regression_slope']:.3f} "
                    f"enr={metrics['enrichment']:.3f} "
                    f"| comp={metrics['composite']:.4f} [{early_stop_metric}={sel:.4f}]")
            if gen and "gen_val_loss" in metrics:
                line += (f" | valGen={metrics['gen_val_loss']:.4f} "
                         f"(xy={metrics['gen_val_xy']:.4f} st={metrics['gen_val_station']:.4f})")
            logf.write(line + "\n")
            logf.flush()
            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == 1):
                print(line)

            if sel > best_metric + min_delta:
                best_metric = sel
                best_epoch = epoch
                best_metrics = dict(metrics)
                since_improve = 0
                torch.save(_checkpoint_payload(epoch, sel, metrics), best_path)
            else:
                since_improve += 1

            if save_metric_checkpoints:
                for metric_name, state in auxiliary_best.items():
                    value = metrics.get(metric_name)
                    if value is None or np.isnan(value):
                        continue
                    improved = (
                        float(value) < float(state["value"]) - min_delta
                        if state["mode"] == "min"
                        else float(value) > float(state["value"]) + min_delta)
                    if improved:
                        state["value"] = float(value)
                        state["epoch"] = int(epoch)
                        torch.save(
                            _checkpoint_payload(
                                epoch, float(value), metrics,
                                select_metric=metric_name),
                            str(state["path"]),
                        )

            if early_stop_patience > 0 and since_improve >= early_stop_patience:
                if verbose:
                    print(f"[train] early stop at epoch {epoch} "
                          f"(no improve of '{early_stop_metric}' for {early_stop_patience}); "
                          f"best={best_metric:.4f} @ epoch {best_epoch}")
                logf.write(f"[early stop] epoch {epoch}, best={best_metric:.4f} @ {best_epoch}\n")
                break

    wall_time_s = time.time() - t0
    config = _run_config(best_epoch, best_metric, best_metrics, wall_time_s)
    auxiliary_selection = {
        name: {
            "best_value": (
                None if not np.isfinite(float(state["value"]))
                else float(state["value"])),
            "best_epoch": int(state["epoch"]),
            "checkpoint": os.path.abspath(str(state["path"])),
        }
        for name, state in auxiliary_best.items()
    } if save_metric_checkpoints else {}
    config["auxiliary_checkpoint_selection"] = auxiliary_selection
    metrics_out = {
        "best_epoch": best_epoch,
        "best_metric_name": early_stop_metric,
        "best_metric_value": float(best_metric),
        "training_time_seconds": wall_time_s,
        "parameter_count": int(n_params),
        "auxiliary_checkpoint_selection": auxiliary_selection,
        **best_metrics,
    }
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)
    _write_training_history_csv(os.path.join(save_dir, "training_history.csv"), history)
    # 向后兼容: 保留旧命名日志副本
    try:
        with open(log_path, "r", encoding="utf-8") as src, \
                open(legacy_log_path, "w", encoding="utf-8") as dst:
            dst.write(src.read())
    except OSError:
        pass
    # 最终 checkpoint 写入完整元数据
    if os.path.isfile(best_path):
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        ckpt.update(_run_config(best_epoch, best_metric, best_metrics, wall_time_s))
        torch.save(ckpt, best_path)

    summary = {
        "model_name": model_name,
        "select_metric": early_stop_metric,
        "best_metric": float(best_metric),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "best_path": best_path,
        "config_path": os.path.join(save_dir, "config.json"),
        "metrics_path": os.path.join(save_dir, "metrics.json"),
        "split_indices_path": os.path.join(save_dir, "split_indices.json"),
        "split_mode": split_mode,
        "feature_version": feature_version,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "train_feasible_rate": (n_pos / max(len(train_idx), 1)),
        "train_positive_count": int(n_pos),
        "train_negative_count": int(n_neg),
        "requested_pos_weight": requested_pos_weight,
        "effective_pos_weight": float(weights.pos_weight),
        "parameter_count": int(n_params),
        "auxiliary_checkpoint_selection": auxiliary_selection,
        "wall_time_s": wall_time_s,
        "final_metrics": history[-1] if history else {},
        "history": history,
        "config": config,
    }
    with open(os.path.join(save_dir, f"{model_name}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"[train] done. best {early_stop_metric}={best_metric:.4f} "
              f"@ epoch {best_epoch} -> {best_path}")
    return summary
