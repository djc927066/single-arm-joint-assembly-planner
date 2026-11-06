"""动态 staging 输入诊断: SAGPN (static-only) vs SeqRel (full dynamic).

从数据集中找两个静态属性相同、staging_xy 不同的候选布局, 比较 feasibility logit
与 score_pred 是否随 staging 变化, 并做 staging 梯度/扰动测试。

输出:
    sealp/examples/layout/_output/seqrel_diagnostics/dynamic_input_diagnostic.json

用法:
    python -m sealp.examples.layout.layout_learning._dynamic_input_diagnostic
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_LAYOUT = os.path.dirname(_THIS)
if _LAYOUT not in sys.path:
    sys.path.insert(0, _LAYOUT)

from layout_learning import features as F
from layout_learning.dataset import collate_items, load_jsonl, move_batch, sample_to_item
from layout_learning.models import build_model


def _staging_signature(sample: Dict) -> List[List[float]]:
    parts = F._ordered_parts(sample)
    out = []
    for p in parts:
        xy = p.get("staging_xy") or p.get("staging_pos") or [0.0, 0.0]
        out.append([round(float(xy[0]), 4), round(float(xy[1]), 4)])
    return out


def _static_signature(sample: Dict, feature_version: str) -> Tuple:
    node, _ = F.build_set_feature(sample, feature_version)
    static = node[:, : F._STATIC_DIM].astype(np.float32)
    n = F.sample_num_parts(sample)
    return (
        int(sample.get("seed", -1)),
        tuple(np.round(static[:n].flatten(), 4).tolist()),
    )


def _find_pair(samples: List[Dict], feature_version: str) -> Tuple[int, int]:
    """找同一 seed、静态特征相同、staging 不同的两条样本。"""
    buckets: Dict[Tuple, List[int]] = {}
    for i, s in enumerate(samples):
        key = _static_signature(s, feature_version)
        buckets.setdefault(key, []).append(i)
    for indices in buckets.values():
        if len(indices) < 2:
            continue
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                ia, ib = indices[a], indices[b]
                if _staging_signature(samples[ia]) != _staging_signature(samples[ib]):
                    return ia, ib
    raise RuntimeError("未找到 static 相同但 staging 不同的样本对")


def _sample_id(sample: Dict) -> int:
    sid = sample.get("sample_id") or sample.get("id")
    if sid is not None:
        return int(sid)
    return int(abs(hash(json.dumps(_staging_signature(sample), sort_keys=True))) % (10 ** 18))


def _forward_logits(model, batch: Dict) -> Tuple[float, float]:
    out = model(batch)
    logit = float(out["feas_logit"].detach().cpu().numpy().ravel()[0])
    score = float(out["score_pred"].detach().cpu().numpy().ravel()[0])
    return logit, score


def _staging_grad_test(model, batch: Dict, device: str) -> Dict:
    """对 node_feat 的 staging 维 (18:20) 求梯度。"""
    model.zero_grad(set_to_none=True)
    nf = batch["node_feat"].clone().detach().requires_grad_(True)
    b2 = dict(batch)
    b2["node_feat"] = nf
    out = model(b2)
    loss = out["feas_logit"].sum() + out["score_pred"].sum()
    loss.backward()
    grad = nf.grad
    if grad is None:
        return {"max_abs_grad_on_staging_dims": 0.0,
                "staging_in_computation_graph": False}
    # staging 在 static 之后的 dynamic 段; v1/v2 均为 dim 18:20
    staging_grad = grad[..., F._STATIC_DIM: F._STATIC_DIM + 2]
    max_g = float(staging_grad.abs().max().item())
    return {
        "max_abs_grad_on_staging_dims": max_g,
        "staging_in_computation_graph": max_g > 0.0,
    }


def _perturb_staging(sample: Dict, part_idx: int = 1,
                     delta: Tuple[float, float] = (0.15, -0.10)) -> Dict:
    import copy
    s2 = copy.deepcopy(sample)
    parts = F._ordered_parts(s2)
    if part_idx >= len(parts):
        part_idx = 0
    key = "staging_xy" if "staging_xy" in parts[part_idx] else "staging_pos"
    xy = list(parts[part_idx].get(key, [0.0, 0.0]))
    before = [float(xy[0]), float(xy[1])]
    after = [before[0] + delta[0], before[1] + delta[1]]
    parts[part_idx][key] = after
    return s2, before, after, part_idx


def _run_model_diag(model_name: str, ckpt_path: str, samples: List[Dict],
                    pair: Tuple[int, int], feature_version: str,
                    device: str) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(model_name, flat_dim=ckpt["flat_dim"],
                        **ckpt.get("model_kwargs", {})).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ia, ib = pair
    max_parts = ckpt.get("max_parts", F.MAX_PARTS_DEFAULT)
    items = [
        sample_to_item(samples[ia], max_parts, feature_version),
        sample_to_item(samples[ib], max_parts, feature_version),
    ]
    batch = move_batch(collate_items(items), device)

    with torch.no_grad():
        out = model(batch)
        logits = out["feas_logit"].detach().cpu().numpy().ravel().tolist()
        scores = out["score_pred"].detach().cpu().numpy().ravel().tolist()

    grad_info = _staging_grad_test(
        model, move_batch(collate_items([items[0]]), device), device)
    s_pert, before, after, pidx = _perturb_staging(samples[ia])
    batch_a = move_batch(collate_items([items[0]]), device)
    batch_p = move_batch(collate_items([
        sample_to_item(s_pert, max_parts, feature_version)]), device)
    with torch.no_grad():
        la0, sa0 = _forward_logits(model, batch_a)
        lp, sp = _forward_logits(model, batch_p)

    logit_diff_p = abs(lp - la0)
    score_diff_p = abs(sp - sa0)

    is_sagpn = model_name == "sagpn"
    return {
        "checkpoint": os.path.abspath(ckpt_path),
        "feas_logit_A": float(logits[0]),
        "feas_logit_B": float(logits[1]),
        "score_pred_A": float(scores[0]),
        "score_pred_B": float(scores[1]),
        "logit_difference": float(abs(logits[0] - logits[1])),
        "score_difference": float(abs(scores[0] - scores[1])),
        "outputs_identical_or_near": float(abs(logits[0] - logits[1])) < 1e-6
        and float(abs(scores[0] - scores[1])) < 1e-6,
        "staging_perturbation_test": {
            "perturbed_part_index": int(pidx),
            "staging_before": before,
            "staging_after": after,
            "logit_diff": float(logit_diff_p),
            "score_diff": float(score_diff_p),
            "output_varies": logit_diff_p > 1e-6 or score_diff_p > 1e-6,
        },
        "staging_gradient_test": grad_info,
        "uses_dynamic_staging_in_scorer": not is_sagpn,
        "note": (
            "SAGPN scorer 头使用 static_only(node); staging_xy 在 DYNAMIC 维被置零"
            if is_sagpn else
            "SeqRel 完整消费 DYNAMIC staging 特征并用于 pairwise 几何"
        ),
    }


def main():
    root = os.path.abspath(os.path.join(_LAYOUT, "..", "..", ".."))
    default_ds = os.path.join(root, "sealp", "examples", "layout", "_output",
                              "layout_dataset_v2.jsonl")
    default_out = os.path.join(root, "sealp", "examples", "layout", "_output",
                               "seqrel_diagnostics", "dynamic_input_diagnostic.json")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=default_ds)
    p.add_argument("--feature-version", default="v2")
    p.add_argument("--sagpn-checkpoint",
                   default=os.path.join(root, "checkpoints",
                                        "layout_models_transfer_v3", "sagpn_best.pt"))
    p.add_argument("--seqrel-checkpoint", default=None,
                   help="SeqRel checkpoint; 默认 repro stratified seed0")
    p.add_argument("--output", default=default_out)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    seqrel_ckpt = args.seqrel_checkpoint or os.path.join(
        root, "checkpoints", "layout_models_repro", "seqrel", "stratified",
        "seed0", "seqrel_best.pt")

    samples = load_jsonl(args.dataset)
    ia, ib = _find_pair(samples, args.feature_version)

    if not os.path.isfile(args.sagpn_checkpoint):
        raise FileNotFoundError(f"SAGPN checkpoint 不存在: {args.sagpn_checkpoint}")
    if not os.path.isfile(seqrel_ckpt):
        raise FileNotFoundError(f"SeqRel checkpoint 不存在: {seqrel_ckpt}")

    report = {
        "dataset": os.path.abspath(args.dataset),
        "feature_version": args.feature_version,
        "sample_indices": [ia, ib],
        "sample_ids": [_sample_id(samples[ia]), _sample_id(samples[ib])],
        "staging_signatures": [
            _staging_signature(samples[ia]),
            _staging_signature(samples[ib]),
        ],
        "static_signature_match": True,
        "sagpn": _run_model_diag("sagpn", args.sagpn_checkpoint, samples,
                                 (ia, ib), args.feature_version, device),
        "seqrel": _run_model_diag("seqrel", seqrel_ckpt, samples,
                                  (ia, ib), args.feature_version, device),
    }
    report["conclusion"] = (
        "SAGPN scorer 对 staging 不敏感 (static-only); "
        "SeqRel 对 staging / pairwise 几何敏感。"
        if report["sagpn"]["logit_difference"] < 1e-5
        and report["seqrel"]["logit_difference"] > 1e-5
        else "见各模型 logit/score 差值与梯度测试。"
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[diag] wrote {args.output}")
    print(f"[diag] SAGPN logit_diff={report['sagpn']['logit_difference']:.6f} "
          f"SeqRel logit_diff={report['seqrel']['logit_difference']:.6f}")


if __name__ == "__main__":
    main()
