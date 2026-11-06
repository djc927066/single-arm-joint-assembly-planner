"""layout 数据集诊断脚本 (standalone, 仅依赖 numpy)。

对 ``generate_layout_dataset.py`` 产出的 jsonl 做统计分析, 用于:
    - 判断数据量/正负比是否适合训练;
    - 给出 pos_weight 建议;
    - 给出 "先训 MLP/DeepSets 还是可以上 SAGPN" 的建议。

不加载 torch / wrs, 可在任意 python 环境直接运行。

用法:
    python -m sealp.examples.layout.analyze_layout_dataset \
        --dataset sealp/examples/layout/_output/layout_dataset_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Dict, List

import numpy as np


# ---- SAGPN / MLP 训练门槛 (经验值) ----
SAGPN_MIN_SAMPLES = 3000
SAGPN_MIN_FEASIBLE = 500
SCORER_MIN_SAMPLES = 200
SCORER_MIN_FEASIBLE = 40


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # 崩溃时可能写了半行, 跳过尾部坏行
                    continue
    return out


def _layout_signature(s: Dict) -> tuple:
    """粗粒度 layout 签名 (装配站 + 各零件 staging, 保留 3 位小数) 用于查重。"""
    st = s.get("assembly_station_pos") or []
    st_key = tuple(round(float(v), 3) for v in st[:2])
    parts = []
    for p in s.get("parts", []):
        xy = p.get("staging_xy")
        if xy is None:
            parts.append((str(p.get("part_id")), None))
        else:
            parts.append((str(p.get("part_id")), (round(float(xy[0]), 3), round(float(xy[1]), 3))))
    parts.sort(key=lambda t: t[0])
    return (st_key, tuple(parts))


def _fmt_quantiles(vals: np.ndarray, qs=(0, 10, 25, 50, 70, 75, 90, 100)) -> str:
    if len(vals) == 0:
        return "(空)"
    parts = [f"p{q}={np.percentile(vals, q):.4f}" for q in qs]
    return "  ".join(parts)


def analyze(path: str, topk: int = 10) -> Dict:
    samples = load_jsonl(path)
    n = len(samples)
    if n == 0:
        raise RuntimeError(f"数据集为空或不可读: {path}")

    feas_flags = np.array([bool(s.get("l2_pass", False)) for s in samples])
    n_pos = int(feas_flags.sum())
    n_neg = n - n_pos
    feas_rate = n_pos / n if n else 0.0

    scores_all = np.array([float(s.get("layout_score", 0.0)) for s in samples])
    scores_feas = scores_all[feas_flags]

    pos_weight = (n_neg / max(n_pos, 1)) if n_pos else float("inf")

    # ---- fail_reason / fail_part ----
    fail_reason = Counter(str(s.get("fail_reason", "") or "(none)")
                          for s in samples if not s.get("l2_pass", False))
    fail_part = Counter(str(s.get("fail_part", "") or "(none)")
                        for s in samples if not s.get("l2_pass", False))

    # ---- station_mode ----
    station_mode = Counter(str(s.get("station_mode", "(unknown)")) for s in samples)

    # ---- seed 分布 + 每 seed feasible rate ----
    seeds = [int(s.get("seed", -1)) for s in samples]
    seed_count = Counter(seeds)
    seed_feas: Dict[int, List[int]] = {}
    for s, f in zip(seeds, feas_flags):
        seed_feas.setdefault(s, [0, 0])
        seed_feas[s][0] += int(f)
        seed_feas[s][1] += 1

    # ---- assembly_station x/y 范围 ----
    st = np.array([[float(v) for v in (s.get("assembly_station_pos") or [0, 0, 0])[:2]]
                   for s in samples], dtype=float)
    x_min, x_max = (float(st[:, 0].min()), float(st[:, 0].max())) if len(st) else (0, 0)
    y_min, y_max = (float(st[:, 1].min()), float(st[:, 1].max())) if len(st) else (0, 0)

    # ---- 重复 signature ----
    sigs = [_layout_signature(s) for s in samples]
    sig_count = Counter(sigs)
    n_unique = len(sig_count)
    n_dup = n - n_unique

    # ---- 建议 ----
    can_scorer = (n >= SCORER_MIN_SAMPLES and n_pos >= SCORER_MIN_FEASIBLE)
    can_sagpn = (n >= SAGPN_MIN_SAMPLES and n_pos >= SAGPN_MIN_FEASIBLE)

    result = {
        "path": path,
        "total": n,
        "feasible": n_pos,
        "infeasible": n_neg,
        "feasible_rate": feas_rate,
        "pos_neg_ratio": (n_pos / max(n_neg, 1)),
        "recommended_pos_weight": pos_weight,
        "score_all": {"min": float(scores_all.min()), "mean": float(scores_all.mean()),
                      "max": float(scores_all.max())},
        "score_feasible_quantiles": {
            f"p{q}": float(np.percentile(scores_feas, q)) for q in (0, 25, 50, 70, 75, 90, 100)
        } if len(scores_feas) else {},
        "fail_reason": dict(fail_reason.most_common()),
        "fail_part": dict(fail_part.most_common()),
        "station_mode": dict(station_mode),
        "seed_count": dict(sorted(seed_count.items())),
        "station_x_range": [x_min, x_max],
        "station_y_range": [y_min, y_max],
        "unique_signatures": n_unique,
        "duplicate_signatures": n_dup,
        "recommend_train_scorer": can_scorer,
        "recommend_train_sagpn": can_sagpn,
    }

    # -------- 打印报告 --------
    print("=" * 72)
    print(f"[analyze] dataset = {path}")
    print("=" * 72)
    print(f"total samples          : {n}")
    print(f"feasible (l2_pass)     : {n_pos}")
    print(f"infeasible             : {n_neg}")
    print(f"feasible rate          : {feas_rate*100:.2f}%")
    print(f"positive/negative      : {n_pos} / {n_neg}  (ratio={result['pos_neg_ratio']:.3f})")
    print(f"recommended pos_weight : {pos_weight:.3f}   (= num_neg / num_pos)")
    print("-" * 72)
    print(f"layout_score (all)     : min={scores_all.min():.4f} "
          f"mean={scores_all.mean():.4f} max={scores_all.max():.4f}")
    print(f"layout_score (feasible): {_fmt_quantiles(scores_feas)}")
    if len(scores_feas):
        print(f"  -> elite 阈值(p70)   : {np.percentile(scores_feas, 70):.4f}  "
              f"(>= 该分数的 feasible 参与 proposal 监督)")
    print("-" * 72)
    print("fail_reason 统计:")
    for r, c in fail_reason.most_common(10):
        print(f"    {c:5d}  {r}")
    print("fail_part 统计:")
    for r, c in fail_part.most_common(10):
        print(f"    {c:5d}  {r}")
    print("-" * 72)
    print("station_mode 统计:")
    for m, c in station_mode.items():
        print(f"    {c:5d}  {m}")
    print("-" * 72)
    print("每个 seed 的样本数 / feasible rate:")
    for sd in sorted(seed_feas):
        pos, tot = seed_feas[sd]
        print(f"    seed={sd:<4d} n={tot:<5d} feasible={pos:<5d} rate={pos/max(tot,1)*100:5.1f}%")
    print("-" * 72)
    print(f"assembly_station x range: [{x_min:.4f}, {x_max:.4f}]")
    print(f"assembly_station y range: [{y_min:.4f}, {y_max:.4f}]")
    print(f"unique layout signatures: {n_unique}   duplicates: {n_dup} "
          f"({n_dup/max(n,1)*100:.1f}%)")
    print("=" * 72)
    print("训练建议:")
    if not can_scorer:
        print(f"  [!] 数据偏少 (need >= {SCORER_MIN_SAMPLES} samples & "
              f">= {SCORER_MIN_FEASIBLE} feasible). 可以先小规模训练 MLP 验证管线, "
              f"但结果参考价值有限, 建议继续采集数据。")
    else:
        print(f"  [OK] 适合训练 scorer (MLP / DeepSets / GCN / GAT)。"
              f" 建议优先 MLP 作为 baseline。")
    if can_sagpn:
        print(f"  [OK] 数据量足够, 可以训练 SAGPN base。")
    else:
        print(f"  [!] 数据量不足以训练大 SAGPN "
              f"(建议 >= {SAGPN_MIN_SAMPLES} samples & >= {SAGPN_MIN_FEASIBLE} feasible)。")
        print(f"      -> 现阶段若要试 SAGPN, 请用 --model-size small; "
              f"生成式建议等数据更多再上 base。")
    print(f"  建议 pos_weight = {pos_weight:.2f}, elite score-threshold 用 quantile:0.70。")
    print("=" * 72)
    return result


def main():
    p = argparse.ArgumentParser(description="Layout dataset diagnostic")
    p.add_argument("--dataset", required=True, help="layout_dataset.jsonl 路径")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--json-out", default=None, help="可选: 把统计结果写成 json")
    args = p.parse_args()
    res = analyze(args.dataset, topk=args.topk)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"[analyze] json 已写入 {args.json_out}")


if __name__ == "__main__":
    main()
