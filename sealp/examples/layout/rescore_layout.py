#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""rescore_layout.py —— 对**同一个已有布局**用任意权重重新打分。

用途
====
搜索脚本 (global / neural / nsga2 ...) 会把结果写成
``{name}.layout`` (YAML) 和 ``{name}_debug.json``。两者的 ``score_components``
里都存了**与权重无关**的已归一化聚合分项::

    grasp / manip / dist / rot / spatial_y_distribution   ∈ [0, 1]

而最终 ``layout_score`` 的合成方式 (见 evaluate_layout)::

    base  = w_grasp*grasp + w_manip*manip + w_dist*dist + w_rot*rot   # 四权重归一化, 和=1
    score = base * (0.90 + 0.10 * spatial)                            # spatial 固定占 10%

因此对**固定布局**(位姿/手臂/坐标都不变)换权重重新打分, 就是纯算术, 不需要
重跑 evaluate_layout, 也**不依赖 wrs / torch / panda3d**。本脚本正是这样做的,
结果与 evaluate_layout 的口径逐位一致。

注意
====
* 这只是"换评分口径", **不代表布局变好**。要用来做论文对比, 必须让所有方法
  用同一套权重 (否则不公平)。
* 若你想在**全新搜索**中提分, 请把同样的 ``--w-*`` 传给搜索脚本; 那时权重还会
  影响每个零件选哪个姿态/手臂, 属于"重新搜索"而非"重新打分"。

示例
====
    # 用原文件自带权重复算 (应当与文件里的 score 完全一致, 用来自检)
    python -m sealp.examples.layout.rescore_layout _output/tower_neural_sagpn.layout

    # 换一套权重看同一布局分数怎么变
    python -m sealp.examples.layout.rescore_layout _output/tower_neural_sagpn.layout \
        --w-grasp 1 --w-manip 1 --w-dist 4 --w-rot 4

    # 批量 (通配符) + 导出 csv + 给出"该布局能拿到的最高分对应的权重"
    python -m sealp.examples.layout.rescore_layout "_output/*.layout" --suggest --csv rescore.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional, Tuple

# 与 find_optimal_initial_layout_tower_strict_pycharm.py 中的 DEFAULT_W_* 保持一致
DEFAULT_WEIGHTS: Dict[str, float] = {
    "grasp": 0.30,
    "manip": 0.30,
    "dist": 0.15,
    "rot": 0.25,
}

# spatial 修正: score = base * (SPATIAL_BASE + SPATIAL_GAIN * spatial)
SPATIAL_BASE = 0.90
SPATIAL_GAIN = 0.10

_COMPONENT_KEYS = ("grasp", "manip", "dist", "rot")


# ----------------------------------------------------------------------------
# 读取 layout 文件 (.layout / _debug.json)
# ----------------------------------------------------------------------------
def _load_yaml(path: str) -> dict:
    try:
        import yaml  # PyYAML, .layout 本身就是 yaml
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "读取 .layout 需要 PyYAML: pip install pyyaml\n"
            "(或者直接对 *_debug.json 运行本脚本, 那是纯 json 不需要 yaml)"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_layout_record(path: str) -> Dict:
    """从 .layout 或 _debug.json 里提取打分所需字段。

    返回 dict: {name, score_components{grasp,manip,dist,rot,spatial},
                orig_weights{...}, orig_score}
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        comps = doc.get("score_components", {})
        # debug.json 不存 weights, 只能回退到默认权重
        weights = doc.get("weights")
        orig_score = doc.get("score")
        name = os.path.splitext(os.path.basename(path))[0]
    else:
        doc = _load_yaml(path)
        meta = doc.get("metadata", {}) if isinstance(doc, dict) else {}
        comps = meta.get("score_components", {})
        weights = meta.get("weights")
        orig_score = meta.get("score")
        name = doc.get("name") or os.path.splitext(os.path.basename(path))[0]

    if not comps:
        raise ValueError(
            f"{path} 里找不到 score_components, 无法重新打分 "
            f"(该文件可能不是搜索脚本产出的 layout)"
        )

    spatial = comps.get("spatial_y_distribution", comps.get("spatial", 0.0))
    components = {
        "grasp": float(comps.get("grasp", 0.0)),
        "manip": float(comps.get("manip", 0.0)),
        "dist": float(comps.get("dist", 0.0)),
        "rot": float(comps.get("rot", 0.0)),
        "spatial": float(spatial),
    }
    orig_weights = None
    if weights:
        orig_weights = {k: float(weights.get(k, 0.0)) for k in _COMPONENT_KEYS}

    return {
        "path": path,
        "name": name,
        "components": components,
        "orig_weights": orig_weights,
        "orig_score": None if orig_score is None else float(orig_score),
    }


# ----------------------------------------------------------------------------
# 打分 (与 evaluate_layout 完全一致的口径)
# ----------------------------------------------------------------------------
def normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(0.0, float(w.get(k, 0.0))) for k in _COMPONENT_KEYS)
    if s <= 1e-9:  # 与源码一致: 退化到默认权重
        w = dict(DEFAULT_WEIGHTS)
        s = sum(w[k] for k in _COMPONENT_KEYS)
    return {k: max(0.0, float(w.get(k, 0.0))) / s for k in _COMPONENT_KEYS}


def score_layout(components: Dict[str, float], weights: Dict[str, float]) -> Tuple[float, float, Dict[str, float]]:
    """返回 (layout_score, base_score, 各分项对 base 的贡献)。"""
    wn = normalize_weights(weights)
    contrib = {k: wn[k] * components[k] for k in _COMPONENT_KEYS}
    base = sum(contrib.values())
    spatial_factor = SPATIAL_BASE + SPATIAL_GAIN * components["spatial"]
    return base * spatial_factor, base, contrib


def best_weights(components: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    """对固定布局, base 是各分项的凸组合 -> 全部权重压到分值最高的那一项时 base 最大。"""
    best_k = max(_COMPONENT_KEYS, key=lambda k: components[k])
    w = {k: (1.0 if k == best_k else 0.0) for k in _COMPONENT_KEYS}
    score, _, _ = score_layout(components, w)
    return w, score


# ----------------------------------------------------------------------------
# 展示
# ----------------------------------------------------------------------------
def _fmt_w(w: Dict[str, float]) -> str:
    return "  ".join(f"{k}={w[k]:.3f}" for k in _COMPONENT_KEYS)


def print_report(rec: Dict, new_weights: Dict[str, float], suggest: bool) -> Dict:
    comps = rec["components"]
    new_score, new_base, contrib = score_layout(comps, new_weights)
    nw = normalize_weights(new_weights)

    print("=" * 74)
    print(f"[{rec['name']}]  {rec['path']}")
    print("-" * 74)
    print("分项 (与权重无关, 归一化到 [0,1]):")
    print(f"    grasp={comps['grasp']:.4f}  manip={comps['manip']:.4f}  "
          f"dist={comps['dist']:.4f}  rot={comps['rot']:.4f}  "
          f"spatial={comps['spatial']:.4f}")
    print(f"    spatial 修正因子 = {SPATIAL_BASE} + {SPATIAL_GAIN}*spatial = "
          f"{SPATIAL_BASE + SPATIAL_GAIN * comps['spatial']:.4f}")

    if rec["orig_weights"] is not None:
        ow = normalize_weights(rec["orig_weights"])
        o_score, _, _ = score_layout(comps, rec["orig_weights"])
        print("\n原始权重:")
        print(f"    {_fmt_w(ow)}")
        note = ""
        if rec["orig_score"] is not None:
            note = f"   (文件记录 score={rec['orig_score']:.6f}, 复算={o_score:.6f})"
        print(f"    -> layout_score = {o_score:.6f}{note}")

    print("\n新权重:")
    print(f"    {_fmt_w(nw)}")
    print("    各分项对 base 的贡献: " +
          "  ".join(f"{k}={contrib[k]:.4f}" for k in _COMPONENT_KEYS))
    print(f"    base = {new_base:.6f}")
    print(f"    -> layout_score = {new_score:.6f}")

    if rec["orig_score"] is not None:
        delta = new_score - rec["orig_score"]
        arrow = "↑" if delta > 1e-9 else ("↓" if delta < -1e-9 else "=")
        print(f"    相对文件原分变化: {arrow} {delta:+.6f}")

    if suggest:
        bw, bs = best_weights(comps)
        print("\n该布局的分数上限 (把权重全压到最高分项):")
        print(f"    最优权重 {_fmt_w(bw)}")
        print(f"    -> 最高 layout_score = {bs:.6f}")

    return {
        "name": rec["name"],
        "path": rec["path"],
        **{f"comp_{k}": comps[k] for k in _COMPONENT_KEYS},
        "comp_spatial": comps["spatial"],
        **{f"w_{k}": nw[k] for k in _COMPONENT_KEYS},
        "orig_score": rec["orig_score"],
        "new_score": new_score,
    }


# ----------------------------------------------------------------------------
def expand_inputs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for p in patterns:
        matched = glob.glob(p)
        if matched:
            files.extend(sorted(matched))
        elif os.path.exists(p):
            files.append(p)
        else:
            print(f"[warn] 找不到文件/未匹配: {p}")
    # 去重, 保序
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="对已有布局用任意权重重新打分 (纯算术, 不重跑 evaluate_layout)。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+",
                    help=".layout 或 _debug.json 路径, 支持通配符 (记得加引号)")
    ap.add_argument("--w-grasp", type=float, default=None, help="grasp 权重 (默认沿用文件权重)")
    ap.add_argument("--w-manip", type=float, default=None, help="manip 权重")
    ap.add_argument("--w-dist", type=float, default=None, help="dist 权重")
    ap.add_argument("--w-rot", type=float, default=None, help="rot 权重")
    ap.add_argument("--suggest", action="store_true",
                    help="额外给出该布局能拿到的最高分及对应权重")
    ap.add_argument("--csv", type=str, default=None, help="把结果写入 csv")
    return ap.parse_args()


def resolve_weights(args: argparse.Namespace, rec: Dict) -> Dict[str, float]:
    cli = {"grasp": args.w_grasp, "manip": args.w_manip,
           "dist": args.w_dist, "rot": args.w_rot}
    # 命令行没给任何权重 -> 沿用文件权重; 文件也没有 -> 默认权重
    if all(v is None for v in cli.values()):
        return dict(rec["orig_weights"] or DEFAULT_WEIGHTS)
    base = dict(rec["orig_weights"] or DEFAULT_WEIGHTS)
    for k, v in cli.items():
        if v is not None:
            base[k] = float(v)
    return base


def main() -> None:
    args = parse_args()
    files = expand_inputs(args.inputs)
    if not files:
        raise SystemExit("没有可处理的文件。")

    rows: List[Dict] = []
    for path in files:
        try:
            rec = load_layout_record(path)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {path}: {e}")
            continue
        weights = resolve_weights(args, rec)
        rows.append(print_report(rec, weights, args.suggest))

    if args.csv and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("\n" + "=" * 74)
        print(f"[OK] 已写出 csv -> {args.csv}  ({len(rows)} 行)")


if __name__ == "__main__":
    main()
