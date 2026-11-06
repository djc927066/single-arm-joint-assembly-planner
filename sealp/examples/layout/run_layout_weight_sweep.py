#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_layout_weight_sweep.py —— layout_score 权重搜索 / 权重调参 (论文分析用)。

目的
====
layout_score 由四个**与权重无关**的归一化分项加权而成 (见 evaluate_layout)::

    base  = w_grasp*grasp + w_manip*manip + w_dist*dist + w_rot*rot   # 四权重归一化, 和=1
    score = base * (0.90 + 0.10 * spatial)

本脚本对一组**权重预设 (preset)**, 在**完全相同**的搜索配置 (同 seed / global_explore /
refine_steps / max_evals / cdprim / obstacle_mode) 下各跑一次 global search, 汇总对比,
用于分析"哪种评分偏好更适合当前任务"。

关键原则 (务必牢记):
    * 不修改 evaluate_layout / global search 核心逻辑, 只透传 --w-* 权重;
    * 换权重会改变每个零件选哪个姿态/手臂 (part_score 用同一套权重), 所以是"重新搜索"
      而非"重新打分" —— 因此每个 preset 都真跑一次 global search;
    * 高的 layout_score 不代表真实执行更好, 故**同时输出原始各分项**, 不只输出总分;
    * 最终做方法对比实验时, 所有方法必须统一到同一套权重 (本脚本帮你挑出这套权重)。

用法
====
    python -m sealp.examples.layout.run_layout_weight_sweep \
        --presets balanced,grasp_first,manip_first,distance_first,rotation_first,execution_safe \
        --seeds 0,1,2,3 \
        --global-explore 80 --global-elite 5 \
        --global-refine-steps 0.03,0.015,0.008 --global-max-evals 500 \
        --cdprim-type box \
        --output-dir sealp/examples/layout/_output/weight_sweep \
        --python "D:\\Soft\\tools\\anaconda\\envs\\spatialvla\\python.exe"

输出
====
    weight_sweep_results.csv    每 (preset, seed) 一行 (总分 + 5 个原始分项 + evals/grasp/dist/runtime)
    weight_sweep_summary.json   每 preset 跨 seed 聚合 + 全局 best
    best_layout.layout          最优配置对应的 layout (复制)
    best_layout_debug.json      对应 debug json (复制)
    best_weight.json            最优权重 config
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_THIS_DIR, "_output")
_GLOBAL_MODULE = "sealp.examples.layout.find_optimal_initial_layout_tower_global"

# 权重预设 (w_grasp, w_manip, w_dist, w_rot); 会在搜索脚本内部归一化, 只看相对比例。
PRESETS: Dict[str, Dict[str, float]] = {
    "balanced":       {"grasp": 0.25, "manip": 0.25, "dist": 0.25, "rot": 0.25},
    "grasp_first":    {"grasp": 0.40, "manip": 0.25, "dist": 0.15, "rot": 0.20},
    "manip_first":    {"grasp": 0.25, "manip": 0.40, "dist": 0.15, "rot": 0.20},
    "distance_first": {"grasp": 0.25, "manip": 0.25, "dist": 0.35, "rot": 0.15},
    "rotation_first": {"grasp": 0.25, "manip": 0.25, "dist": 0.15, "rot": 0.35},
    "execution_safe": {"grasp": 0.35, "manip": 0.35, "dist": 0.10, "rot": 0.20},
    # my_current: 使用搜索脚本默认权重 (0.30/0.30/0.15/0.25), 不显式传 --w-*。
    "my_current":     None,  # type: ignore[dict-item]
}


def _parse_args():
    p = argparse.ArgumentParser(description="Layout score weight sweep",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--presets", default="balanced,grasp_first,manip_first,"
                   "distance_first,rotation_first,execution_safe,my_current",
                   help="逗号分隔的预设名 (见 PRESETS)")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--python", default=sys.executable, help="运行子进程的解释器")
    p.add_argument("--common", default="",
                   help="所有 preset 共享的透传 CLI (保证公平), 如 '--planner-obstacle-mode staging_aware'")
    p.add_argument("--cdprim-type", default="box", help="碰撞几何类型 (透传给搜索脚本)")
    p.add_argument("--global-explore", type=int, default=80)
    p.add_argument("--global-elite", type=int, default=5)
    p.add_argument("--global-refine-steps", default="0.03,0.015,0.008")
    p.add_argument("--global-max-evals", type=int, default=500)
    p.add_argument("--n-samples", type=int, default=40)
    p.add_argument("--enable-l3", action="store_true", help="每个 preset 额外做 L3 验证")
    p.add_argument("--l3-top-k", type=int, default=3)
    p.add_argument("--select-by", choices=["weighted_score", "execution_safe"],
                   default="weighted_score",
                   help="best 选择准则: 加权总分 / 执行安全(L3>grasp>manip>dist)")
    p.add_argument("--output-dir", default=os.path.join(_OUTPUT_DIR, "weight_sweep"))
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _build_cmd(args, preset: str, seed: int, out_name: str) -> List[str]:
    common = shlex.split(args.common)
    if args.cdprim_type:
        common += ["--cdprim-type", args.cdprim_type]
    cmd = [args.python, "-m", _GLOBAL_MODULE,
           "--output-name", out_name,
           "--seed", str(seed),
           "--n-samples", str(args.n_samples),
           "--global-explore", str(args.global_explore),
           "--global-elite", str(args.global_elite),
           "--global-refine-steps", args.global_refine_steps,
           "--global-max-evals", str(args.global_max_evals)]
    w = PRESETS.get(preset)
    if w is not None:  # my_current -> 不传, 用脚本默认权重
        cmd += ["--w-grasp", str(w["grasp"]), "--w-manip", str(w["manip"]),
                "--w-dist", str(w["dist"]), "--w-rot", str(w["rot"])]
    if args.enable_l3:
        cmd += ["--enable-l3", "--l3-top-k", str(args.l3_top_k)]
    cmd += common
    return cmd


def _parse_stdout(text: str) -> Dict[str, Optional[float]]:
    def _num(pat, cast=float):
        m = re.search(pat, text)
        return cast(m.group(1)) if m else None
    return {
        "real_evals": _num(r"real evaluations\s*=\s*(\d+)", int),
        "cache_hits": _num(r"eval cache hits\s*=\s*(\d+)", int),
        "feasible_found": _num(r"feasible found\s*=\s*(\d+)", int),
        "wall_time": _num(r"wall-clock total = ([\d.]+)s"),
        "l3_passed": 1 if re.search(r"\[OK\] L3 passed", text) else
                     (0 if re.search(r"L3 (top-k all failed|failed)", text) else None),
    }


def _read_debug(out_name: str) -> Dict:
    path = os.path.join(_OUTPUT_DIR, f"{out_name}_debug.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _run_one(args, preset: str, seed: int) -> Dict:
    out_name = f"wsweep_{preset}_s{seed}"
    cmd = _build_cmd(args, preset, seed, out_name)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{out_name}.log")
    w = PRESETS.get(preset) or {"grasp": 0.30, "manip": 0.30, "dist": 0.15, "rot": 0.25}

    print(f"\n>>> [{preset} seed={seed}] {' '.join(cmd)}")
    if args.dry_run:
        return {"preset": preset, "seed": seed, "cmd": " ".join(cmd)}

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=os.getcwd())
    dt = time.time() - t0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    parsed = _parse_stdout(text)
    dbg = _read_debug(out_name)
    comp = dbg.get("score_components", {}) if dbg else {}
    gc = list((dbg.get("grasp_counts") or {}).values()) if dbg else []
    pd = list((dbg.get("per_part_dist") or {}).values()) if dbg else []

    row = {
        "preset": preset,
        "seed": seed,
        "w_grasp": w["grasp"], "w_manip": w["manip"],
        "w_dist": w["dist"], "w_rot": w["rot"],
        "exit_code": proc.returncode,
        "best_score": dbg.get("score") if dbg else None,
        "feasible": 1 if (dbg and dbg.get("score") is not None) else 0,
        "grasp": comp.get("grasp"),
        "manip": comp.get("manip"),
        "dist": comp.get("dist"),
        "rot": comp.get("rot"),
        "spatial": comp.get("spatial_y_distribution"),
        "l2_feasible_found": parsed.get("feasible_found"),
        "l3_passed": parsed.get("l3_passed"),
        "avg_grasp": float(statistics.mean(gc)) if gc else None,
        "avg_dist": float(statistics.mean(pd)) if pd else None,
        "real_evals": parsed.get("real_evals"),
        "cache_hits": parsed.get("cache_hits"),
        "runtime_s": round(parsed.get("wall_time") or dt, 2),
        "layout_path": (dbg.get("layout_path") if dbg else None)
                        or os.path.join(_OUTPUT_DIR, f"{out_name}.layout"),
        "log": log_path,
    }
    print(f"<<< [{preset} seed={seed}] score={row['best_score']} "
          f"grasp={row['grasp']} manip={row['manip']} dist={row['dist']} rot={row['rot']} "
          f"evals={row['real_evals']} runtime={row['runtime_s']}s")
    return row


def _mean(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def _aggregate(rows: List[Dict], presets: List[str]) -> Dict:
    summary = {}
    for pr in presets:
        rs = [r for r in rows if r["preset"] == pr]
        if not rs:
            continue
        scores = [r["best_score"] for r in rs]
        summary[pr] = {
            "weights": {k: rs[0][f"w_{k}"] for k in ("grasp", "manip", "dist", "rot")},
            "n_runs": len(rs),
            "feasible_runs": sum(r["feasible"] for r in rs),
            "best_score_mean": _mean(scores),
            "best_score_max": max([s for s in scores if s is not None], default=None),
            "grasp_mean": _mean([r["grasp"] for r in rs]),
            "manip_mean": _mean([r["manip"] for r in rs]),
            "dist_mean": _mean([r["dist"] for r in rs]),
            "rot_mean": _mean([r["rot"] for r in rs]),
            "spatial_mean": _mean([r["spatial"] for r in rs]),
            "l3_pass_runs": sum(1 for r in rs if r["l3_passed"]),
            "avg_grasp_mean": _mean([r["avg_grasp"] for r in rs]),
            "avg_dist_mean": _mean([r["avg_dist"] for r in rs]),
            "real_evals_mean": _mean([r["real_evals"] for r in rs]),
            "runtime_s_mean": _mean([r["runtime_s"] for r in rs]),
        }
    return summary


def _pick_best(rows: List[Dict], select_by: str) -> Optional[Dict]:
    feas = [r for r in rows if r["feasible"]]
    if not feas:
        return None
    if select_by == "execution_safe":
        # L3 pass 优先, 其次 avg_grasp, 再 manip, 再更小 dist。
        def key(r):
            return (
                1 if r.get("l3_passed") else 0,
                r.get("avg_grasp") or 0.0,
                r.get("manip") or 0.0,
                -(r.get("avg_dist") or 1e9),
            )
        return max(feas, key=key)
    # weighted_score
    return max(feas, key=lambda r: (r.get("best_score") or -1.0))


def main():
    args = _parse_args()
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    unknown = [p for p in presets if p not in PRESETS]
    if unknown:
        raise SystemExit(f"未知 preset: {unknown}. 可用: {list(PRESETS.keys())}")
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    os.makedirs(args.output_dir, exist_ok=True)

    rows: List[Dict] = []
    for pr in presets:
        for sd in seeds:
            rows.append(_run_one(args, pr, sd))

    if args.dry_run:
        return

    # 明细 csv
    csv_path = os.path.join(args.output_dir, "weight_sweep_results.csv")
    fields = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[wsweep] 明细 -> {csv_path}")

    summary = _aggregate(rows, presets)
    best = _pick_best(rows, args.select_by)

    summary_json = {
        "select_by": args.select_by,
        "config": {
            "seeds": seeds,
            "global_explore": args.global_explore,
            "global_elite": args.global_elite,
            "global_refine_steps": args.global_refine_steps,
            "global_max_evals": args.global_max_evals,
            "enable_l3": bool(args.enable_l3),
            "common": args.common,
        },
        "per_preset": summary,
        "best": None if best is None else {
            "preset": best["preset"], "seed": best["seed"],
            "weights": {k: best[f"w_{k}"] for k in ("grasp", "manip", "dist", "rot")},
            "best_score": best["best_score"],
            "score_components": {k: best.get(k) for k in
                                 ("grasp", "manip", "dist", "rot", "spatial")},
            "l3_passed": best.get("l3_passed"),
            "avg_grasp": best.get("avg_grasp"), "avg_dist": best.get("avg_dist"),
            "layout_path": best.get("layout_path"),
        },
    }
    sj_path = os.path.join(args.output_dir, "weight_sweep_summary.json")
    with open(sj_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    print(f"[wsweep] 汇总 -> {sj_path}")

    # 控制台紧凑表
    print("\n================ Weight sweep summary ================")
    print("preset".ljust(15) + "score".rjust(9) + "grasp".rjust(9) +
          "manip".rjust(9) + "dist".rjust(9) + "rot".rjust(9) + "spatial".rjust(9))
    for pr in presets:
        s = summary.get(pr)
        if not s:
            continue
        def _f(v):
            return "-" if v is None else f"{v:.4f}"
        print(pr.ljust(15) + _f(s["best_score_mean"]).rjust(9) +
              _f(s["grasp_mean"]).rjust(9) + _f(s["manip_mean"]).rjust(9) +
              _f(s["dist_mean"]).rjust(9) + _f(s["rot_mean"]).rjust(9) +
              _f(s["spatial_mean"]).rjust(9))

    # 复制 best layout / debug / weight config
    if best is not None:
        w = {k: best[f"w_{k}"] for k in ("grasp", "manip", "dist", "rot")}
        with open(os.path.join(args.output_dir, "best_weight.json"), "w", encoding="utf-8") as f:
            json.dump({"preset": best["preset"], "seed": best["seed"],
                       "select_by": args.select_by, "weights": w,
                       "best_score": best["best_score"]}, f, ensure_ascii=False, indent=2)
        src_layout = best.get("layout_path")
        if src_layout and os.path.isfile(src_layout):
            shutil.copyfile(src_layout, os.path.join(args.output_dir, "best_layout.layout"))
            dbg_src = src_layout.replace(".layout", "_debug.json")
            if os.path.isfile(dbg_src):
                shutil.copyfile(dbg_src, os.path.join(args.output_dir, "best_layout_debug.json"))
        print(f"\n[wsweep] BEST (select_by={args.select_by}): preset={best['preset']} "
              f"seed={best['seed']} score={best['best_score']} weights={w}")
        print(f"[wsweep] best layout -> {os.path.join(args.output_dir, 'best_layout.layout')}")
    else:
        print("\n[wsweep] WARN: 没有可行结果, 无法选择 best。")

    print("\n[wsweep] 完成。注意: 加权总分高 != 真实执行更好, 请结合原始分项与(可选)L3 综合判断。")


if __name__ == "__main__":
    main()
