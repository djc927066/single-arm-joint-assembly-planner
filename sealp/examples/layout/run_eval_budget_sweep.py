#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""run_eval_budget_sweep.py —— 评估预算扫描 + score-eval 曲线 (论文用)。

动机
====
论文里"神经网络加速布局搜索"的正确卖点不是**无限预算下的最高分**, 而是
**在相同(且昂贵的) evaluate_layout 预算下, 谁的分数更高 / 谁用更少评估达到同一分数**。

本脚本对多个方法 (默认 global vs sagpn) 在一串 ``--budgets`` (即 --global-max-evals)
上分别跑多 seed, 汇总:

    best layout_score  vs  评估预算 (max_evals)
    best layout_score  vs  实际 evaluate_layout 次数 (real_evals)

并画成折线图 (每方法一条线, 阴影=seed 间 std)。这张图直接体现"同预算更高分"
或"更少评估达标", 是"NN 加速"的核心证据。

每个 (method, budget, seed) 作为子进程调用现有搜索脚本, 与真实评估口径完全一致;
复用 run_layout_experiments 的命令构造与解析逻辑, 避免口径漂移。

用法
====
    conda activate spatialvla

    python -m sealp.examples.layout.run_eval_budget_sweep \
        --methods global,sagpn \
        --budgets 50,100,200,400,800 \
        --seeds 0,1,2 \
        --checkpoint-dir checkpoints/layout_models \
        --station-mode grid3x3 \
        --results-dir sealp/examples/layout/_output/sweep \
        --python "D:\\Soft\\tools\\anaconda\\envs\\spatialvla\\python.exe"

说明
====
* --station-mode 只透传给神经方法 (global/random/nsga2 不认识该参数)。
* --target-score S: 额外统计每方法"达到分数 S 所需的最小预算/评估" (evals-to-target)。
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
import time
import types
from typing import Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# 复用批量实验脚本里的命令构造 / 解析 / 读取 debug 逻辑, 保证口径一致。
import run_layout_experiments as rle  # type: ignore


def _parse_args():
    p = argparse.ArgumentParser(description="Eval-budget sweep + score-eval curve")
    p.add_argument("--methods", default="global,sagpn",
                   help="逗号分隔; 支持 random,nsga2,global 与 NN 方法 (mlp/gcn/gat/sagpn...)")
    p.add_argument("--budgets", default="50,100,200,400,800",
                   help="逗号分隔的 max_evals 预算列表")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--checkpoint-dir", default=os.path.join("checkpoints", "layout_models"))
    p.add_argument("--results-dir", default=os.path.join(rle._OUTPUT_DIR, "sweep"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--common", default="--cdprim-type box",
                   help="所有方法共享的透传 CLI 参数")
    p.add_argument("--station-mode", default=None, choices=[None, "continuous", "grid3x3"],
                   help="只透传给神经方法的装配站模式")
    p.add_argument("--top-k-proposals", type=int, default=64)
    p.add_argument("--scorer-pool", type=int, default=400)
    p.add_argument("--global-elite", type=int, default=3)
    p.add_argument("--global-refine-steps", default="0.03,0.015,0.008")
    p.add_argument("--n-samples", type=int, default=40)
    p.add_argument("--target-score", type=float, default=None,
                   help="统计达到该分数所需的最小预算/评估")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _mk_rle_args(args, budget: int) -> types.SimpleNamespace:
    """构造一个能喂给 rle._build_cmd 的 args 对象 (固定该预算)。"""
    return types.SimpleNamespace(
        python=args.python,
        common=args.common,
        max_evals=int(budget),
        top_k_proposals=args.top_k_proposals,
        scorer_pool=args.scorer_pool,
        global_elite=args.global_elite,
        global_refine_steps=args.global_refine_steps,
        n_samples=args.n_samples,
        enable_l3=False,
        checkpoint_dir=args.checkpoint_dir,
    )


def _run_one(args, method: str, budget: int, seed: int) -> Dict:
    out_name = f"sweep_{method}_b{budget}_s{seed}"
    rle_args = _mk_rle_args(args, budget)
    cmd = rle._build_cmd(rle_args, method, seed, out_name)

    # station-mode 只对神经方法追加 (其它脚本不识别该 flag)。
    if args.station_mode and method in rle.NN_METHODS:
        cmd += ["--station-mode", args.station_mode]

    log_dir = os.path.join(args.results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{out_name}.log")

    print(f"\n>>> [{method} budget={budget} seed={seed}] {' '.join(cmd)}")
    if args.dry_run:
        return {"method": method, "budget": budget, "seed": seed, "cmd": " ".join(cmd)}

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=os.getcwd())
    dt = time.time() - t0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    parsed = rle._parse_stdout(text)
    dbg = rle._read_debug(out_name)
    md = rle._metrics_from_debug(dbg)

    row = {
        "method": method,
        "budget": budget,
        "seed": seed,
        "exit_code": proc.returncode,
        "runtime_s": round(parsed.get("wall_time") or dt, 2),
        "best_score": md["best_score"],
        "feasible": 1 if md["best_score"] is not None else 0,
        "real_evals": parsed.get("real_evals"),
        "feasible_found": parsed.get("feasible_found"),
        "cache_hits": parsed.get("cache_hits"),
        "score_grasp": md["grasp"],
        "score_manip": md["manip"],
        "score_dist": md["dist"],
        "score_rot": md["rot"],
        "score_spatial": md["spatial"],
        "log": log_path,
    }
    print(f"<<< [{method} budget={budget} seed={seed}] "
          f"score={row['best_score']} real_evals={row['real_evals']} "
          f"runtime={row['runtime_s']}s")
    return row


def _mean(vals) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def _std(vals) -> float:
    vals = [v for v in vals if v is not None]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def _aggregate(rows: List[Dict], methods: List[str], budgets: List[int]) -> Dict:
    """agg[method][budget] = {score_mean, score_std, evals_mean, runtime_mean, n}"""
    agg: Dict[str, Dict[int, Dict]] = {m: {} for m in methods}
    for m in methods:
        for b in budgets:
            sub = [r for r in rows if r["method"] == m and r["budget"] == b]
            scores = [r["best_score"] for r in sub]
            agg[m][b] = {
                "score_mean": _mean(scores),
                "score_std": _std(scores),
                "score_max": max([s for s in scores if s is not None], default=None),
                "evals_mean": _mean([r["real_evals"] for r in sub]),
                "runtime_mean": _mean([r["runtime_s"] for r in sub]),
                "success": sum(r["feasible"] for r in sub),
                "n": len(sub),
            }
    return agg


def _evals_to_target(rows: List[Dict], methods: List[str], budgets: List[int],
                     target: float) -> Dict[str, Optional[int]]:
    """每方法达到 target_score 所需的最小预算 (按 budget 升序找第一个 mean>=target)。"""
    out: Dict[str, Optional[int]] = {}
    for m in methods:
        hit = None
        for b in sorted(budgets):
            sub = [r["best_score"] for r in rows
                   if r["method"] == m and r["budget"] == b and r["best_score"] is not None]
            if sub and statistics.mean(sub) >= target:
                hit = b
                break
        out[m] = hit
    return out


def _plot(agg: Dict, methods: List[str], budgets: List[int], results_dir: str,
          target: Optional[float]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[sweep] matplotlib 不可用, 跳过绘图。")
        return

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
              "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]
    budgets_sorted = sorted(budgets)

    # ---- 图1: score vs budget (max_evals) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(methods):
        xs, ys, es = [], [], []
        for b in budgets_sorted:
            cell = agg[m][b]
            if cell["score_mean"] is None:
                continue
            xs.append(b)
            ys.append(cell["score_mean"])
            es.append(cell["score_std"])
        if not xs:
            continue
        c = colors[i % len(colors)]
        ax.plot(xs, ys, "-o", color=c, label=m, linewidth=2)
        lo = [y - e for y, e in zip(ys, es)]
        hi = [y + e for y, e in zip(ys, es)]
        ax.fill_between(xs, lo, hi, color=c, alpha=0.15)
    if target is not None:
        ax.axhline(target, ls="--", color="gray", lw=1, label=f"target={target}")
    ax.set_xlabel("evaluation budget (--global-max-evals)")
    ax.set_ylabel("best layout_score (mean +/- std)")
    ax.set_title("Score vs. evaluation budget")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out1 = os.path.join(results_dir, "score_vs_budget.png")
    plt.savefig(out1, dpi=140)
    plt.close(fig)
    print(f"[sweep] 绘图 -> {out1}")

    # ---- 图2: score vs 实际 evaluate_layout 次数 ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(methods):
        pts = []
        for b in budgets_sorted:
            cell = agg[m][b]
            if cell["score_mean"] is None or cell["evals_mean"] is None:
                continue
            pts.append((cell["evals_mean"], cell["score_mean"]))
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        c = colors[i % len(colors)]
        ax.plot(xs, ys, "-o", color=c, label=m, linewidth=2)
    if target is not None:
        ax.axhline(target, ls="--", color="gray", lw=1, label=f"target={target}")
    ax.set_xlabel("actual #evaluate_layout (real_evals, mean)")
    ax.set_ylabel("best layout_score (mean)")
    ax.set_title("Score vs. actual evaluations")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    out2 = os.path.join(results_dir, "score_vs_realevals.png")
    plt.savefig(out2, dpi=140)
    plt.close(fig)
    print(f"[sweep] 绘图 -> {out2}")


def main():
    args = _parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    budgets = [int(b) for b in args.budgets.replace(",", " ").split()]
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    os.makedirs(args.results_dir, exist_ok=True)

    rows: List[Dict] = []
    for method in methods:
        for budget in budgets:
            for seed in seeds:
                rows.append(_run_one(args, method, budget, seed))

    if args.dry_run:
        return

    csv_path = os.path.join(args.results_dir, "sweep_results.csv")
    fields = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[sweep] 明细 -> {csv_path}")

    agg = _aggregate(rows, methods, budgets)

    # 汇总表 (每 method x budget 一行)
    agg_csv = os.path.join(args.results_dir, "sweep_summary.csv")
    with open(agg_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["method", "budget", "score_mean", "score_std", "score_max",
                    "real_evals_mean", "runtime_s_mean", "success", "n"])
        for m in methods:
            for b in sorted(budgets):
                c = agg[m][b]
                w.writerow([m, b,
                            None if c["score_mean"] is None else round(c["score_mean"], 4),
                            round(c["score_std"], 4),
                            c["score_max"],
                            None if c["evals_mean"] is None else round(c["evals_mean"], 1),
                            None if c["runtime_mean"] is None else round(c["runtime_mean"], 1),
                            c["success"], c["n"]])
    print(f"[sweep] 汇总 -> {agg_csv}")

    # 控制台打印一张紧凑表
    print("\n================ Sweep summary (score_mean) ================")
    header = "method".ljust(12) + "".join(f"b={b}".rjust(12) for b in sorted(budgets))
    print(header)
    for m in methods:
        line = m.ljust(12)
        for b in sorted(budgets):
            sm = agg[m][b]["score_mean"]
            line += ("-" if sm is None else f"{sm:.4f}").rjust(12)
        print(line)

    if args.target_score is not None:
        ett = _evals_to_target(rows, methods, budgets, args.target_score)
        print(f"\n---- evals-to-target (target_score={args.target_score}) ----")
        for m in methods:
            print(f"  {m.ljust(12)} 最小预算 = {ett[m] if ett[m] is not None else '未达到'}")

    _plot(agg, methods, budgets, args.results_dir, args.target_score)
    print("\n[sweep] 完成。")


if __name__ == "__main__":
    main()
