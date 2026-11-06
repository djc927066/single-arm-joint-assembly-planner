"""布局搜索方法批量对比实验 (论文用)。

统一驱动以下方法, 多 seed 运行, 汇总指标到 results.csv + summary.json:

    Random Search / NSGA-II / Global Search /
    MLP / DeepSets / Set Transformer / GCN / GAT / Transformer /
    CVAE / Diffusion / SAGPN (Ours)

每个方法作为子进程运行 (调用对应的现有脚本), 保证与真实评估口径完全一致;
所有 NN 方法产出的 layout 同样经过原始 evaluate_layout / pattern_refine / L3。

指标:
    best layout_score / L2 feasible rate / L3 success / #exact evaluate_layout /
    runtime / cache hit rate / avg common grasp / avg transport distance /
    score components / 不同 seed 的成功率。

用法示例:
    python -m sealp.examples.layout.run_layout_experiments \
        --methods random,nsga2,global,mlp,gcn,gat,sagpn \
        --seeds 0,1,2 \
        --checkpoint-dir checkpoints/layout_models \
        --results-dir sealp/examples/layout/_output/experiments \
        --max-evals 200 --top-k-proposals 64 --global-elite 3 \
        --common "--cdprim-type box --planner-obstacle-mode staging_aware"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_THIS_DIR, "_output")

# 方法 -> (模块名, 是否 NN, 是否生成式仅作标注)
METHOD_MODULE = {
    "random": "sealp.examples.layout.find_optimal_initial_layout_tower_strict_pycharm",
    "nsga2": "sealp.examples.layout.find_optimal_initial_layout_tower_nsga2_v1",
    "global": "sealp.examples.layout.find_optimal_initial_layout_tower_global",
}
NN_METHODS = ["mlp", "deepsets", "set_transformer", "gcn", "gat",
              "transformer", "pointnet", "cvae", "diffusion", "sagpn", "seqrel"]
NEURAL_MODULE = "sealp.examples.layout.find_optimal_initial_layout_tower_neural"


def _parse_args():
    p = argparse.ArgumentParser(description="Layout search experiments runner")
    p.add_argument("--methods", default="random,nsga2,global,mlp,gcn,gat,sagpn")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--checkpoint-dir", default=os.path.join("checkpoints", "layout_models"))
    p.add_argument("--results-dir", default=os.path.join(_OUTPUT_DIR, "experiments"))
    p.add_argument("--python", default=sys.executable, help="运行子进程的解释器")
    p.add_argument("--common", default="", help="所有方法共享的透传 CLI 参数 (字符串)")
    p.add_argument("--max-evals", type=int, default=200)
    p.add_argument("--top-k-proposals", type=int, default=64)
    p.add_argument("--scorer-pool", type=int, default=400)
    p.add_argument("--global-elite", type=int, default=3)
    p.add_argument("--global-refine-steps", default="0.03,0.015,0.008")
    p.add_argument("--n-samples", type=int, default=40)
    p.add_argument("--enable-l3", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    return p.parse_args()


def _build_cmd(args, method: str, seed: int, out_name: str) -> List[str]:
    py = args.python
    common = shlex.split(args.common)
    max_evals = str(args.max_evals)
    base = [py, "-m"]

    if method in METHOD_MODULE:
        cmd = base + [METHOD_MODULE[method]]
        cmd += ["--output-name", out_name, "--seed", str(seed), "--n-samples", str(args.n_samples)]
        if method == "nsga2":
            cmd += ["--nsga-max-evals", max_evals]
        elif method == "global":
            cmd += ["--global-max-evals", max_evals,
                    "--global-elite", str(args.global_elite),
                    "--global-refine-steps", args.global_refine_steps]
        # random(strict) 没有 eval 上限旋钮, 用 n-samples 控制评估次数
    elif method in NN_METHODS:
        ckpt = os.path.join(args.checkpoint_dir, f"{method}_best.pt")
        cmd = base + [NEURAL_MODULE, "--model", method, "--checkpoint", ckpt,
                      "--output-name", out_name, "--seed", str(seed),
                      "--n-samples", str(args.n_samples),
                      "--top-k-proposals", str(args.top_k_proposals),
                      "--scorer-pool", str(args.scorer_pool),
                      "--global-elite", str(args.global_elite),
                      "--global-refine-steps", args.global_refine_steps,
                      "--global-max-evals", max_evals]
    else:
        raise ValueError(f"未知方法: {method}")

    if args.enable_l3:
        cmd += ["--enable-l3"]
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
        "first_l2_ok_eval": _num(r"first L2_OK at eval # = (\d+)", int),
        "best_score_eval": _num(r"best score first at\s+= #(\d+)", int),
        "l3_passed": 1.0 if re.search(r"\[OK\] L3 passed", text) else None,
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


def _metrics_from_debug(dbg: Dict) -> Dict[str, Optional[float]]:
    if not dbg:
        return {"best_score": None, "avg_grasp": None, "avg_dist": None,
                "grasp": None, "manip": None, "dist": None, "rot": None, "spatial": None,
                "first_l2_ok_eval": None, "best_score_eval": None}
    comp = dbg.get("score_components", {})
    gc = list((dbg.get("grasp_counts") or {}).values())
    pd = list((dbg.get("per_part_dist") or {}).values())
    ses = dbg.get("search_eval_stats") or {}
    return {
        "best_score": dbg.get("score"),
        "avg_grasp": float(statistics.mean(gc)) if gc else None,
        "avg_dist": float(statistics.mean(pd)) if pd else None,
        "grasp": comp.get("grasp"),
        "manip": comp.get("manip"),
        "dist": comp.get("dist"),
        "rot": comp.get("rot"),
        "spatial": comp.get("spatial_y_distribution"),
        "first_l2_ok_eval": ses.get("first_l2_ok_eval"),
        "best_score_eval": ses.get("best_score_eval"),
    }


def _run_one(args, method: str, seed: int) -> Dict:
    out_name = f"exp_{method}_seed{seed}"
    cmd = _build_cmd(args, method, seed, out_name)
    log_dir = os.path.join(args.results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{out_name}.log")

    print(f"\n>>> [{method} seed={seed}] {' '.join(cmd)}")
    if args.dry_run:
        return {"method": method, "seed": seed, "cmd": " ".join(cmd)}

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=os.getcwd())
    dt = time.time() - t0

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    parsed = _parse_stdout(text)
    dbg = _read_debug(out_name)
    md = _metrics_from_debug(dbg)

    real = parsed.get("real_evals")
    feas = parsed.get("feasible_found")
    cache = parsed.get("cache_hits")
    row = {
        "method": method,
        "seed": seed,
        "exit_code": proc.returncode,
        "runtime_s": round(parsed.get("wall_time") or dt, 2),
        "best_score": md["best_score"],
        "feasible": 1 if md["best_score"] is not None else 0,
        "real_evals": real,
        "feasible_found": feas,
        "l2_feasible_rate": (feas / real) if (real and feas is not None) else None,
        "cache_hits": cache,
        "cache_hit_rate": (cache / (real + cache)) if (real and cache is not None) else None,
        "first_l2_ok_eval": parsed.get("first_l2_ok_eval") or md.get("first_l2_ok_eval"),
        "best_score_eval": parsed.get("best_score_eval") or md.get("best_score_eval"),
        "l3_passed": parsed.get("l3_passed"),
        "avg_grasp": md["avg_grasp"],
        "avg_dist": md["avg_dist"],
        "score_grasp": md["grasp"],
        "score_manip": md["manip"],
        "score_dist": md["dist"],
        "score_rot": md["rot"],
        "score_spatial": md["spatial"],
        "log": log_path,
    }
    print(f"<<< [{method} seed={seed}] score={row['best_score']} "
          f"feasible={row['feasible']} evals={real} runtime={row['runtime_s']}s")
    return row


def _aggregate(rows: List[Dict]) -> Dict:
    by_method: Dict[str, List[Dict]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.mean(vals), 4) if vals else None

    def _std(vals):
        vals = [v for v in vals if v is not None]
        return round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0

    summary = {}
    for m, rs in by_method.items():
        scores = [r["best_score"] for r in rs]
        summary[m] = {
            "n_runs": len(rs),
            "success_rate": round(sum(r["feasible"] for r in rs) / len(rs), 3),
            "best_score_mean": _mean(scores),
            "best_score_std": _std(scores),
            "best_score_max": max([s for s in scores if s is not None], default=None),
            "l2_feasible_rate_mean": _mean([r["l2_feasible_rate"] for r in rs]),
            "l3_success_rate": round(sum(1 for r in rs if r["l3_passed"]) / len(rs), 3),
            "real_evals_mean": _mean([r["real_evals"] for r in rs]),
            "runtime_s_mean": _mean([r["runtime_s"] for r in rs]),
            "cache_hit_rate_mean": _mean([r["cache_hit_rate"] for r in rs]),
            "avg_grasp_mean": _mean([r["avg_grasp"] for r in rs]),
            "avg_dist_mean": _mean([r["avg_dist"] for r in rs]),
        }
    return summary


def _maybe_plot(summary: Dict, results_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[exp] matplotlib 不可用, 跳过绘图。")
        return
    methods = list(summary.keys())
    scores = [summary[m]["best_score_mean"] or 0.0 for m in methods]
    fig, ax = plt.subplots(figsize=(max(6, len(methods)), 4))
    ax.bar(methods, scores, color="#4C72B0")
    ax.set_ylabel("mean best layout_score")
    ax.set_title("Layout search methods comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = os.path.join(results_dir, "best_score_bar.png")
    plt.savefig(out, dpi=120)
    print(f"[exp] 绘图 -> {out}")


def main():
    args = _parse_args()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.replace(",", " ").split()]
    os.makedirs(args.results_dir, exist_ok=True)

    rows: List[Dict] = []
    for method in methods:
        for seed in seeds:
            rows.append(_run_one(args, method, seed))

    if args.dry_run:
        return

    csv_path = os.path.join(args.results_dir, "results.csv")
    fields = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[exp] results -> {csv_path}")

    summary = _aggregate(rows)
    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[exp] summary -> {summary_path}")

    print("\n========== Summary ==========")
    for m, s in summary.items():
        print(f"{m:16s} score={s['best_score_mean']}±{s['best_score_std']} "
              f"success={s['success_rate']} evals={s['real_evals_mean']} "
              f"runtime={s['runtime_s_mean']}s")

    _maybe_plot(summary, args.results_dir)


if __name__ == "__main__":
    main()
