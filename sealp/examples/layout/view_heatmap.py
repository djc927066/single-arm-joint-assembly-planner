#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""查看 find_optimal_initial_layout_tower_heatmap_pso.py 产出的可行性热力图缓存
================================================================================

读取 ``_output/heatmap_cache_*.pkl``, 用 matplotlib 把每个装配区里各零件的
"可抓取分图"画出来(颜色越亮=该 staging 位置可行 grasp 越多), 方便组会展示
"先验长什么样"。

用法:
    # 默认自动找 _output 里最新的 heatmap_cache_*.pkl, 每个区存一张 PNG
    python -m sealp.examples.layout.view_heatmap

    # 指定缓存文件
    python -m sealp.examples.layout.view_heatmap --pkl path/to/heatmap_cache_xxx.pkl

    # 只看某个区, 并弹窗显示(而不仅是存 PNG)
    python -m sealp.examples.layout.view_heatmap --region r1_c1 --show

输出:
    PNG 存到 _output/heatmap_view/<region>.png
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_THIS_DIR, "_output")


def _find_latest_pkl() -> str:
    cands = glob.glob(os.path.join(_OUTPUT_DIR, "heatmap_cache_*.pkl"))
    if not cands:
        raise FileNotFoundError(
            f"在 {_OUTPUT_DIR} 没找到 heatmap_cache_*.pkl, "
            f"请先跑 find_optimal_initial_layout_tower_heatmap_pso.py 生成热力图。"
        )
    return max(cands, key=os.path.getmtime)


def _print_ranking(data: dict) -> None:
    print("\n[region promise ranking] (sum of per-part feasible-cell fraction)")
    ranked = sorted(data.items(), key=lambda kv: kv[1].get("promise", 0.0), reverse=True)
    for rid, reg in ranked:
        print(f"  {rid:8s} rc={reg.get('rc')}  promise={reg.get('promise', 0.0):.3f}")


def _plot_region(plt, region_id: str, reg: dict, out_dir: str, show: bool) -> str:
    parts = reg["parts"]
    pids = list(parts.keys())
    n = len(pids)
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    fig.suptitle(f"Feasibility heatmap  region={region_id}  rc={reg.get('rc')}  "
                 f"promise={reg.get('promise', 0.0):.3f}", fontsize=13)

    for k, pid in enumerate(pids):
        ax = axes[k // ncol][k % ncol]
        hm = parts[pid]
        g = hm["grid"]                  # shape (nx, ny): 行=x 列=y
        xs, ys = hm["xs"], hm["ys"]
        extent = [float(ys[0]), float(ys[-1]), float(xs[0]), float(xs[-1])]
        im = ax.imshow(g, origin="lower", aspect="auto", extent=extent,
                       cmap="viridis", interpolation="nearest")
        n_ok = int((g > 0).sum())
        ax.set_title(f"{pid}\nfeasible {n_ok}/{g.size}  max_grasps={int(g.max())}",
                     fontsize=9)
        ax.set_xlabel("y (m)", fontsize=8)
        ax.set_ylabel("x (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 关掉多余空子图
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{region_id}.png")
    fig.savefig(png, dpi=130)
    print(f"  saved {png}")
    if not show:
        plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="", help="热力图缓存路径, 默认自动找最新的")
    ap.add_argument("--region", default="", help="只画指定区, 默认全部")
    ap.add_argument("--show", action="store_true", help="弹窗显示(默认只存 PNG)")
    args = ap.parse_args()

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pkl_path = args.pkl or _find_latest_pkl()
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"[view] loaded {pkl_path}")
    print(f"[view] regions = {list(data.keys())}")
    _print_ranking(data)

    out_dir = os.path.join(_OUTPUT_DIR, "heatmap_view")
    regions = [args.region] if args.region else list(data.keys())
    print(f"\n[view] rendering {len(regions)} region(s) -> {out_dir}")
    for rid in regions:
        if rid not in data:
            print(f"  [WARN] region {rid} 不在缓存里, 跳过。可选: {list(data.keys())}")
            continue
        _plot_region(plt, rid, data[rid], out_dir, args.show)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
