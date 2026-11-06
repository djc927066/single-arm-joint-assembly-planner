"""Tables for the weekly report: grasp screening speed and cost ranking vs WRS RRT.

    python -m lazy_cost._design.summ_report
Writes lazy_cost/_output/report_tables.md (markdown tables) and report_tables.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "_output"


def grasp_table(lines, J):
    g = json.loads((OUT / "grasp_speed_triangular_frame6.json").read_text(encoding="utf-8"))
    lines += ["## 1. 共同抓法筛选：cuRobo GPU 批量 vs WRS 逐个", "",
              f"规则相同：终点（插入位姿）IK + 无碰撞 → 起点（40 个候选位姿）IK + 无碰撞；世界 = 桌子 + 底座。"
              f" WRS IK：{g['ik_backend_wrs']}（单次求解，逐个调用）；WRS 时间 = 抽样实测每次耗时 × 完整筛选所需调用次数。"
              f" cuRobo 一次性初始化 {g['curobo_setup_s']:.1f} s（不计入）。", "",
              "| 零件 | WRS 抓法数 | 终点可行（cuRobo） | 两端可行对数（cuRobo） | cuRobo 用时 (s) | WRS 每次调用 (ms) | WRS 估计用时 (s) | 加速比 | 抽样判断一致率 终点 / 起点 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in g["parts"]:
        c, w, a = r["curobo"], r["wrs"], r["agree"]
        lines.append(f"| {r['part']} | {r['grasps']} | {c['goal_ik_free']} | {c['pairs']} | {c['total_s']:.2f} | "
                     f"{w['ms_per_goal_call']:.0f} / {w['ms_per_pick_call']:.0f} | {w['total_s']:.0f} | ×{r['speedup']:.0f} | "
                     f"{a['goal_same']:.2f} / {a['pick_same']:.2f} |")
    t = g["total"]
    lines += [f"| **合计** | | | {t['curobo_pairs']} | **{t['curobo_s']:.1f}** | | **{t['wrs_s']:.0f}（≈{t['wrs_s'] / 60:.0f} min）** | **×{t['speedup']:.0f}** | |", ""]
    J["grasp"] = g


def rank_table(lines, J):
    rows = [json.loads(l) for l in (OUT / "wrs_rank_triangular_frame6.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if r.get("metrics")]
    names = {"lb": "只用直线下界", "old": "下界 × (1 + 0.243 + 1.09·深度)（旧）", "new": "下界 + 4.7·深度（新）"}
    lines += ["## 2. 候选位姿排序：快速 cost vs WRS RRT 参照", "",
              f"{len(ok)} 个场景（只有零件做障碍，随机顺序/步骤/桌面布局），每个场景 ≤10 个候选位姿 × 3 种抓法。"
              " 参照：每段先用 WRS 网格碰撞检查关节直线，没被挡 = 下界；被挡 = WRS RRT-Connect + 捷径 + 圆角 + TOPP 计时。", "",
              "| 排序方法 | Spearman | 第一名命中 | 第一名在最优 2% 内 | 前 3 名召回 | 平均遗憾 | 最大遗憾 | 候选 cost 平均误差 |",
              "|---|---|---|---|---|---|---|---|"]
    summ = {}
    for k, nm in names.items():
        m = [r["metrics"][k] for r in ok]
        err = [abs(r["pred"][k][x] / r["truth"][x] - 1) for r in ok for x in r["truth"] if r["truth"][x]]
        f = lambda q: float(np.mean([x[q] for x in m]))
        s = dict(spearman=f("spearman"), top1=f("top1"), top1_2pct=f("top1_within2pct"), top3=f("topk_recall"),
                 regret=f("regret"), max_regret=float(max(x["regret"] for x in m)), cost_err=float(np.mean(err)))
        summ[k] = s
        lines.append(f"| {nm} | {s['spearman']:.3f} | {s['top1']:.2f} | {s['top1_2pct']:.2f} | {s['top3']:.2f} | "
                     f"{100 * s['regret']:.2f}% | {100 * s['max_regret']:.1f}% | {100 * s['cost_err']:.1f}% |")
    fast = np.array([r["fast_s"] for r in rows])
    wrs = np.array([r["wrs_s"] for r in rows])
    cand = np.array([r["alive"] for r in rows])
    rrt = np.array([r["wrs_rrt_segments"] for r in rows])
    lines += ["", "| 用时（每个场景） | 快速 cost（GPU） | WRS RRT 参照 |", "|---|---|---|",
              f"| 平均 | {1e3 * fast.mean():.0f} ms | {wrs.mean():.1f} s |",
              f"| 每个候选位姿 | {1e3 * (fast / cand).mean():.1f} ms | {(wrs / cand).mean():.2f} s |",
              f"| 比值 | 1 | ×{(wrs / fast).mean():.0f} |", ""]
    # straight-line agreement: our sphere checker vs WRS mesh
    segs = [s for r in rows for s in r["segments"]]
    a = lambda o, w: sum(1 for s in segs if (s["ours"] == "straight") == o and (s["wrs"] == "straight") == w)
    lines += [f"直线是否被挡（{len(segs)} 段）：", "", "| | WRS：直线可走 | WRS：被挡（RRT） |", "|---|---|---|",
              f"| 我们的检查器：直线可走 | {a(True, True)} | {a(True, False)} |",
              f"| 我们的检查器：被挡 | {a(False, True)} | {a(False, False)} |", ""]
    rr = [s for s in segs if s["wrs"] == "rrt"]
    if rr:
        lines.append(f"WRS RRT 段：{len(rr)} 段，平均每段 {np.mean([s['wrs_s'] for s in rr]):.1f} s；RRT 失败 "
                     f"{sum(s['wrs'] == 'rrt_failed' for s in segs)} 段。")
    J["rank"] = dict(contexts=len(ok), methods=summ, fast_ms=float(1e3 * fast.mean()), wrs_s=float(wrs.mean()),
                     ratio=float((wrs / fast).mean()), rrt_segments=int(rrt.sum()), segments=len(segs))
    an = OUT / "wrs_rank_segments.json"
    if an.exists():
        A = json.loads(an.read_text(encoding="utf-8"))["summary"]
        fb, tb = A["false_blocked"], A["true_blocked"]
        lines += ["", "### 2b. 为什么修正后排序反而变差：两套碰撞模型对“被挡”的判断不同", "",
                  "| 我们判为被挡的段 | 段数 | 穿透深度 p10 / 中位 / p90 (mm) | WRS 实际多花时间 中位 (s) | 新公式惩罚 中位 (s) |",
                  "|---|---|---|---|---|",
                  f"| WRS 网格也被挡（真被挡） | {tb['n']} | {' / '.join(f'{v:.0f}' for v in tb['depth_mm'])} | {tb['wrs_extra_s'][1]:.2f}（×{tb['wrs_over_lb'][1]:.2f} 下界） | {tb['new_penalty_s'][1]:.2f} |",
                  f"| WRS 网格直线可走（误判） | {fb['n']} | {' / '.join(f'{v:.0f}' for v in fb['depth_mm'])} | 0 | {fb['new_penalty_s'][1]:.2f} |", "",
                  f"真被挡的段上，深度与多花时间的 Spearman = {tb['rho_depth_extra']:.2f}，拟合系数 c = {tb['fit_c']:.1f} s/m（现用 4.7）。"
                  f" 误判来自我们的球模型更保守（机械臂碰撞球 + 2 mm 余量、零件表面 4 mm 小球）。", "",
                  "### 2c. 改进：被挡段先用网格复核直线，再加深度修正（用同一批数据重算）", "",
                  "| 排序方法 | Spearman | 第一名命中 | 第一名在最优 2% 内 | 前 3 名召回 | 平均遗憾 | 最大遗憾 | 候选 cost 平均误差 |",
                  "|---|---|---|---|---|---|---|---|"]
        nm = {"lb": "只用直线下界", "old": "旧公式", "new": "新公式（球模型判被挡）", "new_mesh": "**新公式 + 网格复核被挡段**"}
        for k, s in A["rescored"].items():
            lines.append(f"| {nm[k]} | {s['spearman']:.3f} | {s['top1']:.2f} | {s['top1_2pct']:.2f} | {s['top3']:.2f} | "
                         f"{100 * s['regret']:.2f}% | {100 * s['max_regret']:.1f}% | {100 * s['cost_err']:.1f}% |")
        lines += ["", f"网格复核一条直线中位 {1e3 * A['mesh_recheck_s_per_segment']:.0f} ms（WRS/ODE，CPU），只对球模型判为被挡的段做"
                  f"（平均每场景 {sum(1 for _ in range(1)) and round((tb['n'] + fb['n']) / max(len(ok), 1))} 段）。"
                  " 注意：复核与参照用的是同一套 WRS 网格碰撞，所以这一行对“是否被挡”是理想情况；绕行时间仍是深度估计，不用 RRT。", ""]
        J["rank_analysis"] = A


def main():
    lines, J = ["# 本周实验表格", ""], {}
    for fn in (grasp_table, rank_table):
        try:
            fn(lines, J)
        except FileNotFoundError as e:
            lines += [f"（缺少数据：{e.filename}）", ""]
    (OUT / "report_tables.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "report_tables.json").write_text(json.dumps(J, indent=1, default=float), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
