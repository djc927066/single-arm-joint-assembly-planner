"""One figure with the five scenes of experiment 1 (collision bench): top view + 3D view per scene.

    & $py -X utf8 -m joint_search.viz_bench_scenes

Reads the scene layouts the bench actually used (joint_search/_output/bench/collision_bench.json) and
the search's IK table (joint_search/_cache) for the candidate poses; no GPU needed.
Output: joint_search/_output/bench/scenes_all.png
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "_output" / "bench"
CACHE = HERE / "_cache" / "triangular_frame6_n40_ik_v3_closed_d2_b1_r3.pkl"
INFO = {
    "A_empty": ("A 空场景", "待装零件 post_A\n桌上无其他零件，未安装任何零件\n障碍物 0"),
    "B_crowded_table": ("B 桌面拥挤", "待装零件 post_A\n其余 5 个零件都摆在桌上（靠近装配位）\n障碍物 5"),
    "C_mid_assembly": ("C 装配中途", "待装零件 beam_BC（第 5 步）\n已装 3 根立柱 + beam_AB，桌上 beam_CA\n障碍物 5"),
    "D_last_insert": ("D 最后一件", "待装零件 beam_CA（第 6 步）\n其余 5 个零件已安装（横梁间隙 4–7 mm）\n障碍物 5"),
    "E_rack10": ("E rack10", "rack10 的 10 个零件全部摆在桌上\n关节限位内随机 300 个构型\n障碍物 10"),
}
TESTS = "每个场景 300 次查询：插入时刻 75 + 抓取时刻 75 + 空手移动段 75 + 持件移动段 75"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull
    from lazy_cost.scene import load_scene
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    res = json.loads((OUT / "collision_bench.json").read_text(encoding="utf-8"))["results"]
    cat = pickle.loads(CACHE.read_bytes())
    frame, rack = load_scene("triangular_frame6"), load_scene("rack10")

    fig = plt.figure(figsize=(26, 12.5))
    for col, name in enumerate(INFO):
        R = res[name]
        sc = rack if name == "E_rack10" else frame
        installed, table = R.get("installed", []), R["table"]
        p = R.get("p")
        xs = []
        if p:   # candidates of p that the bench used (feasible in the IK table, >= 1 cm from parts on the table)
            xs = [x for x in cat["pick"][p] if all(sc.polygons[p][x].distance(sc.polygons[q][k]) >= 0.01 for q, k in table.items())]
        objs = ([("base", sc.goal["base"], "#bdbdbd")] if "base" in sc.goal else []) + \
               [(q, sc.goal[q], "#4c78a8") for q in installed] + [(q, sc.pose[q][k], "#f58518") for q, k in table.items()]

        # ---- top view -------------------------------------------------------------------------------
        ax = fig.add_subplot(2, 5, col + 1)
        c, s = sc.table_center, sc.table_size
        ax.add_patch(Rectangle((c[0] - s[0] / 2, c[1] - s[1] / 2), s[0], s[1], fc="#f2efe8", ec="#999"))

        def foot(pid, T, **kw):
            v = np.asarray(sc.meshes[pid].vertices) @ T[:3, :3].T + T[:3, 3]
            ax.add_patch(Polygon(v[ConvexHull(v[:, :2]).vertices, :2], **kw))
            return v[:, :2].mean(0)
        for x in xs:
            foot(p, sc.pose[p][x], fc="none", ec="#2ca02c", ls=":", lw=0.8)
        for pid, T, colr in objs:
            m = foot(pid, T, fc=colr, ec="k", alpha=0.6 if pid == "base" else 0.9)
            if pid != "base":
                ax.text(*m, pid, ha="center", va="center", fontsize=7, color="w" if colr == "#4c78a8" else "k")
        if p:
            m = foot(p, sc.goal[p], fc="none", ec="#d62728", lw=2.0, ls="--")
            ax.text(m[0], m[1] - 0.025, f"{p} 装配位", ha="center", fontsize=8, color="#d62728")
        ax.plot(0, 0, "k^", ms=10)
        ax.text(0.0, -0.03, "机械臂基座", ha="center", fontsize=8)
        ax.set_xlim(-0.09, 0.56); ax.set_ylim(-0.32, 0.32); ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        title, desc = INFO[name]
        ax.set_title(f"{title}\n{desc}", fontsize=11, loc="left")

        # ---- 3D view --------------------------------------------------------------------------------
        a3 = fig.add_subplot(2, 5, col + 6, projection="3d")
        a3.computed_zorder = False
        tx, ty = c[0] - s[0] / 2, c[1] - s[1] / 2
        a3.add_collection3d(Poly3DCollection([[(tx, ty, 0), (tx + s[0], ty, 0), (tx + s[0], ty + s[1], 0), (tx, ty + s[1], 0)]],
                                             facecolor="#f2efe8", edgecolor="#999", alpha=1.0, zorder=0))
        for pid, T, colr in objs + ([(p, sc.goal[p], "#d62728")] if p else []):
            m = sc.meshes[pid]
            v = np.asarray(m.vertices) @ T[:3, :3].T + T[:3, 3]
            a3.add_collection3d(Poly3DCollection(v[np.asarray(m.faces)], facecolors=colr, edgecolor="none",
                                                 alpha=0.35 if (pid == p and colr == "#d62728") else 1.0, zorder=1, shade=True))
        a3.scatter([0], [0], [0], marker="^", s=60, c="k")
        a3.set_xlim(-0.09, 0.56); a3.set_ylim(-0.32, 0.32); a3.set_zlim(0, 0.25)
        a3.set_box_aspect((0.65, 0.64, 0.25 * 1.6), zoom=1.25); a3.view_init(32, -62)
        a3.tick_params(labelsize=6)
        n = sum(v["n"] for k, v in R["confusion"].items() if k != "all")
        a3.set_title(f"{title}：3D 视图（查询 {n} 次）", fontsize=11, pad=0)

    fig.suptitle("实验一：五个测试场景　　蓝 = 已安装　橙 = 摆在桌上（障碍物）　灰 = 装配底座　红虚线 / 半透明红 = 待装零件的装配位　"
                 "绿虚线 = 待装零件的候选摆放位置\n" + TESTS + "（D 只有 65 个可插入的抓法，共 290 次；E 为随机构型）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3.0)
    fig.savefig(OUT / "scenes_all.png", dpi=110)
    print("->", OUT / "scenes_all.png")


if __name__ == "__main__":
    main()
