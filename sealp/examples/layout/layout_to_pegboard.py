"""把布局搜索结果(.layout)里的零件 staging 位置换算成洞洞板(pegboard)孔位坐标。

桌面是一块洞洞板：宽(x 轴方向)有 ``--nx`` 个孔，长(y 轴方向)有 ``--ny`` 个孔。
本脚本读取 ``WorkspaceLayout`` 的连续坐标 (x, y)[m]，映射到最近的孔，
输出 "第几列(x) / 第几行(y)"，并给出对齐误差(mm)，方便实物摆放。

孔栅格默认从 sample_config.yaml 里的 work_table 几何推出：
  - x 范围 = [pos_x - ext_x/2, pos_x + ext_x/2]
  - y 范围 = [pos_y - ext_y/2, pos_y + ext_y/2]
两种孔模型：
  - center (默认): 孔在每个网格的格心, pitch = ext/N, 第 1 孔距边 0.5*pitch。
                   这种模型下 0.6/23 == 1.2/46 == 26.087mm, 正好方形栅格。
  - edge        : 第 1 孔在边角, pitch = ext/(N-1), 首尾孔落在桌边。
pitch / 原点 / 列行数 / 计数起点 都可用命令行覆盖。

用法示例:
  python -m sealp.examples.layout.layout_to_pegboard
  python -m sealp.examples.layout.layout_to_pegboard --nx 23 --ny 46 --hole-model center
  python -m sealp.examples.layout.layout_to_pegboard --layout path/to/x.layout --x-from max
"""
import argparse
import math
import os

import yaml

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SEALP_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
DEFAULT_CONFIG = os.path.join(SEALP_ROOT, "config", "sample_config.yaml")
DEFAULT_LAYOUT = os.path.join(_THIS_DIR, "_output", "tower_optimal_initial.layout")


def _load_table(config_path, table_name):
    """从 config 读取指定桌子的 extent / pos。"""
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    obstacles = (cfg.get("environment", {}) or {}).get("obstacles", []) or []
    for ob in obstacles:
        if ob.get("name") == table_name:
            ext = [float(v) for v in ob["extent"]]
            pos = [float(v) for v in ob["pos"]]
            return ext, pos
    raise SystemExit(f"[ERROR] 在 {config_path} 里没找到 obstacle name='{table_name}'")


def _load_staging(layout_path):
    """读取 .layout(YAML) 里的 staging / assembly_station 坐标。返回 {name: (x, y)}。"""
    with open(layout_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    out = {}
    staging = data.get("staging", {}) or {}
    for pid, entry in staging.items():
        pos = entry.get("pos", [0, 0, 0])
        out[pid] = (float(pos[0]), float(pos[1]))
    station = data.get("assembly_station", {}) or {}
    if "pos" in station:
        p = station["pos"]
        out["[assembly_station]"] = (float(p[0]), float(p[1]))
    return out, data


def _build_grid(lo, hi, n, model):
    """返回 (centers[list], pitch)。centers[i] 是第 i 个孔(0-index)的坐标。"""
    span = hi - lo
    if model == "center":
        pitch = span / n
        centers = [lo + (i + 0.5) * pitch for i in range(n)]
    elif model == "edge":
        pitch = span / (n - 1) if n > 1 else span
        centers = [lo + i * pitch for i in range(n)]
    else:
        raise SystemExit(f"[ERROR] 未知 hole-model: {model}")
    return centers, pitch


def _nearest_index(centers, value):
    """最近孔的 0-index 及其坐标。"""
    best_i, best_d = 0, float("inf")
    for i, c in enumerate(centers):
        d = abs(c - value)
        if d < best_d:
            best_i, best_d = i, d
    return best_i, centers[best_i]


def main():
    ap = argparse.ArgumentParser(description="把 .layout 的 staging 位置换算成洞洞板孔位")
    ap.add_argument("--layout", default=DEFAULT_LAYOUT, help=".layout 文件路径")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="sample_config.yaml 路径")
    ap.add_argument("--table-name", default="work_table", help="洞洞板对应的 obstacle 名称")
    ap.add_argument("--nx", type=int, default=23, help="x 轴方向(宽)孔数")
    ap.add_argument("--ny", type=int, default=46, help="y 轴方向(长)孔数")
    ap.add_argument("--hole-model", choices=["center", "edge"], default="center",
                    help="center: 孔在格心(pitch=ext/N); edge: 第1孔在桌边(pitch=ext/(N-1))")
    # 手动覆盖栅格(可选): 给了就不从 config 推
    ap.add_argument("--x-min", type=float, default=None, help="覆盖 x 最小值(m)")
    ap.add_argument("--x-max", type=float, default=None, help="覆盖 x 最大值(m)")
    ap.add_argument("--y-min", type=float, default=None, help="覆盖 y 最小值(m)")
    ap.add_argument("--y-max", type=float, default=None, help="覆盖 y 最大值(m)")
    # 计数起点: min=从坐标小的一端开始数第1个; max=从坐标大的一端数
    ap.add_argument("--x-from", choices=["min", "max"], default="min",
                    help="x 方向第1列从哪端开始数(默认 min, 即 -x 端)")
    ap.add_argument("--y-from", choices=["min", "max"], default="min",
                    help="y 方向第1行从哪端开始数(默认 min, 即 -y 端)")
    args = ap.parse_args()

    ext, pos = _load_table(args.config, args.table_name)
    x_min = args.x_min if args.x_min is not None else pos[0] - ext[0] / 2.0
    x_max = args.x_max if args.x_max is not None else pos[0] + ext[0] / 2.0
    y_min = args.y_min if args.y_min is not None else pos[1] - ext[1] / 2.0
    y_max = args.y_max if args.y_max is not None else pos[1] + ext[1] / 2.0

    cx, pitch_x = _build_grid(x_min, x_max, args.nx, args.hole_model)
    cy, pitch_y = _build_grid(y_min, y_max, args.ny, args.hole_model)

    staging, _ = _load_staging(args.layout)

    print("=" * 78)
    print(f"洞洞板栅格 (hole-model={args.hole_model})")
    print(f"  桌面 '{args.table_name}': extent={ext[0]:.3f} x {ext[1]:.3f} m, "
          f"center=({pos[0]:.3f}, {pos[1]:.3f})")
    print(f"  x: [{x_min:+.4f}, {x_max:+.4f}] m -> {args.nx} 孔, pitch={pitch_x*1000:.2f} mm "
          f"(列号从 {args.x_from} 端起)")
    print(f"  y: [{y_min:+.4f}, {y_max:+.4f}] m -> {args.ny} 孔, pitch={pitch_y*1000:.2f} mm "
          f"(行号从 {args.y_from} 端起)")
    print(f"  layout: {args.layout}")
    print("=" * 78)
    header = f"{'part':<16}{'x[m]':>9}{'y[m]':>9}  {'列(x/'+str(args.nx)+')':>10}{'行(y/'+str(args.ny)+')':>10}   {'误差[mm]':>9}"
    print(header)
    print("-" * 78)

    for pid in staging:
        x, y = staging[pid]
        ix, hx = _nearest_index(cx, x)
        iy, hy = _nearest_index(cy, y)
        # 计数起点换算 (1-index)
        col = ix + 1 if args.x_from == "min" else args.nx - ix
        row = iy + 1 if args.y_from == "min" else args.ny - iy
        err = math.hypot(x - hx, y - hy) * 1000.0
        flag = "" if err < pitch_x * 1000 * 0.25 else "  <- 偏离孔心较大"
        print(f"{pid:<16}{x:>9.4f}{y:>9.4f}  {col:>10d}{row:>10d}   {err:>9.2f}{flag}")

    print("-" * 78)
    print("说明: 列(x) 沿桌面宽度方向; 行(y) 沿桌面长度方向。误差=该点到最近孔心的距离。")
    print("如果你的孔模型/原点不同(例如孔在格线交点而非格心, 或从另一角数),")
    print("可用 --hole-model / --x-from / --y-from / --x-min 等参数调整。")


if __name__ == "__main__":
    main()
