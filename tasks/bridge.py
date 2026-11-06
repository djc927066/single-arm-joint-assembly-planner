"""小型 Bridge 基准场景：底座 + 两根支柱 + 横梁 + 载荷。

这是联合搜索的第一个调通用例。选它的理由：可动件少（4 个）、接触以平面对平面
为主（asmcore 的网格接触后端直接可算、不需要 SDF），而且它天然包含一个单臂
绕不开的不稳定中间状态——横梁只搭在一根支柱上时质心悬空。

**固定件**：`table` 与 `base` 都是 `fixed=True`。底座位置由人给定，既不参与
装配顺序搜索，也不是布局变量；它对应论文一 BLS 里的 preassembled 基座
（`sealp/examples/layout/bsfs` 的 `preassembled_pid`）。需要优化的只有
post_l / post_r / beam / load 四件的初始摆放与装配顺序。

底座上的两个方形盲孔把支柱的插入方向约束成 −Z 单轴，几何由
`tasks/make_base_plate.py` 生成为 `assets/base_plate.stl`（单位：米）。

    load            ┌────┐              0.188 ~ 0.248
    beam     ┌──────┴────┴──────┐       0.158 ~ 0.188
    posts    │  ┌─┐        ┌─┐  │       0.008 ~ 0.158
    base    ┌┴──┴─┴────────┴─┴──┴┐      0.000 ~ 0.020   孔底 0.008
    table   └────────────────────┘     -0.020 ~ 0.000

关于孔的间隙（实测结论，勿重复踩）
----------------------------------
0.2 mm 单边间隙下，四个孔壁被 `analyze_contacts` 判为 `near` 而非 `active`，
而约束与方向锥求解只吃 `active` 接触。后果是：孔壁**既不参与静力平衡、也不
收窄装配方向锥**——只装 post_l 时方向锥仍是整个上半球（dim=3, A=10），与没有
底座时完全一致。把 `contact_tol_m` 放大到 3e-4 也不改变分类。

把间隙设为 0 会让孔壁转成 `active`，方向锥随之塌到 dim=2、主轴精确指向 +Z，
但同时出现 `unknown` 接触面，可装配性判定变为 unresolved（面面精确重合在数值
上是病态的）。

因此本场景保留物理上合理的 0.2 mm 滑配，并且**不依赖方向锥求解器去重新发现
插入轴**：插入方向由场景数据直接给定（`plan_removal(..., directions=±Z)`），
孔的作用是定义这条 mating axis 和插入所需的扫掠体，而不是被反推出来。这与
`asmcore/planning/ports.py` 的分工一致，也顺带绕开了 SOCP/clarabel 依赖。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from asmcore._stl import read_stl_arrays
from asmcore.model import Assembly, MeshData, Part

ASSETS = Path(__file__).resolve().parent / "assets"
BASE_STL = ASSETS / "base_plate.stl"

TABLE = (0.80, 0.60, 0.02)
POST = (0.05, 0.05, 0.15)
BEAM = (0.34, 0.06, 0.03)
LOAD = (0.10, 0.06, 0.06)

BASE_THICK = 0.020
POCKET_DEPTH = 0.012
POCKET_FLOOR_Z = BASE_THICK - POCKET_DEPTH     # 0.008，支柱底面落在这里
POST_SPAN = 0.12                               # 两支柱中心到原点的距离
TABLE_TOP_Z = 0.0

_POST_BOTTOM = POCKET_FLOOR_Z
_BEAM_BOTTOM = _POST_BOTTOM + POST[2]
_LOAD_BOTTOM = _BEAM_BOTTOM + BEAM[2]

#: 可动零件：(part_id, 尺寸, 质量 kg, 中心世界坐标)
MOVABLE = (
    ("post_l", POST, 0.45, (-POST_SPAN, 0.0, _POST_BOTTOM + POST[2] / 2)),
    ("post_r", POST, 0.45, (+POST_SPAN, 0.0, _POST_BOTTOM + POST[2] / 2)),
    ("beam", BEAM, 0.60, (0.0, 0.0, _BEAM_BOTTOM + BEAM[2] / 2)),
    ("load", LOAD, 0.80, (0.0, 0.0, _LOAD_BOTTOM + LOAD[2] / 2)),
)


def box_mesh(sx: float, sy: float, sz: float) -> MeshData:
    """局部原点在中心的轴对齐长方体，外法向。"""
    h = np.array([sx, sy, sz], dtype=float) / 2.0
    v = np.array([[x, y, z] for x in (-h[0], h[0])
                  for y in (-h[1], h[1]) for z in (-h[2], h[2])], dtype=float)
    f = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
        [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
        [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ], dtype=np.int64)
    return MeshData(v, f)


def base_mesh() -> MeshData:
    """读取底座 STL（已是米制，局部坐标即世界坐标）。"""
    if not BASE_STL.exists():
        raise FileNotFoundError(
            f"缺少 {BASE_STL}，先运行 python tasks/make_base_plate.py 生成")
    vertices, faces = read_stl_arrays(BASE_STL)
    return MeshData(vertices, faces)


def pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def build_bridge(*, friction: float = 0.5, with_base: bool = True,
                 scale: float = 1.0) -> Assembly:
    """返回装配完成状态下的 Bridge。

    `with_base=False` 退回无底座的早期版本（支柱直接立在桌面上），保留它是为了
    对照"有无插入约束"对装配方向锥与稳定性判定的影响。

    `scale` 对底座（含盲孔）、四个可动零件及其装配位置统一缩放，桌子保持不变。
    例如缩小 1/3 使用 `scale=2/3`。假设材料密度不变，质量按 scale³ 缩放。
    缩放中心是装配体底面中心；之后由调用者整体平移到桌面上的固定装配位置。
    原始尺寸常量与 STL 不修改，默认 1.0 保留既有调用行为。
    """
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    parts = [
        Part("table", box_mesh(*TABLE), pose(0.0, 0.0, TABLE_TOP_Z - TABLE[2] / 2),
             mass_kg=40.0, com_local_m=np.zeros(3), friction=friction, fixed=True)
    ]
    if with_base:
        parts.append(
            Part("base", base_mesh(), pose(0.0, 0.0, 0.0),
                 mass_kg=6.1, com_local_m=np.array([0.0, 0.0, BASE_THICK / 2]),
                 friction=friction, fixed=True)
        )
    drop = 0.0 if with_base else POCKET_FLOOR_Z   # 无底座时整体下移，支柱坐桌面
    for part_id, size, mass, centre in MOVABLE:
        x, y, z = centre
        parts.append(
            Part(part_id, box_mesh(*size), pose(x, y, z - drop),
                 mass_kg=mass, com_local_m=np.zeros(3), friction=friction)
        )
    if scale != 1.0:
        for i, part in enumerate(parts):
            if part.part_id == "table":
                continue
            tf = part.assembled_tf.copy()
            tf[:3, 3] *= scale
            parts[i] = replace(
                part,
                geometry=replace(
                    part.geometry,
                    vertices=part.geometry.vertices * scale,
                    geometry_error_m=part.geometry.geometry_error_m * scale,
                ),
                assembled_tf=tf,
                mass_kg=None if part.mass_kg is None else part.mass_kg * scale ** 3,
                com_local_m=None if part.com_local_m is None else part.com_local_m * scale,
            )
    return Assembly(tuple(parts))


def movable_ids() -> list[str]:
    return [p for p, _, _, _ in MOVABLE]


def viz_primitives() -> dict:
    """给可视化用的纯几何描述，米。底座单列，因为它不是长方体。"""
    hole = 0.050 + 2 * 0.0002
    return dict(
        boxes=[
            dict(part_id="table", size=list(TABLE),
                 centre=[0.0, 0.0, TABLE_TOP_Z - TABLE[2] / 2], fixed=True),
            *[dict(part_id=p, size=list(s), centre=list(c), fixed=False)
              for p, s, _, c in MOVABLE],
        ],
        plate=dict(
            part_id="base", fixed=True, size=[0.340, 0.340, BASE_THICK],
            centre=[0.0, 0.0, BASE_THICK / 2],
            pockets=[dict(centre=[cx, 0.0], side=hole, depth=POCKET_DEPTH)
                     for cx in (-POST_SPAN, POST_SPAN)],
        ),
    )
