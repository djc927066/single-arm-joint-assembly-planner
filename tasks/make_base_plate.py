"""生成 Bridge 基准场景的底座 STL：方形板 + 两个方形盲孔。

底座在联合搜索中是**固定件**：位置由人给定、不参与布局优化，也不进入装配顺序
（asmcore 里 `fixed=True`）。它对应论文一 BLS 里的 preassembled 基座
（`sealp/examples/layout/bsfs` 的 `preassembled_pid`）。

两个盲孔正好容纳两根 post，单边留 0.2 mm 间隙形成滑配。孔不透底，孔底承载
post 的自重，孔壁在 post 倾覆时提供侧向约束。插入方向因此被限定为 −Z 单轴，
这正是装配方向锥分析（论文二 Assemblability Analyzer）想看到的强约束。

输出单位为**米**，与本项目及 WRS1 的场景约定一致。

    俯视                        剖面 A-A
    ┌─────────────────┐        ┌──┬────┬──┬────┬──┐   z=0.020
    │  ┌──┐     ┌──┐  │        │  │    │  │    │  │
    │  │  │     │  │  │        │  └────┘  └────┘  │   z=0.008  孔底
    │  └──┘     └──┘  │        │                  │
    └─────────────────┘        └──────────────────┘   z=0
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

PLATE = 0.340          # 正方形边长 (m)
THICK = 0.020          # 板厚 (m)
POCKET_DEPTH = 0.012   # 盲孔深度 (m)
POST_SIDE = 0.050      # post 截面边长 (m)
CLEARANCE = 0.0002     # 单边间隙 (m)，滑配
HOLE_SIDE = POST_SIDE + 2 * CLEARANCE
HOLE_CENTRES = ((-0.12, 0.0), (0.12, 0.0))   # 与 tasks.bridge 的 post 位置一致


class MeshBuilder:
    """收集三角面片，按期望法向自动修正绕序。"""

    def __init__(self) -> None:
        self.tris: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def quad(self, p0, p1, p2, p3, normal) -> None:
        """四点共面矩形，`normal` 是期望的外法向（指离固体材料的一侧）。"""
        pts = [np.asarray(p, dtype=float) for p in (p0, p1, p2, p3)]
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        if float(np.dot(n, np.asarray(normal, dtype=float))) < 0:
            pts = pts[::-1]
        self.tris.append((pts[0], pts[1], pts[2]))
        self.tris.append((pts[0], pts[2], pts[3]))

    def rect_z(self, x0, x1, y0, y1, z, normal_z) -> None:
        self.quad((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z), (0, 0, normal_z))

    def rect_x(self, x, y0, y1, z0, z1, normal_x) -> None:
        self.quad((x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z1), (normal_x, 0, 0))

    def rect_y(self, y, x0, x1, z0, z1, normal_y) -> None:
        self.quad((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1), (0, normal_y, 0))

    def arrays(self):
        """去重顶点，返回 (vertices, faces)。"""
        flat = np.asarray([p for tri in self.tris for p in tri], dtype=float)
        keyed = np.round(flat, 9)
        verts, inverse = np.unique(keyed, axis=0, return_inverse=True)
        return verts, inverse.reshape(-1, 3).astype(np.int64)


def build() -> MeshBuilder:
    """在统一的 x/y 断点网格上铺砌所有水平面与外侧壁。

    每个面都按同一组断点切分，相邻面的边才能一一配对；否则顶面切块处会与
    整块的底面、外侧壁形成 T 型接缝——几何上闭合、拓扑上非流形，多数 CAD
    与网格工具会判为坏件。
    """
    m = MeshBuilder()
    h = PLATE / 2.0
    zt, zb = THICK, 0.0
    zf = THICK - POCKET_DEPTH          # 孔底高度
    hs = HOLE_SIDE / 2.0

    holes = sorted((cx - hs, cx + hs, cy - hs, cy + hs) for cx, cy in HOLE_CENTRES)
    (lx0, lx1, ly0, ly1), (rx0, rx1, ry0, ry1) = holes
    assert (ly0, ly1) == (ry0, ry1), "本分解假设两孔 y 范围相同"

    xs = [-h, lx0, lx1, rx0, rx1, h]
    ys = [-h, ly0, ly1, h]
    cells = [(xs[i], xs[i + 1], ys[j], ys[j + 1])
             for i in range(len(xs) - 1) for j in range(len(ys) - 1)]
    is_hole = lambda c: any(abs(c[0] - k[0]) < 1e-12 and abs(c[2] - k[2]) < 1e-12
                            for k in holes)

    # 底面与顶面：同一网格；顶面跳过两个孔所在的格子
    for cell in cells:
        x0, x1, y0, y1 = cell
        if x1 - x0 < 1e-12 or y1 - y0 < 1e-12:
            continue
        m.rect_z(x0, x1, y0, y1, zb, -1)
        if not is_hole(cell):
            m.rect_z(x0, x1, y0, y1, zt, +1)

    # 外侧壁：±X 壁按 y 断点切，±Y 壁按 x 断点切
    for y0, y1 in zip(ys, ys[1:]):
        m.rect_x(-h, y0, y1, zb, zt, -1)
        m.rect_x(+h, y0, y1, zb, zt, +1)
    for x0, x1 in zip(xs, xs[1:]):
        m.rect_y(-h, x0, x1, zb, zt, -1)
        m.rect_y(+h, x0, x1, zb, zt, +1)

    # 每个盲孔：4 个孔壁（外法向指向孔内空腔）+ 1 个孔底
    for x0, x1, y0, y1 in holes:
        m.rect_x(x0, y0, y1, zf, zt, +1)   # 左壁，材料在 x<x0
        m.rect_x(x1, y0, y1, zf, zt, -1)   # 右壁
        m.rect_y(y0, x0, x1, zf, zt, +1)   # 前壁
        m.rect_y(y1, x0, x1, zf, zt, -1)   # 后壁
        m.rect_z(x0, x1, y0, y1, zf, +1)   # 孔底

    return m


def check(verts: np.ndarray, faces: np.ndarray) -> dict:
    """水密性与体积校验：每条边必须恰好被两个方向相反的半边使用。"""
    half = {}
    for tri in faces:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            half[(int(a), int(b))] = half.get((int(a), int(b)), 0) + 1
    unmatched = [e for e in half if half[e] != 1 or half.get((e[1], e[0]), 0) != 1]
    # 散度定理求有符号体积
    p = verts[faces]
    volume = float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)
    expected = PLATE * PLATE * THICK - len(HOLE_CENTRES) * HOLE_SIDE ** 2 * POCKET_DEPTH
    return dict(
        vertices=len(verts), faces=len(faces),
        watertight=not unmatched, bad_edges=len(unmatched),
        volume_m3=volume, expected_m3=expected,
        volume_ok=abs(volume - expected) < 1e-12,
    )


def write_binary_stl(verts: np.ndarray, faces: np.ndarray, path: Path, title: str) -> None:
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, norms, out=np.zeros_like(n), where=norms > 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(title.encode("ascii", "replace")[:80].ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(faces)))
        for i in range(len(faces)):
            fh.write(struct.pack("<12fH", *n[i], *tri[i, 0], *tri[i, 1], *tri[i, 2], 0))


def main() -> None:
    mesh = build()
    verts, faces = mesh.arrays()
    report = check(verts, faces)
    out = Path(__file__).resolve().parent / "assets" / "base_plate.stl"
    write_binary_stl(verts, faces, out, "bridge base plate, metres, two blind pockets")
    for key, value in report.items():
        print(f"  {key:<12} {value}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
