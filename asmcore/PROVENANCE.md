# asmcore 来源与改动记录

搬运日期：2026-09-17
来源：`E:\Projects\WRS2-assembly-planner\wrs\assembly`（WRS 第三代仓库）
对应论文：Chen, Wan, Koyama, Harada, *Planning to Build Block Structures with
Unstable Intermediate States Using Two Manipulators*, IEEE T-ASE, 2021。
`planning/quality.py` 的 docstring 直接引用该文 Eq.(2)。

## 为什么能整包搬

对源包做过导入审计：47 个 `.py` 里只有 6 个出现 `wrs.*`，且多数在 `TYPE_CHECKING`
或函数内延迟导入。`geometry/`、`contact/`、`mechanics/`、`motion/`、`planning/`
**零 `wrs` 导入**，是纯 numpy/scipy 代码。源包 `__init__.py` 本身就是全懒加载
`__getattr__`，且全包没有 PEP-695 等 3.12 专属语法，因此可在 3.11 下运行。

## 改动清单

| 改动 | 原因 |
|---|---|
| 新增 `_stl.py`，`io.read_mesh` 改为 `from ._stl import read_stl_arrays` | 原先延迟导入 `wrs.geom.loader.read_stl_arrays`，是核心里唯一一处非 robotics 层的 wrs 触点。替代实现无外部依赖，ASCII/二进制 STL 均支持 |
| 删除 `robotics/execution.py`（63 KB） | WRS3 运行时胶水（`MJCollider`、`SingleArmManipulation`、`wrs.grasp.*`）。WRS1 下的等价物是 `sealp/primitives/direct_transport.py` |
| `robotics/` 先整包移出 → `reference/wrs3/robotics_wrs3/`，**后已迁回**（2026-10-13 已删除该目录，原件在 WRS2 仓库） | 见下方「G 指标的迁移」 |
| 新增 `planning/ports.py`，`planning/quality_search.py` 的 `TYPE_CHECKING` 导入改指它 | 上一条把 `robotics/` 移走后，`quality_search.py` 仍 `from ..robotics.graspability import ...`，模块能导入（TYPE_CHECKING 守卫）但 `plan_quality_sequence` 无法调用。现改为 asmcore 自己声明 `GraspabilityPort` 结构化协议，由 `adapters/` 实现 |
| 从上条中抽出 `check_force_closure` → `mechanics/force_closure.py` | 6D 正张成 LP，纯数学，是论文二 G（graspability）质量的数学核心，与 WRS3 无关 |
| 删除 `adapters/wrs_scene.py`、`visualization/contact.py` | WRS3 场景对象与渲染 |
| `contact/models.py` 的 `TYPE_CHECKING` 中 `SceneObject` 改为 `Any` | 仅类型注解引用 WRS3 |
| `__init__.py` / `robotics/__init__.py` 相应删除 execution、Graspability* 的导出与懒加载分支 | 与上面的删除保持一致 |

## 刻意**未**改的地方

- **schema 版本字符串保持原样**（`"wrs.assembly/1"`、`"wrs.assembly.sequence/1"` 等）。
  它们是数据格式标识，保留后可直接读取 WRS3 那边
  `examples/assembly/assets/paper2021/assemblies/*.assembly.json` 的论文数据。
- 数值算法、容差默认值、类型注解、缓存与懒加载策略一律未动。

## 依赖

必需：`numpy`、`scipy`。其余都可以通过换配置绕开，但**上游的默认配置绕不开**：

| 包 | 什么时候真的需要 | 怎么绕开 |
|---|---|---|
| `clarabel` | `motion/directions.py` 的 `method="socp"`，**这是 `DirectionConfig` 的默认值** | `DirectionConfig(method="fibonacci")`；或显式传入 mating axis 给 `plan_removal` |
| `torch` + CUDA | `mechanics/_batched_lp.py` 的 `backend="cuda"`，**这是 `StabilitySweepConfig` 的默认值** | `StabilitySweepConfig(backend="numpy")`（或 `"highs"`） |
| `open3d` | `geometry/sdf.py` 的 SDF 接触后端 | 默认的网格后端不需要 |

后两条在上游 `pyproject.toml` 里**完全没有声明**（`torch` 连 optional 都不是）。
直接 `plan_quality_sequence(assembly, analyzer)` 会先后撞上 CUDA 和 SOCP 两堵墙。
纯 CPU 的可用组合见 `tests/test_quality_search_cpu.py::cpu_config`。

## 上游同步

源仓库仍在演进。若要同步上游改动，只需重放本文件的改动清单——它们全部集中在
包边界上，没有侵入任何算法实现。


## G 指标（Graspability Analyzer）的迁移

论文二的 G 质量最初没有 vendored：`robotics/graspability.py` 依赖 `execution.py`
的几何 helpers、`adapters/wrs_scene.py` 的场景转换，以及 WRS3 的
`Grasp` / `MechBase` / `JawView`。现已迁回，判据与数值一律未改：

| 新位置 | 来源 | 改动 |
|---|---|---|
| `robotics/grasp_model.py` | 新写 | 把 `Grasp` + `MechBase/JawView` + `wrs_scene` 收敛成 `GraspRecord` 与 `GripperPort` 一个协议 |
| `robotics/grasp_geometry.py` | `execution.py` 的 `_box_grasps` / `generate_execution_grasps` / `_grasp_contacts` / `_finger_force_contacts` | 逐字照搬；`part_from_scene_object(link)` → `gripper.link_parts()`，`rigid_tf_from_wrs(link.tf)` → `part.assembled_tf` |
| `robotics/graspability.py` | `robotics/graspability.py` | 逐字照搬；力闭合 LP、贴合面豁免、自碰撞/环境碰撞筛选、缓存键、结果字段全部保持 |

`GraspabilityAnalyzer` 满足 `planning/ports.py` 的 `GraspabilityPort`，可直接交给
`plan_quality_sequence`。Panthera 的端口实现在仓库根 `adapters/panthera_gripper.py`
（两指沿 ±Y 平动，开口 0–85 mm，TCP 在基座 +Z 0.17 m）。

`execution.py` 本体（WRS3 的抓取—搬运—插入—交接回放，63 KB）仍未迁移：WRS1 下的
等价物是 `sealp/primitives/direct_transport.py`。
