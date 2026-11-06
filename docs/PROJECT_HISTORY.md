# single-arm-joint-assembly-planner

单臂 Panthera-HT 的**装配顺序 × 初始布局 联合反向搜索**。

把两条既有工作合到一起：

| 来源 | 论文 | 贡献的东西 |
|---|---|---|
| `wrs-sealp-assemble/sealp` | Zhang, Dai, Chen, *Backward Layout Search for Sequence-Constrained Robotic Assembly*, ROBIO 2026 | 反向后缀搜索、四层可行性 oracle、required-motion masks、Hall 匹配剪枝、admissible 下界与 A*/beam |
| `WRS2-assembly-planner/wrs/assembly` | Chen, Wan, Koyama, Harada, *Planning to Build Block Structures with Unstable Intermediate States Using Two Manipulators*, TASE 2021 | 接触图、静力平衡与摩擦锥、扰动扫掠稳定性、装配方向锥、S/G/A 质量、assembly-by-disassembly 序列搜索 |

## 地基决策

**地基 = WRS1**（Panda3D 那一代），已经复制进本仓库根目录，与研究包平级。

WRS 不是 pip 库，是**工作区仓库**：仓库根目录进 `sys.path`，`wrs/` 与你的研究包
是平级的顶层包。`wrs-sealp-assemble` 根目录没有任何打包元数据（`install_package.py`
装的只是 panda3d/numpy/open3d 等第三方依赖），上游 README 的用法是"clone 下来，
用 PyCharm 打开这个文件夹"，sealp 的脚本也都显式做
`sys.path.insert(0, _PROJ_ROOT)` 之后才 `import wrs.basis...`。
`sealp/` 与 `dual_sealp/` 本身就是那个 wrs fork 里的平级目录——本仓库是同一约定的
又一个实例。根目录的 `conftest.py` 负责把仓库根放进 `sys.path`，使
`import wrs` / `import sealp` / `import asmcore` 在 IDE 之外也能工作。

选 WRS1 而非 WRS3 的理由（2026-09-17 审计）：

1. **两个 `wrs` 不能共存。** 两个仓库顶层包名与 distribution name 都是 `wrs`，而且它们不是同一代码的两个版本——WRS2-assembly-planner 自述是**第三代**重写（1 代 Panda3D / 2 代 Pyglet / 3 代 WGPU），API 完全不同，且要求 Python ≥3.12。必须二选一。
2. **Panthera-HT 只存在于 WRS1。** 机械臂（URDF + 网格）、Panthera 夹爪、`PantheraHTSglArm`（含针对腕部/夹爪网格重叠调过的 `setup_cc`）、以及真机控制器 `robot_con/panthera_ht/fafu_robot_controller.py` 全在 WRS1。WRS3 的 manipulators 只有 denso/fanuc/franka/kawasaki/openarm/universal_robots/xarm。`sealp/layout/layout_robot_factory.py` 还已经带好了单臂模式。
3. **论文二的内核几乎不依赖 WRS，可以搬过来。** 见下。反向移植（把 BLS 搬去 WRS3）要重写论文一实验所依赖的全部搜索与可行性代码，复现风险高。

## 仓库结构

全部是平级顶层包，仓库根目录在 `sys.path` 上：

```
wrs/        地基。从 wrs-sealp-assemble 复制, 含 Panthera-HT 与真机控制器
sealp/      论文一的布局能力。从 wrs-sealp-assemble 复制
asmcore/    论文二的科学内核。vendored, 零 wrs 依赖, 只需 numpy + scipy
jbls/       顺序 × 布局 联合反向搜索：欧氏距离基线与 Bridge 局部可行性筛选
lattice_joint/  主线（2026-11-02 起）：已装集合网上的精确联合搜索（顺序 × 布局），见 lattice_joint/README.md
joint_search/   旧主线：反向 beam 搜索（顺序 × 布局 × 抓取）+ IK 目录 + cuRobo 验证，保留
adapters/   唯一与地基耦合的一层: Panthera-HT / 抓取 / 运动
conftest.py 把仓库根放进 sys.path
```

### WRS1 与 SEALP 来源

当前 `wrs/` 与 `sealp/` 已从以下工作树复制：

- 源仓库：`E:\Projects\wrs-sealp-assemble`
- 分支：`sealp`
- 基准提交：`057d43eb2b66db27017f26a9bf83e75b1647849d`
- 复制日期：2026-09-17

详细版本和本地改动说明见 `WRS_FOUNDATION.md`。如需重新同步，可执行：

```powershell
cd E:\Projects\single-arm-joint-assembly-planner
robocopy E:\Projects\wrs-sealp-assemble\wrs   .\wrs   /E /XD __pycache__ /XF *.pyc
robocopy E:\Projects\wrs-sealp-assemble\sealp .\sealp /E /XD __pycache__ /XF *.pyc
```

robocopy 成功复制文件后返回退出码 1，不是报错。
`wrs/robot_sim/manipulators/panthera_ht.zip`（36 MB）是网格备份，可加 `/XF *.zip`
跳过。其余机械臂（ur3/xarm7/franka…）本项目用不到，但先整包复制、确认能跑之后
再裁剪，避免 `__init__` 连带导入失败。

依赖：`python wrs\install_package.py` 装 WRS1 的第三方依赖，`asmcore` 另需 numpy + scipy
（已在依赖表内）。

**不变量：`asmcore` 不 import 任何 `wrs.*`，`adapters/` 是唯一耦合面。** 这样将来若要换到 WRS3，代价只是重写 `adapters/` 的三个文件，而不是重写算法。该不变量由 `tests/test_asmcore_standalone.py` 锁死。

## 已验证

纯数学层在 Python 3.11 + numpy + scipy、无 WRS/clarabel 导入的条件下通过：

- 对 `asmcore` 全部 42 个模块逐一 import，**零 `wrs.*` 被拉入**；导入期也不拉 clarabel / open3d / mujoco / panda3d。
- 接触分析 + 静力平衡在 table + 两柱 + 横梁场景上正确工作：完整结构与双柱状态 `feasible`；横梁悬臂在单柱上的状态 `infeasible`，即论文二的 **S=0 不稳定中间状态**。
- **单臂 S=0 硬剪枝闸门**按既定方案工作：`post_a → post_b → lintel` 可行；任何"横梁先于第二根柱子"的顺序在第 2 步被剪枝。
- `plan_sequence` 在 0 辅助资源（单臂、无第二只手）下搜到全程稳定的顺序，且横梁最后装。
- 显式传入 mating axis 时 `plan_removal` 成功且**不触发 SOCP**——联合搜索的插入方向来自 asmdef，因此 `clarabel` 不是规划回路的必需依赖。

`tests/test_panthera_foundation.py` 另外验证：

- 导入的 `wrs` 确实来自本仓库，而不是环境里另一份同名包；
- `PantheraHTSglArm` 为 6 自由度，碰撞模块能够初始化；
- home TCP 的 FK → IK 能得到六关节解，且回到该解时不误报碰撞；
- 论文二 G 指标依赖的 Force Closure 参考算例能够运行。

## 单臂方案要点

论文二的 Support Analyzer 前提是双臂：`sequence_quality()` 里 `min(G)·min(A)/(λ·n)` 分支中的 `n` 数的是需要第二只手托住的状态数。**本项目采用硬剪枝**：单臂没有第二只手，任一中间状态 `S=0` 即判该装配顺序不可行，λ 分支不参与排序。

好处是两篇论文的形式化**直接对齐**了：论文一把可行性当硬约束、目标函数只含运动代价 `J_nom`；稳定性于是成为步可行性 `φ_k` 的一个新合取项：

```
φ_k = 稳定性 ∧ 抓取可行 ∧ IK ∧ 碰撞 ∧ 运动
```

`asmcore/mechanics/equilibrium.py` 的 `find_support_requirements` 仍保留，但只用作**剪枝原因诊断**（"这个顺序被剪掉是因为第 k 步的哪个零件失去支撑"），不进入搜索。

## 当前完成度与待办

- [x] Panthera-HT 单臂提供器与 Panthera 真实夹爪端口
- [x] S/G/A 正向与反向顺序搜索的 Bridge 验证例
- [x] cuRobo 的 Panthera 运动学、碰撞球、批量 IK、multi-env 规划和携物附着实验桥接
- [x] `jbls/` 欧氏距离基线：反向同时选择零件和初始位姿，S/G/A、共同抓取与局部动作筛选，输出排名和布局网页。见 [运行说明](jbls/README.md)。尚未验证完整自由搬运轨迹，不能直接执行。
- [ ] `GraspFeasibility` 的“取放两端共同抓取 + IK”正式适配器；目前相关逻辑仍散在示例中
- [ ] `MotionWitness` 的正式 cuRobo 实现；当前 `curobo_bridge/planner.py` 还是实验性分级规划器
- [ ] `jbls/`：把 `sealp/examples/layout/bsfs` 提升为正式包，加入顺序层和布局 cost API
- [ ] asmdef ↔ `wrs.assembly/1` JSON 的转换（schema 字符串已保留，可直接读 WRS3 的 paper2021 数据）
- [ ] 理论：π 成为搜索变量后，重新叙述论文一的后缀保持性（extension invariance）
