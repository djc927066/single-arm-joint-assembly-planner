# curobo_bridge —— 带碰撞的 Panthera cuRobo 模型与 WRS 对照

cuRobo V2（commit `8e734f3`，即官方当前版本；之后上游只有一次 yaml safeloader 修改）
与本仓库 WRS1 Panthera-HT 的桥接层。算法包（`asmcore`、`jbls`）不 import 本包。

## 运行环境

只在 cuRobo 独立 venv 中运行：

```powershell
$py = 'E:\Projects\wrs-sealp-assemble\dual_sealp\cuRobotest\.venv\Scripts\python.exe'
cd E:\Projects\single-arm-joint-assembly-planner
& $py -X utf8 -m curobo_bridge.build_spheres --densities 0.5 1 2   # 生成球模型（约 2.5 min/密度）
& $py -X utf8 -m curobo_bridge.bench_collision --scenes 6            # 碰撞：误报/漏报/耗时
& $py -X utf8 -m curobo_bridge.bench_motion --scenes 4 --tasks 8     # 规划：cuRobo vs WRS RRT
& $py -X utf8 -m curobo_bridge.bench_self_collision --states 4000    # 自碰撞补偿选值
& $py -X utf8 -m curobo_bridge.bench_hybrid --scenes 4 --tasks 8     # 分级规划器端到端
& $py -X utf8 -m pytest tests/test_curobo_bridge.py -q
```

该 venv 以 `--system-site-packages` 继承 WRS 依赖，**同一进程可同时加载 WRS 与 cuRobo**
（`runtime.prepare()` 保证先 import torch）。没有安装/升级任何包：`RobotBuilder` 顶层
`import viser`，而 viser 要求 trimesh<5 与 WRS 冲突，所以 `runtime` 注入一个占位 `viser`
模块，仅 `visualize()` 不可用。

## 流程

1. **`export_model.py`**：从 WRS 实际模型导出 URDF + 每个碰撞部件网格。顶点直接用 WRS
   `cmodel.homomat` 变换到所属关节坐标系，不重写运动学。随机 32 组关节角实测：
   网格三角形顶点误差 5.8e-9 m（STL float32），TCP 误差 2.5e-16 m。
   `panthera_wrsext` = WRS 外部碰撞集合（link2–5、掌部、两指）；`panthera_full` 另含
   base_link、link1。
2. **保守包络**：Panthera CAD 网格多为非封闭多壳体（link2 有 32 万面），MorphIt 与其
   质量指标依赖内外判定，直接用会耗尽内存（link2 需 5.6 GB）。每个部件做 1.5 mm
   体素化 → 填洞 → 膨胀 1 体素 → marching cubes → 简化到 6000 面：封闭、包含全部原始
   表面采样点、外扩 1.5–3 mm（与 WRS 图元的 2 mm 余量相当）。
3. **`build_spheres.py`**：`RobotBuilder` + MorphIt 拟合包络。密度 0.5/1/2 → 52/58/67 个球
   （原主线的 AABB 外接球覆盖是 1127 个）。
4. **`curobo_models.py`**：装配运行配置，含两处修正（见下）。

## 两处必须知道的修正

* **HOME 自碰撞**：sealp 所有脚本的 HOME 是全零，即关节下限处完全折叠的姿态，
  此时 link2/link5 真实网格间隙约 0.1 mm。不忽略这对 link，HOME 就被判自碰撞、所有从
  HOME 出发的规划直接失败。已加入 `self_collision_ignore`。WRS 不登记任何自碰撞对，
  这里仍比 WRS 严格（link2 与 link4、夹爪等仍检查）。
* **cuRobo 上游缺陷**：`KinematicsLoader._build_collision_model` 本意是让
  `collision_sphere_buffer` 不影响自碰撞，它把补偿写进了局部变量，却把未补偿的
  `self.self_collision_buffer` 传给 `SelfCollisionKinematicsCfg`。结果场景缓冲也膨胀
  自碰撞：2 mm 缓冲就让 HOME 下相距 10.4 mm 的 link2/link4 "相碰"。`with_buffer()`
  在配置中显式写入每个 link 的 `-buffer` 自碰撞补偿，未改官方源码。

## 碰撞保真度（`bench_collision`，6 个场景 4800 个状态）

两个参照：`wrs_prim` = WRS/RRT 当前实际语义（部件局部 AABB+2 mm vs 障碍局部 AABB+2 mm。
注意 sealp `make_collision_model` 传的 `cdprim_type="triangles"` 在 WRS1 里不是合法枚举，
**静默回退为 AABB**）；`wrs_mesh` = ODE 三角网格求交（几何真值）。
状态 = 每场景 300 均匀 + 500 边界（表面距离 < 3 mm 的难例）。

| 模型 | 球数 | vs 网格 FN | vs 网格 FP（全部 / 均匀） |
|---|---:|---:|---:|
| WRS 图元（现状） | — | 0 | 669 = 28.4% / 6.6% |
| AABB 外接球 20 mm（原主线） | 1127 | 0 | 718 = 30.5% / 7.4% |
| MorphIt d1 | 58 | 3 | 245 = 10.4% / 2.5% |
| MorphIt d2 | 67 | 0 | 222 = 9.4% / 2.5% |
| **MorphIt d1 + 2 mm** | 58 | **0** | 315 = 13.4% / 3.1% |
| MorphIt d2 + 2 mm | 67 | 0 | 284 = 12.1% / 2.9% |

MorphIt 无缓冲时的 3 个漏报都是擦边接触（点采样距离 0.3–4.5 mm），加 2 mm 缓冲全部消失。
推荐运行配置：**MorphIt d1 + 2 mm**——对几何真值零漏报，误报约为 WRS 现状的一半。

耗时（常驻 GPU，同进程无 RPC）：WRS 图元 0.42 ms/状态、WRS 网格 0.66 ms/状态；
cuRobo MorphIt 2048 个状态 1.73 ms（0.8 µs/状态，约 500 倍），单状态 1.9 ms（比 WRS 慢）；
AABB 外接球 2048 个状态 8.4 ms（球多 20 倍，慢约 5 倍）。检查器构建 ≈0.04 s（首次含编译 0.5 s）。

## 自碰撞（`bench_self_collision`，4000 个随机状态，真值 = ODE 逐对网格求交）

WRS 不做任何自碰撞检查。cuRobo 检查 6 对未忽略 link（link2–link4、link2/3/4–夹爪等），
真实网格自碰撞 252 例：

| 自碰撞补偿 | FN | FP |
|---|---:|---:|
| 0 mm（默认，`curobo` 级） | 0 | 13.8% |
| −0.5 mm | 0 | 11.5% |
| −2 mm（`curobo_relaxed` 级） | 8 | 5.5% |
| −4 mm | 80 | 0.9% |

包络外扩在两侧叠加，自碰撞误报偏高，会让 TrajOpt 卡住（下面 2 个失败任务的输出轨迹
场景无碰撞，但有 37 个状态被判自碰撞，网格复核全部是假阳性）。所以放宽只用于第二级，
并且必须经过复核。

## 运动规划（`bench_motion` / `bench_hybrid`，4 场景 × 8 任务，24 个需绕行）

所有路径按 0.01 rad 加密后用 WRS ODE 网格逐状态复核。

| 规划器 | 成功 | 真实网格碰撞的路径 | 违反 WRS 图元语义的路径 | 规划时间 中位 / 合计 | 速度界运动时间中位 |
|---|---:|---:|---:|---:|---:|
| WRS RRT ext 0.3（sealp L3） | 32 | **2** | 14 | 2.17 s / 75.8 s | 0.634 s |
| WRS RRT ext 0.1 | 32 | 0 | 20 | 6.58 s / 234 s | 0.609 s |
| cuRobo MotionPlanner | 30 | 0 | — | 0.45 s / 16.9 s | 0.558 s |
| cuRobo Batch（每场景一批） | 30 | 0 | — | 0.13 s / 4.8 s | 0.558 s |
| cuRobo Batch multi_env（32 个世界一次） | 29 | 0 | — | 0.15 s / 4.9 s | 0.573 s |
| **分级 `PantheraPlanner` 单个** | **32** | 0 | — | 0.44 s / 21.5 s | 0.558 s |
| **分级 `plan_batch`（multi_env + 兜底）** | **32** | 0 | — | 0.15 s / 8.8 s | 0.558 s |

* RRT 只在 ext_dist 采样点上检测：ext 0.3 有 2 条路径真的擦碰网格，且 14 条在段内违反
  WRS 自身的图元语义。cuRobo 路径贴近障碍走，WRS 图元（误报 28%）会判碰，但网格真值无碰撞，
  因此该列对 cuRobo 不适用。
* 分级规划：30 个由 cuRobo 直接解出，1 个由放宽自碰撞解出，1 个 RRT 兜底；复核拒绝 0 条。
  复核耗时合计 3.5 s（约 0.11 s/路径）。
* 同一任务配对（30 个），cuRobo 轨迹的速度界时间是 RRT 的 0.93 倍，关节路径长度中位 3.27 vs 3.78 rad。
* 一次性开销另计：MotionPlanner 构建+预热 3–5 s；32 环境 multi_env 构建+预热 18.8 s。

失败分析：2 个纯 cuRobo 失败任务里，RRT 找到的路径在 cuRobo 自己的模型下完全无碰撞，
PRM 图搜索也能连通，失败在 TrajOpt；增加重试到 30 次（9 s）仍失败。

## 尚未接入

* `AttachmentManager.attach`（携物球拟合）与 `plan_grasp` / `ToolPoseCriteria.linear_motion`
  （取放直线段）：接口已确认可用，但尚无基准，下一步接到反向搜索的搬运段 cost。
* 夹爪按导出时的张开宽度（0.085 m）固定；携物时应按抓取宽度重新导出或锁定手指关节。
