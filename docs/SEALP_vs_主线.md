# SEALP（BLS 原版）与当前主线（joint_search）：按执行顺序对照

> 写于 2026-10-23。只是说明，没有改任何代码。
> SEALP 代码目录：`E:\Projects\wrs-sealp-assemble\sealp\examples\layout\`（下文简写为 `layout/`）
> 主线代码目录：`E:\Projects\single-arm-joint-assembly-planner\`
> 例子统一用三角支撑框架 `triangular_frame6`：6 个零件（底座 + 3 根立柱 post_A/B/C + 横梁），立柱 58×62×100 mm。

---

## 0. 一句话区别

| | SEALP / BLS | 主线 |
|---|---|---|
| 搜什么 | **顺序固定**，只搜布局（每个零件放哪、什么姿态） | **顺序和布局一起搜**（每层还要选"这一步拆哪个零件"） |
| 搜索方向 | 反向：从最后装的零件往前定位置 | 反向：从最后装的零件往前拆，同时定顺序、位置、抓取 |
| 每步可行性 | WRS 网格模型：IK + 网格碰撞（L0/L1/L1.5/L2） | GPU 球体碰撞（cuRobo 球，67 球）+ IK 缓存 + 直线段扫描 |
| cost | 物体原点的**欧氏折线长度**（米） | **关节时间下界**（秒，各关节梯形速度，异步） |
| 最后验证 | WRS `evaluate_layout`（common grasp，非 RRT），可选 L3 RRT | 全流程验证（2026-10-25 起）：每段自由运动都用 cuRobo 规划 + 整个执行过程（含固定段）WRS 网格碰撞检测；原 lazy confirm + witness 已合并删除 |

---

## 1. 入口和输入

### SEALP
入口 `layout/bsfs/run.py: main()`（约 743 行），一共 9 个阶段（文件头 15–23 行有列表）：

```
[1/9] 加载 任务/机器人/CAD/抓取/稳定姿态/cost
[2/9] 生成粗网格装配中心      [3/9] 便宜的确定性中心筛选      [4/9] 细化中心
[5/9] 每个中心跑 BSFS（beam 或 exact）
[6/9] 全局取最好            [7/9] 最终 yaw 细化
[8/9] WRS evaluate_layout witness（带种子重试）   [9/9] 保存
```

主要参数（`run.py` 106–184 行）：

| 参数 | 默认值 | 意义 |
|---|---|---|
| `--part-order` | 空 → 用 asmdef 里的顺序 | **顺序是输入，不搜** |
| `--goal-pos` | `0.373,0,0` | 装配中心 |
| centers 模式 | `single` | 默认只用一个中心，不搜中心 |
| `--mode` | `beam` | 也可以选 `exact`（A*） |
| `--beam-width` | 8 | 每层保留 B=8 个节点 |
| `--cand-per-part` | 12 | 每个零件最多 12 个位置 |
| `--poses-per-xy` | 1 | 每个位置只要第一个可行姿态 |
| `--grid-spacing` | 0.05 | 候选网格间距 5 cm |
| `--lift` | 0.10 | cost 折线的抬升高度 |
| `--tau-clear` / `--tau-manip` | 0.005 / 1e-3 | 间隙、可操作度阈值 |
| `--req-motion-samples` / `--req-motion-release-radius` | 5 / 0.015 | 运动走廊采样 |
| `--yaw-step-deg` | 20 | yaw 细化步长 |
| `--witness-retries` | 5 | witness 失败重试次数 |

加载用 `infer_assembly_ga.py: _build_searcher`（约 200 行起）：asmdef、config、grasp 文件夹、`table_margin=0.06`、`preassemble_first_part=True`（第一个零件直接放在装配位，不参与布局）。

### 主线
入口 `joint_search/run_frame.py: main()`（17 行）。

```
python -m joint_search.run_frame --model triangular_frame6 --B 10 --M 5 --N 5 --penalty --detax
       --grasps-per-pose 8 --sampler fps --placements sealp --max-placements 3
```

| 参数 | 默认 | 意义 |
|---|---|---|
| `--model` | triangular_frame6 | 场景 |
| `--B` | 10 | 每层保留 10 个节点 |
| `--M` | 5 | 每个"被拆零件"最多保留 5 个子节点（每个位置最多 `per_x=2` 个抓取） |
| `--N` | 5 | 最后去重后送验证的计划数 |
| `--grasps-per-pose` | 8 | 每个位置最多筛 8 个抓取（按下界排序，失败就补） |
| `--sampler` | fps | 候选位置用最远点采样 |
| `--placements` | sealp | 姿态候选用 SEALP 同一套函数 |
| `--penalty` / `--detax` | 开 | 估计值里加绕路惩罚 |

`Problem(...)`（`joint_search/problem.py:78`）按顺序做：
1. `load_scene` → 场景、零件网格、目标位姿、抓取、候选位置；
2. `GpuChecker` → GPU 球碰撞器；
3. `screen.load_or_build` → IK 目录（缓存 pkl，v4）；
4. `_drop_home_colliding`（134 行）→ 删掉在 HOME 姿态下和机械臂相撞的候选；
5. 修剪候选（sealp 模式：每个位置只留 SEALP 顺序下第一个有可用抓取的姿态，取前 n_cand 个位置）；
6. 固定段的表格 fix_pick / fix_goal、每个零件的下界 `_bounds`（158 行）、前驱 `preds`（184 行）。

**区别**：SEALP 的顺序是输入；主线的顺序是输出。SEALP 第一个零件预装、不搜；主线的底座是已经装好的世界。

---

## 2. 候选生成

### 2.1 姿态（放在桌上是哪一面朝下）

| | SEALP | 主线（`--placements sealp`，默认） |
|---|---|---|
| 函数 | `find_optimal_initial_layout_tower_strict_pycharm.py` 里的 `_stable_pose_candidates_from_flatsurface` + `_fallback_rot_candidates_with_names` + `_dedupe_pose_rot_candidates` | `lazy_cost/scene.py:123 sealp_rot_candidates`，**直接 import 调用上面同一组函数** |
| 稳定姿态 | WRS `FSReferencePoses`，阈值 0.10 | 同一个 |
| 补充 | 90° 翻转的 fallback 姿态 | 同一个 |
| 上限 | `MAX_ROT_CANDIDATES = 12` | 12 |
| 排序 | 按 tag 排（identity 在前） | 一样 |

已知问题：WRS `FSReferencePoses` 在盒子类零件上会漏面（质心射线打到三角面对角线上，ODE 报告没命中，6 面只剩 2 面）。SEALP 靠 90° fallback 补回来。主线另外有自己实现的 `stable_placements`（`scene.py:76`，`--placements stable`），不漏面，但默认没用它，因为你要求"跟原 SEALP 调用同一个函数"。

三角框架的实际结果：立柱在 40 个位置上全是 identity；横梁大多 identity，少数竖放。最后选中的计划全用 identity。

### 2.2 位置（x, y）

| | SEALP | 主线 |
|---|---|---|
| 网格 | `infer_assembly_ga.py:282 _part_anchor_grid`：边距 = min(footprint)/2 + 2 cm，间距 5 cm（最小 3 cm），linspace | `scene.py:203 _finish`：桌面上随机/网格撒点（pool = 2×n_cand） |
| 每个点的过滤 | `_enumerate_xy_candidates`（313 行）：至少有一个姿态满足 ① 机械臂禁区矩形 `_staging_arm_keepout_reason`（1625）② 和预装件的网格间隙 `_mesh_clearance_reason`（1456）/ `_pairwise_collision`（1880）③ HOME 碰撞 `_robot_home_collision_reason`（1166） | 生成时只检查桌面范围和姿态能放下；HOME 碰撞在 `Problem._drop_home_colliding` 里做（三角框架上删掉 0 个）；IK 不可解的点在 IK 目录里自然没有 |
| 排序 | `_farthest_point_order`（356 行）：种子 = 离质心最近的点，然后最远点 | `scene.py:176 fps_order`：sealp 模式只对 xy 做最远点 |
| 数量 | 12 | n_cand（三角框架 40） |
| 每个位置几个姿态 | 1（`poses_per_xy=1`，按顺序第一个可行的） | 1（`problem.py` 修剪：第一个有可用抓取的） |

**区别**：
- SEALP 在**搜索中**才判断"这个姿态可不可行"（certify 里），所以"第一个可行姿态"会依赖已放的零件；主线是**预处理时**按"有没有可用抓取"选定，跟搜索状态无关。
- SEALP 12 个候选，主线 40 个。主线的候选多，因为顺序也要搜，每层有好几个零件可以选。

### 2.3 抓取
- SEALP：每个零件一个 `GraspCollection`（WRS 文件），在 certify 里用 `reason_common_gids` 实时求 IK 和碰撞。
- 主线：`screen.py:110 build` 预先对每个（位置, 抓取）解 IK，存在 pkl 里（v4，夹爪闭合，hover 抬到零件顶上方 2 cm，`_rise` 59 行）。搜索时直接查表。**每个位姿只存一个 IK 解**（已知不足）。

---

## 3. 搜索主循环

### SEALP：`bsfs/search.py`（`search_site` → `_search_beam`）

```
节点 = 已经定好位置的零件后缀（从最后装的开始）
for k = n .. 2   （顺序固定，第 k 个零件就是 part_order[k]）
    for 每个节点:
        for 每个 xy（12 个，跳过离后缀里已有 xy 小于 2 cm 的）:
            for 每个姿态（_iter_rot，按顺序）:
                如果 (零件, xy, rot) 在 nogood 里 → 跳过
                res = oracle.certify(...)          ← 第 4 节
                可行 → 生成子节点，停止看这个 xy 的其它姿态（poses_per_xy=1）
    所有子节点按 lex_key(cost, -clearance, -manip) 排序，取前 B=8
```
- 一层只有一个零件，分支 = 节点数 × 位置数。
- `HARD_FAIL` 的原因（`oracle.py:31`）记进 nogood，后面直接跳过。
- `exact` 模式：A*，离散网格 + 占用位图 + 约束传播 + Hall 匹配，h = 每个剩余零件到目标的最近格距离之和。

### 主线：`joint_search/search.py:49 beam_search`

```
节点 Node = (R: 还没拆的零件集合, suffix: 已决定的后缀步骤, table: 桌上的零件, g_lb, g_est, pend, f)
初始: R = 全部零件（除底座），suffix 为空
for 层 = 1 .. n:
    for 每个节点:
        for 每个 p ∈ R 且 step_ok(R, p)（静力稳定 LP + 插入扫掠）:
            children = expand(pb, node, p)      ← 第 4、5 节
            每个位置最多 per_x=2 个，这个 p 最多保留 M=5 个
    合并同 key 的节点（同 R、同头部位姿/抓取），按 f = g_est + h(R) 排序，取前 B=10
最后一层：所有完整计划，按 (顺序 + 布局) 去重，取前 N=5 送验证
```
- 一层有多个零件可选：分支 = 节点数 × |可拆零件| × 位置 × 抓取。
- 有可采纳启发 h(R)（`problem.py:176`），每个剩余零件自己的最小代价下界之和。
- 有 `pend`：反向搜索时，当前零件之前的那段空载移动的起点还不知道，用"所有可能前驱"的最小值作下界。

**区别**：
1. SEALP 的子节点里没有"抓取"这一维（certify 里只要存在 common gid 就行）；主线的状态里有具体的抓取 g，因为关节时间取决于抓取。
2. SEALP 按 `cost` 字典序排；主线按 `f = g + h` 排（有 h，有 pend）。
3. SEALP 有 nogood 缓存；主线没有（已知可加）。

---

## 4. 每一步的可行性检查

### SEALP：`bsfs/oracle.py:91 certify`（共 91–220 行）

按顺序，任一步失败就返回：

| 层 | 检查什么 | 用的函数 | 物体 |
|---|---|---|---|
| L0 | 机械臂禁区、是否正放、与其它已放零件的两两碰撞、网格间隙、HOME 碰撞/间隙 | `_staging_arm_keepout_reason` `_pairwise_collision` `_mesh_clearance_reason` `_robot_home_collision_reason` | 网格 |
| req masks | **当前零件自己**的三段局部运动（拾取后 +Z 抬 5 cm、插入 −Z 5 cm、松手后空手退出：半径 1.5 cm 胶囊）扫过的体积，会不会碰到**比它晚装、此刻还在桌上**的零件；每段 5 个采样 | `bsfs/req_motion_masks.py` | 网格 |
| L1 | **common grasp**：同一个抓取在 [桌上位姿, 装配位姿] 两处都 IK 可解且无碰撞 | `wrs/manipulation/pick_place.py:19 reason_common_gids` | 障碍 = 环境 + 已装零件（装配位）+ 其它零件（桌面位） |
| L1.5 | 拾取后沿 +Z 抬起一段，连续 IK 都可解（只是 IK，不查碰撞） | `_l2_pick_depart_motion_gids`（2345） | — |
| L2（第 2 层以后） | pre-pick / pick / post-pick 三个点，5 个方向（z, ±x, ±y）任一方向通过即可 | `_l2_pick_quick_check_gids`（2247） | 网格 |
| 阈值 | 最小间隙 ≥ 5 mm，可操作度 ≥ 1e-3 | `cost.py: passes_thresholds` | |

**注意**：SEALP 只查**离散的几个点**（端点 + 抬起段的 IK），**不查**中间的转运路径、空载路径是否碰撞。中间路径要到 L3（可选 RRT）才查。

### 主线：`joint_search/expand.py:85 expand`

**第 1 部分：每个抓取的装配侧**（约 105 行）。世界 A = 已装零件（R 外）+ 桌面；世界 B = 已装零件含 p：
- ins（装配位，手臂 + 闭合夹爪）
- preins（装配位上方，带着零件）
- preins → ins 直线（手臂）
- ins + SKIP(0.25) → retreat 直线（张开，不含 p）
- retreat 姿态（世界 B）

**第 2 部分：每个（位置 x, 抓取 g）的拾取侧**（约 117 行）。先 `_bound_order`（64 行）按下界给抓取排序，每个位置先筛 8 个，全失败再补：
- pick（闭合）
- hover（张开，p 作为障碍物）
- hover → pick+SKIP 下降直线（不含 p）
- lift（带零件，闭合）
- pick+SKIP → lift 直线（带零件）

全部用 GPU 球体碰撞批量查（`GpuChecker.batch_free`，67 球，1 mm buffer）。

**区别**：
1. SEALP 用 req masks 查"当前零件的抬起/插入/退出会不会碰到还在桌上的晚装零件"（`bsfs/req_motion_masks.py` 文件头说明）。主线对应的是 expand 里的 hover、下降、lift、抬起直线（带着桌上零件 table 一起查）和装配侧的 preins→ins、退出直线（世界 A 含桌上零件）。两边查的是同一件事，SEALP 用网格平移采样，主线用球体直线段。
2. SEALP 用网格，精确但慢；主线用球体，快、可批量，但有球体近似误差（立柱之间只有约 2 cm 缝时很敏感，见第 6 节）。
3. 主线查了**直线段**（下降、抬起、退出），SEALP L1.5 只查 IK 连续性，不查碰撞。
4. 主线的转运和空载段在搜索时**只算代价、不查碰撞**（lazy，到验证再查）。SEALP 同样不查。两者在这一点上是一样的。

---

## 5. Cost

### SEALP：`bsfs/cost.py: step_cost`
```
HOME → 桌面位上方(lift 0.10) → 桌面位 → 上方 → 装配位上方 → 装配位 → 上方 → HOME
cost = 这条折线的欧氏长度（米），用的是物体原点
```
- 每一步两边都回 HOME，所以每一步独立，跟前后零件无关 → 搜索里的代价是可加的。
- `step_lb`：弦长作下界。
- 排序 `lex_key(cost, -clearance, -manip)`。

### 主线：`expand.py` 168–230 行
```
transfer：lift → preins 的关节直线时间（带零件）
connection：retreat(p) → 下一个零件（后缀头部）的 hover（第一个零件则是 HOME → hover）
pend：p 前面那段空载的起点未知 → 取所有可能前驱 retreat 的最小值（下界）
时间 t_acc(a,b) = max_j t_trap(|Δq_j|)，VMAX=[4.2,5,5,4.2,3.7,3.7] rad/s，AMAX=10 rad/s²
估计值 g_est = 下界 + 绕路惩罚（--penalty --detax）
```
- **不回 HOME**：零件之间直接连接，所以相邻零件的位置会互相影响代价，这是联合搜索的意义。
- 单位是秒，度量是关节空间，跟真正执行时间相关。

**区别**：SEALP 的 cost 是笛卡尔空间物体路程，且每步回 HOME，等于"每个零件单独离装配位近就好"；主线的 cost 是关节时间，零件之间连着算。三角框架上，主线最后的执行时间 ~14.0 s，SEALP 这一套没有直接给时间。

---

## 6. 最后验证

### SEALP
1. `run.py:438 robust_evaluate_layout`：调用 `find_optimal_initial_layout_tower_strict_pycharm.py:2455 evaluate_layout`，**正向**逐个零件重做 L0/L1/L2（reason_common_gids），**不是 RRT**。失败用不同种子重试最多 5 次，区分确定性失败和随机失败。
2. 取最好：先 cost，再指纹（`run.py:573 _site_best`）。
3. `run.py:616 _yaw_refine`：每个零件 yaw 0..340°，步长 20°，最大化 `layout_score`（容差带 0.06，2 轮）。
   `layout_score = (0.3·grasp + 0.4·manip + 0.1·dist + 0.2·rot) × (0.90 + 0.10·左右分布)`（权重 239–242 行，公式 2911–2920 行）：可用抓取数量（min 占 0.7、mean 占 0.3）、平均可操作度、平均距离、平均旋转角。
   **注意：yaw 细化用的是 layout_score，不是搜索时的 cost**，两个目标不一致。
4. [8/9] 再做一次 witness。
5. `run.py:483 _final_certification`：每个零件再 certify 一遍（level 2）。
6. 可选 `--l3-witness`：`validate_full_sequence_l3`（3000 行），用 WRS RRT 真正规划整条序列。**默认不开**。

### 主线：`joint_search/validate.py`
1. `forward_steps`（26 行）：把反向后缀展开成正向的分段（transit / pick / transfer / insert / retreat）。
2. `confirm`（111 行，lazy）：先用自己的球碰撞器查转运直线（带零件、闭合夹爪，`_straight_free` 58 行），直线可行就用下界时间；不行再调 cuRobo（`plan_time` 73 行，最多重试 3 次）。计划按估计值排序，代价超过当前最好就停（LazySP 风格）。
3. `witness`（128 行）：对选中的计划，**每一段都**用 cuRobo 真规划（graph + TrajOpt），用 TOPP 计时。
4. `select`（148 行）：全部 inf 时 best=None。

**区别**：主线默认就做真实运动规划（cuRobo）；SEALP 默认只做离散点检查，RRT 是可选的。

---

## 7. 汇总对照表

| 环节 | SEALP | 主线 | 主线优点 | 主线缺点 / 已知问题 |
|---|---|---|---|---|
| 顺序 | 输入 | 搜索 | 联合优化 | 分支大，需要 B/M 剪枝 |
| 姿态候选 | FS + 90° fallback，cap 12 | 同一函数 | 一致 | 继承 WRS FS 漏面 |
| 位置候选 | 5 cm 网格 + 过滤 + FPS，12 个 | 撒点 + FPS，40 个 | 多 | 过滤在 IK 目录里做，跟搜索状态无关 |
| 每位置姿态 | 搜索中第一个可行 | 预处理中第一个有抓取 | 快 | 不随已放零件变化 |
| HOME 检查 | 有 | 有（10-23 补上） | — | — |
| 抓取 | 不在状态里 | 在状态里 | 关节时间准 | 每个位姿只存 1 个 IK 解 |
| 可行性 | 网格，端点 + IK 连续 | 球体，端点 + 直线段 | 批量快、查了直线段 | 球体近似（2 cm 缝时敏感） |
| 局部运动 vs 桌上晚装零件 | req masks（网格平移采样） | expand 直线段（球体） | 批量快 | 球体近似 |
| nogood | 有 | 无 | — | 可加 |
| cost | 笛卡尔折线，回 HOME | 关节时间，零件相连 | 跟真实时间相关 | 直线下界忽略绕路 |
| 启发 | 无（beam）/ 网格距离（exact） | h(R) + pend | 可采纳 | 早期层排序误差大（Spearman −0.07） |
| 后处理 | yaw 细化（按 layout_score） | 无 | 目标一致 | 没有 yaw 细化 |
| 最终验证 | evaluate_layout（离散），RRT 可选 | cuRobo 真规划 | 结果能执行 | witness 与 confirm 的模型不一致（见下） |

---

## 8. 现在 sealp 模式 witness 失败的原因（只诊断，没改）

- 失败段：post_B 的 transfer（lift → preins）。cuRobo 认为**起点**（lift 姿态）就不可行。
- 去掉桌上的 post_C 后起点就可行 → 是带着的 post_B 和桌上的 post_C 撞了。
- 原因：两套碰撞模型不一样：
  - 我们的碰撞器：post_B 用 903 个半径 4 mm 的表面球 + 1 mm buffer → 判定直线无碰撞，confirm 通过；
  - cuRobo：post_B 用 MorphIt 拟合的 9–10 个球（半径 12–52 mm），比零件胖约 2 cm，而且每次拟合不一样（不确定）。
- 三角框架上两根立柱中心距 7.9 cm，缝只有约 2 cm，post_B 抬起后底面离桌面只有 5 cm，所以胖了 2 cm 就撞。
- 可选修法（还没做）：① cuRobo 那边用更紧、固定种子的拟合球；② 两边用同一套球（把我们的表面球给 cuRobo）；③ lift 抬得更高。

---

## 9. 判断哪个设计更好时，建议看这几点

1. **cost 一致性**：SEALP 搜索用折线、细化用 layout_score，两个目标不一样；主线从头到尾都是关节时间。这一点主线更合理。
2. **可行性的保守程度**：SEALP 网格精确但只查离散点；主线查了直线段但用球近似。两者都不查中间路径。
3. **验证的真实程度**：主线默认真规划，SEALP 默认不规划。
4. **SEALP 值得学的**：nogood 缓存、搜索中按状态选第一个可行姿态、yaw 细化（但应该用关节时间做目标）。
5. **还缺对照实验**：同一场景下 ① 固定顺序 + 主线 cost ② SEALP 原版 ③ 主线联合搜索，比较最终 cuRobo 执行时间。没有这个就没法说联合搜索比 SEALP 好多少。

---

## 10. 2026-10-24 更新：主线的姿态改成搜索中按顺序试（和 SEALP 一样）

- `Problem(orient=...)`，命令行 `--orient first|best|fixed`，默认 `first`。
  - `fixed`：旧做法，搜索前在"只有桌面 + 底座"的世界里，每个位置定死第一个有 IK 抓取的姿态。
  - `first`（默认，SEALP 语义）：候选表保留每个位置所有有可用抓取的姿态（SEALP 顺序）。expand 里每一轮只检查每个未定位置的"当前姿态"（所有位置合成一个 GPU 批次），失败或和桌上零件重叠就下一轮试下一个姿态；抓取的下界排序只对真正检查到的姿态计算。
  - `best`：所有可行姿态都保留，由代价来排。
- 搜索里 `per_x` 改为按位置 (x, y) 计数，不再按 (位置, 姿态) 计数。
- 三角框架结果（`--B 10 --M 5 --N 5`）：

| 模式 | 选中计划 | witness | 搜索时间 | 检查的 (位置, 抓取) |
|---|---|---|---|---|
| fixed | 14.010 s | 失败 | 91 s | 27,907 |
| first | **13.820 s** | 通过 | 113 s | 32,158 |
| best | **13.647 s** | 通过 | 437 s | 125,956 |

`first` 让 post_B 在 (0.144, −0.170) 躺倒放（fs_03），避开了 fixed 模式下 post_B 竖着抬起时撞 post_C 的问题。
