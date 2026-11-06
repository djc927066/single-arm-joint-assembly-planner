# cuRobo 布局 cost：代码审计与建议方案

## 当前已经实现

- `asmcore/`：接触、静力平衡、扰动稳定性、Force Closure、G/A、正向与反向顺序搜索。
- `adapters/`：Panthera 单臂提供器和真实夹爪端口；共同抓取与运动证据还没有正式实现。
- `costlab/`：物体级体素场、空手/携物两段代理代价、条件 top-k，以及少量 WRS RRT 对照实验。
- `curobo_bridge/`：WRS Panthera 导出、MorphIt 碰撞球、批量碰撞、批量 IK、multi-env
  MotionPlanner、携物球附着、WRS 网格复核和若干基准。
- `jbls/`：目前只有设计说明，联合反向搜索尚未形成可运行包。

## 运动层审计中需要先修的点

1. `PantheraPlanner` 默认使用 `panthera_wrsext`，它没有 `base_link` 和 `link1`。布局靠近
   机器人时，这两部分与零件/障碍的碰撞不会进入 cuRobo，也不会进入当前 WRS 复核。
2. `link2-link5` 被全局加入自碰撞忽略表；`checked_pairs()` 又按同一忽略表生成网格
   复核对，因此这一对在规划和复核两层都不会检查。HOME 的模型接触应使用专门的
   起点豁免或更合理 HOME，不能把整条轨迹上的 link 对永久关闭。
3. `planner.py` 从 `bench_self_collision.py` 导入生产函数，核心层反向依赖实验层。
4. 批量 IK 写在 800 多行的 `assembly_demo.py` 中；该文件同时负责塔形数据、抓取筛选、
   规划、计时和 HTML 数据导出，不能作为联合搜索的稳定 API。
5. 现有 `plan_batch()` 面向 `TableScene`，不支持“每个环境不同布局 + 携物附着 + 两段
   cost”的批量查询；而且把整批耗时平均分给候选，不是候选本身的难度值。
6. 携物球的包络补偿由 3000 个表面采样点估计，适合快速代理，最终运动证据仍需更密的
   几何复核。

## 建议的 cost：批量双段轨迹能量

不要为每个布局调用一次完整 MotionPlanner。对一个搜索层中的全部布局候选，计算：

1. **端点集合**：用 cuRobo 批量 IK 得到上一件撤离位形集合，以及当前件每个 Force
   Closure 抓取在 pick/lift/pre-insert/place 的 IK 分支。目标侧结果按
   `(装配状态, part, grasp)` 缓存，布局改变时只重新计算 pick 侧。
2. **两段种子**：空手段为“上一件撤离 → 当前件预抓取”，携物段为“lift →
   pre-insert”。先用关节线性插值，再加少量正弦基函数表示绕障形变，而不是优化每个
   时间点的六个关节。
3. **GPU 批量碰撞能量**：把不同布局装成 multi-env，调用 cuRobo 的
   `get_scene_self_collision_distance_from_joint_trajectory()`；携物段启用当前已有的
   `attached_object` 球。这样机器人全部 link、自己碰自己、所拿零件碰环境都进入 cost。
4. **固定步数修正**：利用 cuRobo 的碰撞梯度，只对低维绕障系数做 4--12 步梯度下降。
   它是截断的轨迹优化，用固定计算预算产生可比较分数，不要求得到严格无碰路径。
5. **抓取边缘化**：同一布局不提前锁死某个 grasp。对所有保持 pick/place 同一相对抓取的
   IK 分支取 soft-min；低温时接近最佳可执行抓取，同时可用有效分支数作为鲁棒性奖励。

对候选布局 `x`、抓取分支 `g`，建议分数为：

```text
E(x,g) = w_len  * 两段关节弧长
       + w_acc  * 关节二阶差分
       + w_col  * 沿途碰撞势垒积分
       + w_btl  * 最窄间隙惩罚
       + w_lim  * 关节限位惩罚
       + w_res  * 固定步优化后的碰撞残差和梯度范数

C_hat(x) = softmin_g E(x,g) - w_rob * log(1 + 有效共同抓取分支数)
```

其中“最窄间隙”和“残差/梯度范数”比单纯欧氏距离多表达两件事：机械臂是否必须经过
窄通道，以及给定少量优化步后轨迹仍有多难被推出障碍。这正对应当前体素场无法看到的
机械臂本体碰撞。

### 进一步收紧：把“固定预算后的优化响应”当作难度

只计算一次直线轨迹碰撞会过度惩罚那些容易绕开的障碍；直接跑到 TrajOpt 收敛又退化成
“对每个布局完整规划”。折中办法是在相同初始种子上只做固定的 0、2、4、8 步修正，记录：

```text
R_k = 第 k 步后的碰撞残差 + 自碰撞残差
AUC = R_0 + R_2 + R_4 + R_8
stagnation = R_8 / (R_0 + eps)
```

容易绕开的布局会快速下降；窄通道、错误 IK 分支或机械臂本体被卡住的布局下降慢。cost
使用 `AUC + stagnation + 最窄间隙惩罚`，完整 MotionPlanner 只验证 top-K。这是把 cuRobo
当成固定计算预算的“运动难度探针”，而不是把规划器输出直接冒充论文方法。

### 布局鲁棒性

每个名义布局同时批量查询少量扰动，例如 `(x±5 mm, y±5 mm, yaw±2°)`，使用均值和
CVaR/最差分位数形成鲁棒 cost。这样低 cost 必须对摆放误差稳定，避免搜索偏爱一个只有
单一 IK 分支、间隙极小的脆弱布局。扰动查询仍能与 100 个名义布局一起放入 GPU batch。

## 与已有工作的创新边界

- cuRobo 已经提供并行 IK、全臂碰撞、几何规划和 TrajOpt；使用这些功能不构成算法创新。
- cuTAMP（RSS 2025）已经并行优化 plan skeleton 下的 grasp、placement 和关节配置，并用
  cuRobo 求完整轨迹。因此“把布局和 grasp 放在 GPU 上一起算”也不能单独作为贡献。
- 本项目可争取的贡献是装配专用组合：稳定性约束下的反向顺序/布局搜索；以前驱零件为
  条件的空手+携物双段 cost；同一 common grasp 贯穿 pick/place；对 grasp/IK 分支做集合
  聚合；以及用固定步优化残差曲线预测完整规划难度。
- 为保持搜索结论可靠，IK/碰撞确定失败可以硬剪枝；代理 cost 只用于排序、beam/top-k。
  若论文要求最优性，A* 下界继续使用可证明的直线弦或关节时间下界，不能把该代理冒充
  admissible heuristic。

## 搜索中的四级用法

1. S/G/A、占据和简单几何先做确定性剪枝。
2. 批量 IK：没有共同抓取/IK 分支的布局直接删除。
3. 全部剩余布局做一次直线轨迹能量；保留每个条件分组的 top-M。
4. top-M 做固定步数的低维轨迹修正；只对最终 top-K 调完整 cuRobo MotionPlanner，取得
   可执行运动证据。反向搜索的 cost 用第 3/4 级，MotionPlanner 只负责确认，不进入每个
   网格点的循环。

不同零件的附着球不同，应按 `(当前零件, 装配状态)` 分批；同一零件的数百个布局可放进
一个 multi-env batch。权重用少量完整 cuRobo 结果拟合排序损失或 top-k 命中率，无需再用
RRT 生成标签。

## 文件清理建议

- 保留 `tests/` 当前八个文件；它们只有约 600 行，分别锁定科学结论、依赖边界、Panthera
  坐标/夹爪和 cuRobo 模型，并非重复的一次性脚本。
- 已删除 `tmp/` 中十个一次性审计脚本和论文文本摘录，并将 `tmp/`、`costlab/_out/`
  加入 `.gitignore`。
- `curobo_bridge/bench_*.py`、`costlab/*experiment.py`、`truth_rrt.py` 和绘图脚本应迁到
  `experiments/`，不要由生产代码导入；它们是论文复现实验，不建议删除。
- `curobo_bridge/_output/`、`costlab/_out/` 是生成物，可以随时清理或移到论文实验归档。
- `panthera_full` 与 `panthera_wrsext` 保存了重复的大网格。生产代码改成只从 full 模型
  选择 collision links 后，可删除 wrsext 的重复 mesh；在修好模型选择前先不要删。
- 不建议现在裁剪整份 `wrs/`。先用 import trace 和 Panthera/SEALP 集成测试确定依赖闭包，
  再做一次单独的 vendor 瘦身提交。

## 实现顺序

1. 修复 full-link 碰撞和 HOME 自碰撞豁免。
2. 把批量 IK、附件球缓存和批量碰撞距离抽成正式 API。
3. 先实现“零步直线能量”，测 100--1000 个布局的吞吐与 top-k 命中。
4. 加低维轨迹形变与固定步梯度，做消融：无机械臂、直线全臂、梯度修正全臂。
5. 接入 `jbls` 的条件 top-k，再用少量完整 MotionPlanner 做最终运动证据。

这套组合可以作为论文方法候选，但“首次提出”需要单独做文献检索后才能下结论。
