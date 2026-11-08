# single-arm-joint-assembly-planner

单臂 Panthera-HT 的**装配顺序 × 初始布局联合搜索**。在离散候选位置和给定代价下，求累计运动代价最小的顺序和布局，再用 cuRobo 和网格碰撞检查验证。

主线是 [`lattice_joint/`](lattice_joint/README.md)。

## 方法

1. **状态网**：节点是已装集合 S，箭头 S —p→ 表示接下来装零件 p。
2. **D(S, p)**：只看已装零件时，p 有可用抓取的桌面位置。
3. **F 表**：去掉"桌上零件互相挡"这个约束，在状态网上做动态规划，得到每个前缀的代价下界。
4. **桌面定价**：桌面上一个点最多被一个零件盖住。给桌面点定价，加进 F 的单步代价，再减去价钱总和，得到更紧但仍然成立的下界（拉格朗日松弛）。每个未定集合单独调价。
5. **反向最好优先搜索（A\* 分支定界）**：从最后一个零件往前定零件和位置，后缀代价精确计算，前缀代价取 F 下界和定价下界中较大的一个。取到一个位置时，才检查它和桌上后装零件的冲突。先用反向贪心得到初始方案 U。
6. **停止**：当前最好方案的代价 U 不大于所有未展开分支的下界时，U 就是离散候选下的最优。
7. **验证**：选抓取，用 cuRobo 规划自由运动，用网格检查整个执行过程。

详细说明见 [lattice_joint/README.md](lattice_joint/README.md)。

## 结果（2026-11-08）

三个模型都在当前筛选下求出最优，并通过 cuRobo 规划和网格碰撞检查。代价是欧氏代理（单位 m）。

| 模型 | 零件 | 稳定顺序 | 最优代价 | 下界：原始 → 定价 | 弹出次数：定价 / 不定价 | 搜索时间：定价 / 不定价 | 验证后执行时间 |
|---|---:|---:|---:|---|---|---|---:|
| triangular_frame6 | 6 | 48 | **3.1407** | 3.0198 → 3.0712 | 183 / 546 | 3.3 s / 7.0 s | 约 15.3 s |
| triangular_frame8 | 8 | 252 | **4.4716** | 4.2390 → 4.3943 | 31,908 / 85,436 | 53.2 s / 94.0 s | 约 21.6 s |
| twin10 | 10 | 504 | **5.2287** | 4.6434 → 5.2251 | 169,243 / 188 万次仍未解出 | 67.6 s / 458 s 时内存不足停止 | 约 25.5 s |

最优方案：

| 模型 | 装配顺序 |
|---|---|
| triangular_frame6 | A → B → C → BC → AB → CA |
| triangular_frame8 | A → B → AB → C → BC → CA → cap_BC → cap_AB |
| twin10 | t_pedestal → t_arm → t_block → t_bar → t_cap → g_post_W → g_post_E → g_beam → g_block → g_cap |

从零开始的全流程时间（建 IK 目录、搜索、验证）：triangular_frame6 459 s，triangular_frame8 803 s，twin10 约 717 s（搜索 647 s 和验证 70 s 分两次测得）。建 IK 目录占总时间的一半以上，搜索只占 1% 到 7%。这些都是单次测量，机器负载会影响时间。

环境是 RTX 5060 笔记本 GPU（8 GB）。方案、验证结果和回放动画在 [`docs/results/`](docs/results)，图在 [`docs/figs/mainline/`](docs/figs/mainline)。[`docs/figs/mainline/reverse_search_demo.html`](docs/figs/mainline/reverse_search_demo.html) 是反向搜索的逐步演示。

## 目录

```
lattice_joint/  主线：状态网、F 表、桌面定价、反向搜索、验证
joint_search/   旧主线（反向 beam）；主线复用它的场景、IK 目录和验证器
lazy_cost/      场景、碰撞检查器、TOPP 计时、cuRobo 路径规划
curobo_bridge/  cuRobo 接口和 Panthera 模型
asmcore/        接触、静力平衡、顺序质量（从 WRS2 取出，不依赖 wrs）
adapters/       与 wrs 耦合的接口层
jbls/  tasks/   早期搜索和建模工具；重新生成场景模型时要用
scenes/         场景数据：triangular_frame6、triangular_frame8、benchmarks（rack10、rack10_tall、twin10、viaduct10）
experiments/    方法探索、穷举对比、基线（experiments/baselines：先定顺序再排布局）、论文和汇报用图的脚本
wrs/  sealp/    第三方基础代码（只上传 Panthera 需要的模型数据）
docs/           设计文档、图、结果；项目早期说明在 docs/PROJECT_HISTORY.md
tests/          测试
```

`wrs/`、`sealp/` 和各研究包都是平级顶层包，要从仓库根目录运行。

## 环境

已验证的环境：Windows 11、Python 3.11、NVIDIA GPU、PyTorch 2.11（CUDA 12.8）。

1. 安装 PyTorch 和 cuRobo（cuRobo 不在 `requirements.txt` 里，需要按其说明单独安装）。
2. 安装其余依赖：

```bash
pip install -r requirements.txt
```

`wrs` 固定使用 `trimesh==5.0.0`。不要把 WRS2 的同名 `wrs` 包装进同一环境。

## 运行

```bash
python -X utf8 -m lattice_joint.run --model twin10 --quiet
```

```bash
python -X utf8 -m lattice_joint.validate --model twin10
```

`--model` 可以是 `triangular_frame6`（默认）、`triangular_frame8`、`twin10` 等。`--no-prices` 关掉桌面定价，用于对比。结果写到 `lattice_joint/_output/`。第一次运行会建 IK 目录（`joint_search/_cache/`）和网格检查缓存（`lattice_joint/_cache/`），每个模型约 5 到 10 分钟。

基准模型（`scenes/benchmarks`）的底座原点在厚度中心，默认装配中心会自动把 z 设为半个底座厚度。手动传 `--goal-pos` 时 z 要用同样的值，否则装配体会陷进桌面。

生成回放动画需要一个转换后的方案文件，示例是 `docs/results/triangular_frame6_replay.json`：

```bash
python -X utf8 -m joint_search.viz_plan docs/results/triangular_frame6_replay.json --jaw contact
```

## 当前状态

- triangular_frame6、triangular_frame8、twin10 已在当前筛选下完成搜索和 cuRobo 验证。
- viaduct10、rack10_tall 还没有用当前筛选重跑和验证。
- 与穷举的一致性是在旧筛选下核对的（三角框架 3.0509、rack10_tall 5.5767）；当前结果还没有和穷举核对。
- 每个桌面位置只保留 8 个抓取，最优是在这些抓取下的最优；增加抓取数时最优值可能变化，还没做敏感性实验。
- `experiments/baselines` 的“SGA 顺序 + 反向搜索”对比是旧筛选下跑的，需要用当前代码重跑。
- 代价是零件原点的欧氏距离，与抓取无关，不等于实际执行时间。
- 搜索不检查工位之间的自由运动，这部分由事后的 cuRobo 验证负责。
- `tests/` 里有 2 个已知失败：`test_vendored_panthera_fk_ik_and_collision` 和 `test_quality_search_runs_on_cpu`，整理目录之前就存在。

## 来源和许可

- `wrs/`、`sealp/`：来自 `wrs-sealp-assemble`（MIT）。说明见 [docs/WRS_FOUNDATION.md](docs/WRS_FOUNDATION.md)。
- `asmcore/`：来自 WRS2-assembly-planner。
- 第三方许可证在 [`third_party_licenses/`](third_party_licenses)。
