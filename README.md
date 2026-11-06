# single-arm-joint-assembly-planner

单臂 Panthera-HT 的**装配顺序 × 初始布局联合搜索**。在离散候选位置和给定代价下，求累计运动代价最小的顺序和布局，再用 cuRobo 和网格碰撞检查验证。

主线是 [`lattice_joint/`](lattice_joint/README.md)。

## 方法

1. **状态网**：节点是已装集合 S，箭头 S —p→ 表示接下来装零件 p。
2. **D(S, p)**：只看已装零件时，p 有可用抓取的桌面位置。
3. **F 表**：去掉"桌上零件互相挡"这个约束，在状态网上做动态规划，得到每个前缀的代价下界。
4. **反向最好优先搜索**：从最后一个零件往前定零件和位置，后缀代价精确计算，前缀代价查 F 表。取到一个位置时，才检查它和桌上后装零件的冲突。
5. **停止**：当前最好方案的代价 U 不大于所有未展开分支的下界时，U 就是离散候选下的最优。
6. **验证**：选抓取，用 cuRobo 规划自由运动，用网格检查整个执行过程。

详细说明、结果和未完成事项见 [lattice_joint/README.md](lattice_joint/README.md)。

## 三角框架结果

| 项目 | 结果 |
|---|---|
| 顺序 | A → B → AB → C → BC → CA |
| 代价（欧氏代理）/ 下界 | 3.1740 / 3.0925 |
| 搜索量 | 492 次弹出，296 次位置检查，48 个稳定顺序，30 条箭头 |
| 验证 | 13 段自由运动全部规划成功，网格检查无碰撞，执行时间约 15.5 s |
| 首次全流程（含建 IK 目录） | 417.2 s |
| 已有 IK 目录时的全流程 | 114.5 s |

环境是 RTX 5060 笔记本 GPU（8 GB）。方案、验证结果和回放动画在 [`docs/results/`](docs/results)，图在 [`docs/figs/mainline/`](docs/figs/mainline)。

## 目录

```
lattice_joint/  主线：状态网、F 表、反向搜索、验证
joint_search/   旧主线（反向 beam）；主线复用它的场景、IK 目录和验证器
lazy_cost/      场景、碰撞检查器、TOPP 计时、cuRobo 路径规划
curobo_bridge/  cuRobo 接口和 Panthera 模型
asmcore/        接触、静力平衡、顺序质量（从 WRS2 取出，不依赖 wrs）
adapters/       与 wrs 耦合的接口层
jbls/  tasks/   早期搜索和建模工具；重新生成场景模型时要用
scenes/         场景数据：triangular_frame6、benchmarks（rack10、rack10_tall）
experiments/    方法探索、穷举对比、论文和汇报用图的脚本
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
python -X utf8 -m lattice_joint.run
```

```bash
python -X utf8 -m lattice_joint.validate
```

结果写到 `lattice_joint/_output/`。第一次运行会建 IK 目录（`joint_search/_cache/`）和手指–零件网格检查缓存（`lattice_joint/_cache/`），三角框架约 5 分钟。

生成回放动画需要一个转换后的方案文件，示例是 `docs/results/triangular_frame6_replay.json`：

```bash
python -X utf8 -m joint_search.viz_plan docs/results/triangular_frame6_replay.json --jaw contact
```

## 当前状态

- 三角框架已在当前筛选下完成搜索和 cuRobo 验证。
- rack10_tall 还没有用当前筛选重跑和验证。
- 与穷举的一致性是在旧筛选下核对的（三角框架 3.0509、rack10_tall 5.5767）；当前的 3.1740 还没有和穷举核对。
- 代价是零件原点的欧氏距离，与抓取无关，不等于实际执行时间。
- 搜索不检查工位之间的自由运动，这部分由事后的 cuRobo 验证负责。
- `tests/` 里有 2 个已知失败：`test_vendored_panthera_fk_ik_and_collision` 和 `test_quality_search_runs_on_cpu`，整理目录之前就存在。

## 来源和许可

- `wrs/`、`sealp/`：来自 `wrs-sealp-assemble`（MIT）。说明见 [docs/WRS_FOUNDATION.md](docs/WRS_FOUNDATION.md)。
- `asmcore/`：来自 WRS2-assembly-planner。
- 第三方许可证在 [`third_party_licenses/`](third_party_licenses)。
