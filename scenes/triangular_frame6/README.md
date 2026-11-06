# 六零件三角支撑框架（隔离模型案例）

本案例用于扩大单臂装配顺序与初始布局联合搜索的测试范围。六个**可动零件**是三根立柱 A/B/C 和三根横梁 AB/BC/CA；固定底座额外提供，不计入六件。所有新增内容在本目录，未替换主线 Bridge。

## 结构与尺寸

- 底座：240 × 240 × 14 mm，三个矩形盲孔，孔深 8 mm，单侧名义间隙 0.2 mm。
- 立柱 A：60 × 54 × 100 mm；B：58 × 62 × 100 mm；C：62 × 56 × 100 mm。
- 横梁 AB：114 × 22 × 20 mm；BC：约 108.353 × 20 × 20 mm；CA：约 123.189 × 18 × 20 mm。
- 完成体高 126 mm。柱底落在距底座底面 6 mm 的孔底，柱顶高 106 mm。
- 三根梁分别沿三个三角形边布置，端部缩短以避免彼此实体相交；梁靠两端柱顶接触面支撑，没有锁扣、胶接或紧固件。
- 六个可动 STL 的模型原点为几何中心；底座原点为底面中心。STL 数值单位是**米**，导入通常默认毫米的 CAD 软件时需乘 1000。
- 固定底座世界位置为 (0.350, 0, 0) m；沿用当前 600 × 600 mm 单桌坐标约定。

## 为什么比 Bridge 更适合作为下一阶段案例

每根梁需要对应的两根柱子：AB 依赖 A+B，BC 依赖 B+C，CA 依赖 C+A。因此可以先安装三根柱子，也可以安装两根柱子后先装它们之间的梁；三个不同横梁都可能成为最后一件。这些依赖应由静力检查核实，而不是直接把预定顺序硬编码给搜索。

三种不同长度、目标朝向的横梁，加上不同截面的柱子，也使布局和抓取候选不完全对称。更多合法顺序不等于已经证明联合优化收益：后续仍要实际比较不同顺序和布局的机器人运动代价。

## 文件

- `build_model.py`：参数化构型、生成 STL、调用现有静力求解器验证子集、生成名义抓取目录和可视化。
- `assets/meshes/`：六个可动 STL 和一个底座 STL。
- `assets/grasps/`：六个 WRS 兼容的 Panthera 名义候选抓取 pickle 及元数据。
- `task.json`：各零件 STL、抓取文件、质量、质心、摩擦、最终位姿、配合关系与示意初始布局。
- `validation.json`：64 个子集静力结果、720 种排列中通过全部前缀检查的顺序、物体级拆卸结果和限制。
- `viewer.html`：离线交互预览，可旋转视角、查看完成体、初始布局和逐件安装状态。
- `model_preview.png`：完成体及爆炸图。
- `build.log`：本次生成日志。

## 验证范围与接入状态

网格封闭性、法向、体积、梁间几何间距以及示意初始布局的桌面边界和零件间距会检查。全部 64 个子集调用现有静力平衡线性规划器，枚举全部 720 种排列并检查每个前缀状态。

这里的“名义静力稳定”只指重力工况，不等于论文 S 的六维扰动评分。名义抓取 pickle 不代表已经通过真实指垫校准、Force Closure、共同抓取或 IK。物体级竖直拆卸不包含夹爪和机械臂；`unknown` 表示通用验证器未能证明，不能当作成功。

初始布局是无几何重叠的手工示意，未优化、未验证机械臂可达。网页中的逐件切换是离散状态，不是连续运动轨迹。本案例尚未接入 Bridge 专用 JBLS 入口、S/G/A 全评分与 cuRobo 全臂连续运动验证，不能直接声称可以实机装配。

## 重新生成

在项目根目录，用现有共享 Python 环境运行（Torch 需要在几何库之前加载，脚本已处理）：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'E:\Projects\wrs-sealp-assemble\dual_sealp\cuRobotest\.venv\Scripts\python.exe' -B -X utf8 -u 'E:\Projects\single-arm-joint-assembly-planner\scenes\triangular_frame6\build_model.py'
```

尺寸参数位于脚本顶部；改变尺寸后如现有 STL/抓取哈希不匹配，需在本隔离案例内显式更新对应资产，避免误复用旧几何抓取。

## 本次实际检查结果

64 个子集中 18 个通过名义静力检查；720 种排列中 48 种每一步通过，与三条两柱支撑规则完全一致。三个横梁均有作为最后一件的静力合格顺序。柱子的通用拆卸验证仍为 unknown（unresolved_clearance_or_unpermitted_contact）；梁的代表性竖直拆卸检查通过。共生成 126 个名义抓取候选，柱子各 28 个、横梁各 14 个。

仅更新预览可在生成命令后加 --preview-only，不重复静力枚举。

联合搜索实验和文献启发见 [experiment_findings.md](experiment_findings.md)；参数化实验脚本为 experiment.py，输出为 experiment_report.json。当前实验只用欧氏距离代理、静力子集结果和桌面AABB碰撞，不是机器人可执行性测试。
## 可复现实验

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'E:\Projects\wrs-sealp-assemble\dual_sealp\cuRobotest\.venv\Scripts\python.exe' -B -X utf8 -u 'E:\Projects\single-arm-joint-assembly-planner\scenes\triangular_frame6\experiment.py'
```

该实验运行现有 `jbls.search.joint_search`，不会调用完整 cuRobo MotionPlanner。网格、Beam 对照、较窄取料区的域传播结果见 `experiment_report.json`。
