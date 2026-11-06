# 单臂联合反向搜索：欧氏距离基线

输入最终装配体、固定底座、Panthera、桌面和候选初始位姿。每次扩展同时决定下一件
反向拆卸的零件和它的初始摆放；不是先选完整顺序再优化布局。

## 当前场景输入

`bridge_scene.json` 是当前 Bridge 尺寸与固定位置输入：

```json
{
  "assembly_scale": 0.6666666666666666,
  "fixed_base_x_m": 0.3535,
  "friction": 0.5
}
```

这里“缩小1/3”表示每个线性尺寸保留2/3。`tasks/bridge.py` 的基准几何（包括
`tasks/assets/base_plate.stl` 中的孔）、相对目标位置和质心一起乘该比例；
质量按同材质假设乘比例的三次方。桌子、桌面孔、Panthera、机器人运动安全间距均保持不变。
目标世界坐标是“固定底座位置 + 缩放后的相对位置”，固定底座中心没有随模型缩放挪动。
`--scene-config` 可切换输入文件；`--assembly-scale` / `--site-x` 可单次覆盖。
原始模型 `build_bridge()` 默认比例仍为1，避免改变其他历史实验入口。

## 目标

以零件局部原点的世界位置为参考点（本 Bridge 可动件原点为几何中心）：

`J = |home-x[first]| + Σ |x[i]-goal[i]| + Σ |goal[previous]-x[next]|`

单位米，不是执行秒数。没有每步回 Home，也不计最终回 Home、旋转、抬升、插入、
抓取时间、机器人关节运动与绕障。自由路径只在未来完整验证阶段求解。
反向加入 p 时，只增加 p 的携物距离和 `goal[p] -> x[原后缀首件]`；
整个顺序完成后再加 `home -> x[首件]`。不预先假定尚未选出的前驱。

## 运行

在项目根目录用现有共享 cuRobo 环境运行：

```powershell
& 'E:\Projects\wrs-sealp-assemble\dual_sealp\cuRobotest\.venv\Scripts\python.exe' -X utf8 -u -m jbls.run_bridge --candidates 12 --beam-width 8 --compare-fixed
```

首次运行先生成持久文件，后续运行加载同一批文件：

```text
jbls/assets/bridge/
├── meshes/
│   ├── base.stl
│   ├── post_l.stl
│   ├── post_r.stl
│   ├── beam.stl
│   └── load.stl
└── grasps/
    ├── post_l_grasps.pickle
    ├── post_r_grasps.pickle
    ├── beam_grasps.pickle
    └── load_grasps.pickle
```

底座固定，不需要机器人抓取，所以有STL但不生成底座抓取目录。桌面仍来自workspace配置。
四个可动件的初始网格由当前Bridge尺寸生成，落盘后重新读取STL；静力、抓取和碰撞
使用这份STL几何。每次运行的输出目录也复制一份STL，保留该次场景。
STL采用17位有效数字的ASCII格式，并以双精度回读，避免二进制STL的float32量化
破坏孔底、柱顶等名义贴合面；没有放宽静力或碰撞判据。

抓取pickle是兼容WRS `GraspCollection.load_from_disk()` 的对象，包含开口、TCP位置/方向，
以及额外保存的 `asmcore_record`（接近位姿、接触点、法向、候选编号等）。目录阶段明确为
`nominal_candidates`：保存的是未校准候选，不是所有状态都可用的common grasp。
**先保存、再从文件读取**，然后进行目标IK、一次指垫校准、Force Closure和当前环境筛选；
后续布局端IK和common grasp只使用这批文件来源的抓取，不额外生成抓取。
目标位置和初始布局不写死进抓取目录；文件坐标是零件局部系、单位米。

`--asset-dir` 可指定另外一套目录。STL尺寸、夹爪或生成器版本不匹配时明确报错，
使用 `--rebuild-assets` 按当前输入重建。`--prepare-assets-only` 仅准备文件和目标端抓取目录，
不运行联合搜索。例如：

```powershell
& 'E:\Projects\wrs-sealp-assemble\dual_sealp\cuRobotest\.venv\Scripts\python.exe' -X utf8 -u -m jbls.run_bridge --prepare-assets-only
```

输出 `jbls/_output/bridge_时间/result.json` 和自包含 `viewer.html`。
结果的 `asset_files` 记录每件STL/pickle路径、摘要与生成/复用状态；每步另记录
`grasp_file_candidate_id`（文件候选编号）与 `common_grasp_id`（校准、去重后的分析器编号）。
两种编号不是同一个索引；最终开口是按原算法校准后的值，可以与pickle内名义开口不同。
网页展示原 SEALP 单桌、Panthera、初始布局、逐步已安装状态、各步代理距离和多个候选。
鼠标旋转和滚轮缩放查看真实网格；初始状态为 Home，其后各步显示已验证的撤离构型，
只切换离散状态，不播放未经求解的连接运动轨迹。
另生成 `scene_initial.png` / `scene_assembled.png` 作为离屏渲染预览。

## 模块

- `search.py`：独立于机器人库的联合 Beam、逐段计费与剩余代价估计。
- `bridge.py`：S/G/A、Panthera 校准与 Force Closure、候选生成、GPU IK 缓存、
  复用既有 WRS 网格局部动作复核。新入口启用夹爪张开/闭合检查。
- `sealp_domain.py`：调用原 SEALP BSFS Beam 的逐零件网格、几何初筛和中心点分散采样；
  适配当前 Bridge 的 STL、固定底座和单臂 Panthera。
- `run_bridge.py`：参数、实验结果、可选固定顺序对照。
- `asset_files.py`：每件STL准备/读取、WRS兼容抓取pickle读写、几何与夹爪摘要绑定。
- `bridge_scene.json`：当前装配体缩放比例、固定底座位置和摩擦系数。
- `viewer.py`：俯视布局与代理路线。
- `workspace.py`：直接读取 `sealp/config/sample_config.yaml` 的单臂与单桌配置。
- `scene.py` / `scene_viewer.html`：复用 SEALP 洞洞板和桌腿、Panthera 实际网格的三维显示。

## 验证范围

拆后静力平衡先检查，再计算当前状态 S/G/A。目标抓取通过后，使用同一套标定抓取
检查取放端六个关键位姿及局部动作。候选几何初筛复用 SEALP 的网格碰撞、顶点间距近似和
HOME 碰撞检查；反向搜索中两个已选初始零件之间仍检查投影凸包10mm间距，属保守候选域。
候选朝向仅绕世界 Z 转 0/90 度。Bridge 插入方向固定为世界 -Z。
场景直接读取原 SEALP 配置：桌板尺寸 `(0.60,0.60,0.02)` 米，中心 `(0.2335,0,-0.01)`；
单臂基座为世界原点。当前 vendored 配置与 `E:/Projects/wrs-sealp-assemble/sealp/config/sample_config.yaml`
逐字节一致（SHA256 记入运行结果）。`--workspace-config` 可指定同格式配置。
未缩放 Bridge 底座宽0.34米，X=0.40会超出原桌右边界；前次默认固定装配位约
`(0.3535,0,0)`，保留1cm桌边。本次缩放后底座宽约0.226667米，配置保留同一固定装配位。
显式指定越界位置会报错，不静默挪动；配置位置为null时才自动选桌面可容纳的位置。
装配位一旦选定，在联合顺序/初始布局搜索中保持固定。
视觉板孔和桌腿直接复用 `show_initial_layout_from_file.py` / `_viz_common.py`；
碰撞几何仍使用 SEALP YAML 中的实心桌板，孔与桌腿不加入原本没有的碰撞约束。
抓取目录在原箱体抓取基础上绕夹持轴
采样接近角，先筛目标 IK，再重新做真实指垫校准与 Force Closure；G 表示这个
有限目录中已确认合格的数量，不是所有可能抓取的总数。

复用的局部检查采用有限 IK 解、关节插值采样和5mm笛卡尔线偏离阈值，不是连续碰撞证明。
保留既有碰撞覆盖范围、指垫/配合件接触豁免。碰撞模型不覆盖 base/link1，不能直接真机执行。
装配方向空间 A 是评分，不代替局部扫掠检查；S 的扰动检查采用有限方向采样。

`result.json` 明确设置 `full_motion_verified=false`、`safe_to_execute=false`。
本轮没有调用自由空手/携物段的完整 cuRobo/RRT，也没有执行时间或全局最优保证。
代理箭头穿障碍不代表真实规划会穿障碍；它只是目标函数的示意。

固定顺序对照共享同一候选域和过滤器，在联合搜索后复用热缓存；可比较 cost，
不能直接用这两次运行的 wall time 声称速度提升。

测试：`python -m pytest tests/test_joint_search.py -q`。小规模穷举校验累计距离、
联合顺序/布局可行性、Home 边界和结构剪枝顺序。

## 首次完整结果（2026-09-28）

`_output/bridge_euclidean_expanded_20260928/result.json`：每件保留12个候选，Beam=8。
找到8个完整局部可行候选，最好顺序为左柱→右柱→横梁→负载，代理距离1.897847米。
预处理50.439秒，联合搜索17.679秒；固定左柱→右柱→横梁→负载的对照也达到相同距离。
这说明管线能运行，尚不能据此声称联合搜索优于固定顺序。
对照使用热缓存，包含对照的总时间70.700秒。
网页已生成，自动浏览检查受工具的本地文件URL策略限制而未完成。

## 原 SEALP 单桌结果（2026-09-28）

`_output/bridge_sealp_single_table_20260928/` 使用原60cm单桌和固定装配位X≈0.3535。
保留域：左右柱各12、横梁8、负载12；Beam=8，找到8个完整局部可行候选。
最好顺序为右柱→左柱→横梁→负载，欧氏距离2.228944米。
预处理54.515秒，搜索14.978秒（不含之后的网格导出和图片渲染）。
桌面缩小且固定装配位变化，本结果不能与旧场景作为纯算法性能对照。
重新生成显示无需搜索：`python -m jbls.scene jbls/_output/bridge_sealp_single_table_20260928`。

## 缩小1/3的单桌结果（2026-09-28）

`_output/bridge_sealp_scale_two_thirds_20260928/` 使用当前 `bridge_scene.json`：
线性比例2/3，固定装配位保持 `(0.3535,0,0)`。每件保留12个候选，Beam=8。
找到8个完整局部可行候选，最好顺序左柱→右柱→横梁→负载，代理距离1.456146米。
预处理49.861秒，搜索20.585秒，可视化导出10.385秒。
底座约226.667×226.667×13.333mm，方孔边长33.6mm、深8mm；
支柱33.333×33.333×100mm，横梁226.667×40×20mm，负载66.667×40×40mm。
14项几何、孔柱配合、工作空间和搜索测试通过。该结果不是完整运动轨迹验证。

## 从输入到输出的当前主线

1. `sealp/config/sample_config.yaml` 给桌面与机器人；`bridge_scene.json` 给模型比例、
   固定底座X与摩擦；`tasks/bridge.py` 和底座STL给零件形状、质量和最终相对位姿。
   `run_bridge.py` 参数给网格、候选数量、Beam宽度和稳定性采样预算。
2. 构建缩小后的装配体，再整体搬到固定底座世界位置。目标位置因此已知；
   初始摆放位置和安装顺序尚未知，这两项是输出。
3. 每件先有STL，再生成对应Panthera候选抓取pickle；已有匹配文件直接加载。
   从pickle读取的抓取经目标端IK预筛、一次指垫校准与Force Closure后用于后续common grasp。
4. 使用原 SEALP 默认 BSFS Beam 的逐零件候选方法，详见下一节。先按该件占地与桌边
   留量确定范围，用默认5cm步长预算铺网格，做几何/HOME初筛，再分散保留最多12个XY中心。
   每个中心分别检查0/90度朝向及文件抓取的关键位姿IK，最多得到24种位姿，不再次截断为12种。
5. 从完整装配体反向选择一个要拆的零件。先检查拆后剩余结构的静力平衡，
   再检查拆前结构S和该零件目标状态的G、A。未通过则剪掉该选择。
6. 同时枚举该件初始摆放位置，与已定布局保留1cm间距。按该步正向安装时的环境，
   用同一个抓取检查取件/安装端IK、接近/抬升/插入/撤离及夹爪开合。
7. 将该件放到已有安装后缀的前面。新计费为“该件初始→目标”，以及存在后缀时的
   “该件目标→后缀首件初始”。全顺序确定后才补“HOME→第一件初始”。
   已计距离加未定件最小携物距离用于排序，每层只保留8个，继续向前搜索。

## 当前候选方法：原 SEALP BSFS Beam（2026-10-08 替换）

旧的统一6cm全桌网格、圆形根部禁区、按目标距离播种并加入朝向奖励的12位姿选择已移除。
本次直接调用仓库内与原 SEALP 一致的 `infer_assembly_ga.py` 三个工具函数：
`_part_anchor_grid`、`_enumerate_xy_candidates`、`_farthest_point_order`；
它们也是 `bsfs/domain.py::continuous_domain` 使用的函数，不运行 GA/网络。

- 桌面XY范围内缩6cm；每件再按所有允许朝向中最小的X/Y占地半宽分别加2cm留量。
- 默认 `--spacing .05`，原方法下限3cm；根据范围和步长计算点数，再用linspace铺点，
  所以实际相邻点间隔不保证恰好5cm。
- 中心点必须至少有一个朝向通过根部矩形禁区（半宽各12cm）、与固定底座的网格碰撞/
  1cm顶点间距近似、Panthera HOME 碰撞初筛。另逐朝向检查完整投影不越过桌边留量。
- 原最远点采样先取离可行点平均位置最近的点，再逐步补最远点，与目标位置距离无关。
- `--candidates 12` 现在表示每件12个XY中心，`0`表示保留全部中心；选中心后再试两种朝向、
  文件抓取与GPU IK。IK淘汰后中心数可能不足12，不从旧网格补点。
- 保留当前Bridge直立0/90度朝向和距桌面0.2mm的高度约定；没有引入原SEALP的侧放稳定姿态，
  也没有采用其 `_build_searcher` 中1cm的悬空高度。原搜索默认每个XY只接受第一个合格朝向，
  当前联合搜索保留两个合格朝向共同搜索。这些适配差异记入结果。

`result.json` 的 `candidate_generation` 记录来源/参数；每件 `domain_stats` 分别记录原始中心、
几何可行中心、选中中心、IK后中心和位姿数量，并保存每级XY坐标及保留位姿。
搜索的S/G/A、common grasp、局部动作、欧氏距离和Beam逻辑沿用现有主线。

本次完整运行 `_output/bridge_sealp_candidates_20261008/` 找到8个局部检查通过的完整方案；
最好顺序为右柱→左柱→横梁→负载，欧氏距离代理1.524290m。
预处理45.360s，搜索24.963s。各件原始中心数：左右柱各81、横梁72、负载72；
几何可行中心分别41、41、26、42，均分散选12个中心；IK后为左柱12中心/24位姿、
右柱12/19、横梁10/10、负载11/14。30项候选、搜索、几何及文件抓取相关测试通过。
候选域和初筛规则已变化，本次代理距离不能作为与旧结果的纯搜索算法性能对比。
结果仍不包含完整自由运动轨迹，`full_motion_verified=false`。
8. `result.json` 保存顺序、初始布局、S/G/A、抓取、关键关节姿态和距离；
   `viewer.html`、两张PNG展示同一批结果。

S/G/A当前用于过滤，不用于最终排序。cuRobo在这条入口里用于批量IK，
没有调用完整MotionPlanner计算cost；WRS只采样检查局部动作。
这条 `jbls.search` 未调用原BSFS的required-motion masks或Hall剪枝。
完整自由空手/持件运动与全局最优性仍不在本次结果的验证范围内。
