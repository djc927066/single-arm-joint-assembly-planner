# WRS1 / Panthera-HT 地基来源

本项目根目录下的 `wrs/` 与 `sealp/` 来自：

- 仓库：`https://github.com/blanketxx/wrs-sealp-assemble.git`
- 本机源目录：`E:\Projects\wrs-sealp-assemble`
- 分支：`sealp`
- 基准提交：`057d43eb2b66db27017f26a9bf83e75b1647849d`
- 复制日期：2026-09-17

复制时排除了 Python 缓存、测试缓存、实验输出，以及作为备份存在的
`wrs/robot_sim/manipulators/panthera_ht/panthera_ht.zip`。Panthera 实际运行所需的
URDF、`meshes1/`、夹爪模型和碰撞模型均已保留。

源工作树复制时不是完全干净的。以下本地修改随 `sealp/` 一起进入了本项目：

- `sealp/README.md`
- `sealp/assembly_sequence/asmdef.py`
- `sealp/primitives/transport.py`

这些文件属于当前本机实验版本；论文实验必须继续在本项目 Git 中记录它们的后续修改。

本项目还对 `wrs/__init__.py` 做了一项地基修正：保留原有快捷别名，但改成按需导入。
上游文件会在任何 `wrs.*` 导入时同时加载可视化、MuJoCo 和大量无关机器人；按需导入
避免无界面搜索进程被无关后端影响。Panthera、碰撞和运动模块的实际 API 不变。
同理，`wrs/visualization/panda/world.py` 只在调用 MuJoCo 物理回调时才导入
MuJoCo，普通 Panthera 几何、IK 和碰撞计算不会加载其原生 DLL。

WRS1 是项目运行地基，负责 Panthera-HT 模型、夹爪、IK、碰撞、运动规划和真机接口。
`asmcore/` 是从 WRS3 装配模块抽出的独立数学层。两个同名 `wrs` 实现不能在同一
Python 导入路径中混用。

第三方许可证保存在 `third_party_licenses/`。
