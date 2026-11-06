"""cuRobo V2 与 WRS1 Panthera-HT 的桥接层。

与 ``asmcore`` 的不变量一致：算法包不 import 本包；本包只在 cuRobo 独立 venv
(``wrs-sealp-assemble/dual_sealp/cuRobotest/.venv``) 中运行，该 venv 以
``--system-site-packages`` 继承 WRS 依赖，因此同一进程可以同时加载 WRS 与 cuRobo。

导入顺序约束（Windows）：必须先 ``import torch`` 再加载 WRS/Panda3D，否则
``c10.dll`` 初始化失败。统一用 :func:`curobo_bridge.runtime.prepare` 处理。
"""
