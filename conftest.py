"""把仓库根目录放进 sys.path。

WRS 不是 pip 包, 是工作区仓库: `wrs/`、`sealp/`、`asmcore/`、`jbls/`、`adapters/`
都是仓库根下的平级顶层包, 靠仓库根在 sys.path 上才能互相导入。PyCharm 把项目根
标为 sources root 时会自动做这件事; 本文件让命令行 pytest 和直接跑脚本时同样成立,
与 sealp 各脚本里的 `sys.path.insert(0, _PROJ_ROOT)` 是同一个意图。
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
