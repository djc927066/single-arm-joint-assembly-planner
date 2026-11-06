"""进程级准备：导入顺序、viser 占位、仓库根路径、TF32 隔离。

cuRobo V2 的 ``RobotBuilder`` 在模块顶层 ``import viser``，而本机 venv 为了保住
WRS 的 ``trimesh==5.0.0`` 故意没装 viser（它要求 trimesh<5）。viser 只在
``RobotBuilder.visualize()`` 里用到，所以这里注入一个占位模块：导入成功，
真正调用可视化时才报错。不改动 venv、不降级任何包。
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CUROBO_VENV_PYTHON = Path(os.environ.get("CUROBO_VENV_PYTHON", sys.executable))
ASSET_DIR = Path(__file__).resolve().parent / "assets"
OUTPUT_DIR = Path(__file__).resolve().parent / "_output"


class _ViserUnavailable:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("viser is intentionally not installed in this venv "
                           "(it requires trimesh<5, WRS pins trimesh 5.0.0)")


def _stub_getattr(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return _ViserUnavailable


def _install_viser_stub() -> None:
    try:
        import viser  # noqa: F401
        return
    except ImportError:
        pass
    root = types.ModuleType("viser")
    extras = types.ModuleType("viser.extras")
    root.__getattr__ = _stub_getattr
    extras.__getattr__ = _stub_getattr
    root.extras = extras
    sys.modules["viser"] = root
    sys.modules["viser.extras"] = extras


def prepare() -> None:
    """Call once, before importing WRS or cuRobo."""
    import torch  # noqa: F401  (must precede Panda3D on Windows)

    _install_viser_stub()
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def tf32_state():
    import torch
    return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def set_tf32(state) -> None:
    import torch
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = state
