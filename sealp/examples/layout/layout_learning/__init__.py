"""Neural-Guided Global-to-Local Layout Search — 学习模块。

该子包实现"神经网络辅助快速布局搜索"框架:
    - features.py : 统一特征提取 (flatten / set / graph / proposal target)
    - dataset.py  : jsonl 数据集 + collate
    - losses.py   : 复合损失 (feasibility / score / xy proposal / region)
    - models/     : 10 个模型 (MLP/DeepSets/SetTransformer/GCN/GAT/Transformer/
                    PointNet/CVAE/Diffusion/SAGPN)
    - train.py    : 统一训练循环 + 指标
    - infer.py    : 统一推理 (proposal / scorer -> top-K candidate layouts)

设计原则(务必遵守):
    神经网络只做 proposal / pre-filter / ranking / score prediction,
    绝不替代原始 evaluate_layout 与 motion-level (L3) validation。
"""

from . import features  # noqa: F401

__all__ = ["features"]
