"""模型注册表。

用法:
    from layout_learning.models import build_model, MODEL_NAMES
    model = build_model("sagpn", flat_dim=..., **kwargs)
"""

from __future__ import annotations

from typing import Dict, Optional

from .base import BaseLayoutModel
from .mlp_surrogate import MLPSurrogate
from .deepsets import DeepSets
from .set_transformer import SetTransformer
from .gcn_predictor import GCNPredictor
from .gat_predictor import GATPredictor
from .transformer_encoder import TransformerEncoderModel
from .pointnet_encoder import PointNetGeometry
from .cvae_proposal import CVAEProposal
from .diffusion_proposal import DiffusionProposal
from .sagpn import SAGPN
from .seqrel_layout_net import SeqRelLayoutNet
from .dynaseqrel_layout_net import DynaSeqRelDynEdgeLayoutNet
from .relseqgen import RelSeqGenLayoutNet
from .part_placement_ranker import PartPlacementRankerNet

# 名称 -> 构造器
_REGISTRY = {
    "mlp": MLPSurrogate,
    "deepsets": DeepSets,
    "set_transformer": SetTransformer,
    "gcn": GCNPredictor,
    "gat": GATPredictor,
    "transformer": TransformerEncoderModel,
    "pointnet": PointNetGeometry,
    "cvae": CVAEProposal,
    "diffusion": DiffusionProposal,
    "sagpn": SAGPN,
    "seqrel": SeqRelLayoutNet,
    "dynaseqrel_dynedge": DynaSeqRelDynEdgeLayoutNet,
    "relseqgen": RelSeqGenLayoutNet,
    "part_placement_ranker": PartPlacementRankerNet,
}

MODEL_NAMES = list(_REGISTRY.keys())

# 生成式模型 (proposal) 集合
GENERATOR_MODELS = {"cvae", "diffusion", "sagpn", "relseqgen"}
SCORER_MODELS = [m for m in MODEL_NAMES if m not in GENERATOR_MODELS]

# 各模型 small / base 尺寸预设。small 用于小数据集 (参数量尽量 < 5万)。
_SIZE_PRESETS = {
    "sagpn": {
        "small": {"hidden": 48, "layers": 2, "dropout": 0.2},
        "base": {"hidden": 128, "layers": 3, "dropout": 0.1},
    },
    "seqrel": {
        "small": {"hidden": 48, "layers": 2, "dropout": 0.2},
        "base": {"hidden": 64, "layers": 2, "dropout": 0.15},
    },
    "dynaseqrel_dynedge": {
        "small": {"hidden": 48, "layers": 2, "dropout": 0.2},
        "base": {"hidden": 64, "layers": 2, "dropout": 0.2},
    },
    "relseqgen": {
        "small": {"hidden": 64, "layers": 2, "heads": 2, "mixtures": 5, "dropout": 0.15},
        "base": {"hidden": 128, "layers": 3, "heads": 4, "mixtures": 8, "dropout": 0.15},
    },
    "part_placement_ranker": {
        "small": {"hidden": 64, "dropout": 0.2},
        "base": {"hidden": 128, "dropout": 0.15},
    },
    "cvae": {
        "small": {"hidden": 64, "dropout": 0.2},
        "base": {"hidden": 128, "dropout": 0.1},
    },
    "diffusion": {
        "small": {"hidden": 64, "dropout": 0.2},
        "base": {"hidden": 128, "dropout": 0.1},
    },
}


def size_kwargs(name: str, size: Optional[str]) -> Dict:
    """把 --model-size 映射为该模型的构造参数 (未定义的模型返回空 dict)。"""
    if not size:
        return {}
    return dict(_SIZE_PRESETS.get(name, {}).get(size, {}))


def build_model(name: str, flat_dim: int, model_size: Optional[str] = None,
                **kwargs) -> BaseLayoutModel:
    if name not in _REGISTRY:
        raise ValueError(f"未知模型 '{name}', 可选: {MODEL_NAMES}")
    cls = _REGISTRY[name]
    preset = size_kwargs(name, model_size)
    preset.update(kwargs)   # 显式 kwargs 覆盖 size 预设
    if name == "mlp":
        return cls(flat_dim=flat_dim, **preset)
    return cls(**preset)


def is_generator(name: str) -> bool:
    return name in GENERATOR_MODELS


__all__ = ["build_model", "size_kwargs", "MODEL_NAMES", "GENERATOR_MODELS",
           "SCORER_MODELS", "is_generator", "BaseLayoutModel"]
