"""Model: SeqRel — Sequence-Aware Relational Layout Scorer (Ours, lightweight).

定位
====
专门面向**顺序约束机器人装配初始布局搜索**的轻量 *scorer / ranker*。它不是
generator, 不回归 staging / station 坐标, 只做两件事:

    1. feasibility 预测 (feas_logit)  —— 候选是否 L2 可行;
    2. 质量排序 (score_pred)          —— 可行候选的相对好坏。

设计动机 (针对 SAGPN 失效根因)
------------------------------
SAGPN 作为 generator, 其评分头只吃 static-only 特征 (staging_xy 被屏蔽), 在单任务
数据集上所有候选静态特征恒等 -> logits 近常数 -> ROC/PR-AUC 被钉死。SeqRel 反其道
而行: **完整消费动态 staging 特征 + 关系边**, 因此能区分同一装配任务下的不同候选。

输入 (全部来自现有 batch, 不改变共享特征维度, 因此兼容所有旧 checkpoint)
    - node_feat  [B,N,PART_DIM]  含动态 staging_xy / 到目标距离;
    - node_mask  [B,N];
    - adj        [B,N,N]         order/parent/spatial 声明边;
    - edge_attr  [B,N,N,EDGE_DIM]  [rel_dx, rel_dy, dist, is_order, is_parent, is_spatial];
    - global_feat[B,GLOBAL_DIM]  装配站相对桌心 / 桌面尺寸 / 零件数 / station_dist。

关系
    * 空间关系: 由 staging 平面坐标现算的 pairwise (dx, dy, dist), 全连接(有效对);
    * 顺序关系: edge_attr 的 order/parent/spatial 标志 (稀疏声明边)。

网络
    Part MLP encoder -> 2 层关系感知消息传递 (残差 + LayerNorm) ->
    mean + max + attention pooling -> 拼接全局特征 -> feas / score / fail 头。

参数量目标 5万~12万 (hidden=64 时约 8.6 万)。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import (BaseLayoutModel, mlp, masked_mean, masked_max,
                   PART_DIM, GLOBAL_DIM, EDGE_DIM)
from .. import features as F

# staging 平面坐标在 node_feat 中的位置 (v1/v2 一致):
#   dim 18:20 = (staging_xy - region_center) / scale  -> 用作 pairwise 空间几何来源
_STAGING_SLICE = slice(F._STATIC_DIM, F._STATIC_DIM + 2)
_PAIR_GEOM_DIM = 3  # dx, dy, dist (来自 staging 平面坐标)


class _RelationMP(nn.Module):
    """关系感知稠密消息传递层。

    msg_ij = MLP([h_j, edge_attr_ij, pair_geom_ij]); 对有效邻居 (含空间全连接) 均值聚合;
    h_i <- LayerNorm(h_i + Dropout(update([h_i, agg])))。
    """

    def __init__(self, dim: int, edge_dim: int = EDGE_DIM, dropout: float = 0.2):
        super().__init__()
        self.msg = mlp([dim + edge_dim + _PAIR_GEOM_DIM, dim, dim], last_act=True)
        self.upd = mlp([2 * dim, dim, dim], last_act=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, rel_attr: torch.Tensor,
                pair_gate: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        h_j = h.unsqueeze(1).expand(B, N, N, D)                  # 源节点 j
        m = self.msg(torch.cat([h_j, rel_attr], dim=-1))        # [B,N,N,D]
        gate = pair_gate.unsqueeze(-1)                          # [B,N,N,1] 有效邻居
        m = m * gate
        deg = gate.sum(dim=2).clamp_min(1.0)
        agg = m.sum(dim=2) / deg                                 # 均值聚合
        upd = self.upd(torch.cat([h, agg], dim=-1))
        h = self.norm(h + self.drop(upd))
        return h * mask.unsqueeze(-1)


class _AttnPool(nn.Module):
    """可学习注意力池化 (masked softmax over valid nodes)。"""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.score(h).squeeze(-1)                      # [B,N]
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(mask <= 0, neg)
        attn = torch.softmax(logits, dim=1).unsqueeze(-1)       # [B,N,1]
        attn = torch.nan_to_num(attn, nan=0.0)
        return (h * attn).sum(dim=1)                            # [B,D]


class SeqRelLayoutNet(BaseLayoutModel):
    """顺序约束装配布局 scorer (feasibility + quality + fail-aux)。"""

    is_generator = False

    def __init__(self, hidden: int = 64, layers: int = 2, dropout: float = 0.2,
                 num_fail_classes: int = F.NUM_FAIL_CLASSES,
                 disable_relation: bool = False,
                 disable_sequence: bool = False, **_):
        super().__init__()
        if disable_relation and disable_sequence:
            raise ValueError(
                "disable_relation 与 disable_sequence 不能同时启用；"
                "结构定位实验必须一次只改变一个因素。")
        self.hidden = int(hidden)
        self.disable_relation = bool(disable_relation)
        self.disable_sequence = bool(disable_sequence)
        self.encoder = mlp([PART_DIM, hidden, hidden], last_act=True, dropout=dropout)
        self.mp = nn.ModuleList([] if self.disable_relation else [
            _RelationMP(hidden, EDGE_DIM, dropout=dropout)
            for _ in range(max(1, int(layers)))
        ])
        self.attn_pool = _AttnPool(hidden)
        pool_dim = 3 * hidden + GLOBAL_DIM        # mean + max + attn + global
        self.trunk = mlp([pool_dim, hidden, hidden], last_act=True, dropout=dropout)
        self.feas_head = nn.Linear(hidden, 1)
        self.score_head = nn.Linear(hidden, 1)
        self.fail_head = nn.Linear(hidden, int(num_fail_classes))

    def _pair_context(self, node: torch.Tensor, edge_attr: torch.Tensor,
                      adj: torch.Tensor, mask: torch.Tensor):
        """构造 pairwise 关系张量与有效邻居门控。

        rel_attr[B,N,N, EDGE_DIM+3] = [edge_attr(order/parent/spatial+goal几何),
                                       staging dx, dy, dist];
        pair_gate[B,N,N] = 有效邻居 (自身除外, j 必须 valid)。空间关系用全连接,
        因此不依赖稀疏 adj, 保证候选布局的相互位置关系一定被看到。
        """
        B, N, _ = node.shape
        sxy = node[..., _STAGING_SLICE]                        # [B,N,2] staging 平面坐标
        dxy = sxy.unsqueeze(1) - sxy.unsqueeze(2)              # [B,N,N,2] (j - i)
        dist = torch.linalg.norm(dxy, dim=-1, keepdim=True)    # [B,N,N,1]
        pair_geom = torch.cat([dxy, dist], dim=-1)             # [B,N,N,3]
        rel_attr = torch.cat([edge_attr, pair_geom], dim=-1)   # [B,N,N,EDGE_DIM+3]

        valid_j = mask.unsqueeze(1).expand(B, N, N)            # j valid
        eye = torch.eye(N, device=node.device, dtype=node.dtype).unsqueeze(0)
        pair_gate = valid_j * (1.0 - eye)                      # 排除自身
        return rel_attr, pair_gate

    def encode(self, batch: Dict) -> torch.Tensor:
        node = batch["node_feat"]
        mask = batch["node_mask"]
        edge_attr = batch["edge_attr"]
        adj = batch["adj"]
        if self.disable_sequence:
            # v1/v2 node dims 14:16 are normalized order_index and is_first.
            # Keep the shared feature schema unchanged; mask only inside this variant.
            node = node.clone()
            node[..., 14:16] = 0.0

            # Keep spatial topology/geometry only.  An edge can carry multiple flags:
            # retain goal-space dx/dy/dist and is_spatial only when spatial==1;
            # remove order-only / parent-only edges entirely.
            spatial = edge_attr[..., 5:6]
            spatial_attr = torch.zeros_like(edge_attr)
            spatial_attr[..., 0:3] = edge_attr[..., 0:3] * spatial
            spatial_attr[..., 5:6] = spatial
            edge_attr = spatial_attr
            adj = (spatial.squeeze(-1) > 0).to(adj.dtype)
        rel_attr, pair_gate = self._pair_context(node, edge_attr, adj, mask)
        h = self.encoder(node) * mask.unsqueeze(-1)
        for layer in self.mp:
            h = layer(h, rel_attr, pair_gate, mask)
        pooled = torch.cat([
            masked_mean(h, mask),
            masked_max(h, mask),
            self.attn_pool(h, mask),
            batch["global_feat"],
        ], dim=-1)
        return self.trunk(pooled)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        z = self.encode(batch)
        return {
            "feas_logit": self.feas_head(z).squeeze(-1),
            "score_pred": torch.sigmoid(self.score_head(z)).squeeze(-1),
            "fail_logits": self.fail_head(z),
        }
