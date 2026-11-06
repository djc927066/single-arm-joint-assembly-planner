"""Model 8: CVAE Layout Proposal Baseline。

条件输入 = assembly sequence / part geometry / goal poses / workspace bounds
(即节点的 STATIC 特征 + global 特征)。通过 latent variable 生成 per-part xy
proposal。生成的 layout 必须再经过 evaluate_layout。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import (BaseLayoutModel, ScorerHead, mlp, masked_mean, static_only,
                   masked_global, PART_DIM, GLOBAL_DIM)


class CVAEProposal(BaseLayoutModel):
    is_generator = True

    def __init__(self, hidden: int = 128, latent: int = 16, dropout: float = 0.1, **_):
        super().__init__()
        self.latent = latent
        # 条件编码 (只用 static 特征)
        self.node_enc = mlp([PART_DIM, hidden, hidden], last_act=True, dropout=dropout)
        self.cond_dim = hidden + GLOBAL_DIM
        # 后验 q(z | x, c): 输入 xy_target + 节点条件, 池化后出 mean/logvar
        self.post_node = mlp([hidden + 2, hidden, hidden], last_act=True)
        self.q_mean = nn.Linear(hidden + GLOBAL_DIM, latent)
        self.q_logvar = nn.Linear(hidden + GLOBAL_DIM, latent)
        # 解码 p(xy | z, c): per-node
        self.dec = mlp([hidden + GLOBAL_DIM + latent, hidden, hidden, 2], last_act=False)
        # feas/score 头 (从条件 + latent)
        self.head = ScorerHead(self.cond_dim + latent, hidden)

    def _encode_cond(self, batch: Dict):
        node_s = static_only(batch["node_feat"], batch["static_mask"])
        nc = self.node_enc(node_s)                        # [B,N,H]
        return nc, batch["node_mask"], masked_global(batch)

    def _decode(self, nc, z, global_feat):
        B, N, H = nc.shape
        zc = z.unsqueeze(1).expand(B, N, self.latent)
        gc = global_feat.unsqueeze(1).expand(B, N, global_feat.shape[-1])
        xy = self.dec(torch.cat([nc, gc, zc], dim=-1))
        return torch.tanh(xy)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        nc, mask, gfeat = self._encode_cond(batch)
        # 后验
        xy_t = batch.get("xy_target")
        if xy_t is None:
            xy_t = torch.zeros(nc.shape[0], nc.shape[1], 2, device=nc.device)
        q_in = self.post_node(torch.cat([nc, xy_t], dim=-1))
        q_pooled = torch.cat([masked_mean(q_in, mask), gfeat], dim=-1)
        mean = self.q_mean(q_pooled)
        logvar = self.q_logvar(q_pooled).clamp(-8, 8)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        xy_pred = self._decode(nc, z, gfeat)
        cond_vec = torch.cat([masked_mean(nc, mask), gfeat], dim=-1)
        out = self.head(torch.cat([cond_vec, z], dim=-1))
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        out.update({"xy_pred": xy_pred, "kl": kl})
        return out

    @torch.no_grad()
    def propose(self, batch: Dict, k: int) -> torch.Tensor:
        nc, mask, gfeat = self._encode_cond(batch)
        B, N, _ = nc.shape
        outs = []
        for _ in range(k):
            z = torch.randn(B, self.latent, device=nc.device)
            outs.append(self._decode(nc, z, gfeat))
        return torch.stack(outs, dim=0).squeeze(1)  # [k, N, 2] (B=1)
