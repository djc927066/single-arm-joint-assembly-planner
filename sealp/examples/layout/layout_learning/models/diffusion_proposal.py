"""Model 9: Diffusion Layout Proposal Baseline (简化 DDPM)。

输入 noise layout, 条件 = assembly sequence / part features / goal poses /
workspace bounds。通过 denoising steps 生成 candidate layout (per-part xy)。
生成的 layout 必须再经过 evaluate_layout。

实现为简化版 DDPM: 去噪网络预测 x0 (per-node xy)。训练时随机采样 t, 对
xy_target 加噪声并让网络还原 x0, 因此可直接用统一的 L_xy 训练; 推理时从纯噪声
迭代去噪。保证实验流程能跑通。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import (BaseLayoutModel, ScorerHead, mlp, masked_mean, static_only,
                   masked_global, PART_DIM, GLOBAL_DIM)


class DiffusionProposal(BaseLayoutModel):
    is_generator = True

    def __init__(self, hidden: int = 128, timesteps: int = 50, dropout: float = 0.1, **_):
        super().__init__()
        self.timesteps = timesteps
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))
        self.node_enc = mlp([PART_DIM, hidden, hidden], last_act=True, dropout=dropout)
        # 去噪网络: [node_cond, xy_t, t_emb, global] -> x0_pred (2)
        self.denoise = mlp([hidden + 2 + 16 + GLOBAL_DIM, hidden, hidden, 2], last_act=False)
        self.t_emb = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 16))
        self.head = ScorerHead(hidden + GLOBAL_DIM, hidden)

    def _cond(self, batch: Dict):
        node_s = static_only(batch["node_feat"], batch["static_mask"])
        return self.node_enc(node_s), batch["node_mask"], masked_global(batch)

    def _predict_x0(self, nc, xy_t, t_norm, gfeat):
        B, N, H = nc.shape
        te = self.t_emb(t_norm.view(B, 1, 1).expand(B, N, 1))
        gc = gfeat.unsqueeze(1).expand(B, N, gfeat.shape[-1])
        x0 = self.denoise(torch.cat([nc, xy_t, te, gc], dim=-1))
        return torch.tanh(x0)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        nc, mask, gfeat = self._cond(batch)
        B, N, _ = nc.shape
        xy_t0 = batch.get("xy_target")
        if xy_t0 is None:
            xy_t0 = torch.zeros(B, N, 2, device=nc.device)
        t = torch.randint(0, self.timesteps, (B,), device=nc.device)
        ab = self.alpha_bar[t].view(B, 1, 1)
        noise = torch.randn_like(xy_t0)
        xy_noisy = torch.sqrt(ab) * xy_t0 + torch.sqrt(1 - ab) * noise
        t_norm = t.float() / self.timesteps
        xy_pred = self._predict_x0(nc, xy_noisy, t_norm, gfeat)
        out = self.head(torch.cat([masked_mean(nc, mask), gfeat], dim=-1))
        out["xy_pred"] = xy_pred
        return out

    @torch.no_grad()
    def propose(self, batch: Dict, k: int) -> torch.Tensor:
        nc, mask, gfeat = self._cond(batch)
        B, N, _ = nc.shape
        outs = []
        steps = list(range(self.timesteps - 1, -1, -1))
        for _ in range(k):
            xy = torch.randn(B, N, 2, device=nc.device)
            for t in steps:
                t_norm = torch.full((B,), t / self.timesteps, device=nc.device)
                x0 = self._predict_x0(nc, xy, t_norm, gfeat)
                ab = self.alpha_bar[t]
                if t > 0:
                    ab_prev = self.alpha_bar[t - 1]
                    noise = torch.randn_like(xy)
                    xy = torch.sqrt(ab_prev) * x0 + torch.sqrt(1 - ab_prev) * noise
                else:
                    xy = x0
            outs.append(torch.tanh(xy))
        return torch.stack(outs, dim=0).squeeze(1)  # [k, N, 2]
