#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Task-agnostic masked BC policy for joint arm/pose/grid actions.

This first model intentionally uses only numeric geometry, graph, pose, grasp
and arm-specific IK observations.  It does not use part-name one-hot features
or a task ID.

Output shape:
    [B, 2, max_poses, H, W]

The architecture is deliberately lightweight for the first pipeline test:
- CNN over occupancy and arm-specific IK maps;
- numeric graph-context encoder for the current part;
- pose/grasp encoder;
- learned arm embedding;
- masked joint categorical action loss.

PointNet mesh embeddings and a full GAT can be added after this loader, loss
and rollout path are verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class ArmBCConfig:
    n_arms: int = 2
    max_poses: int = 16
    grid_height: int = 54
    grid_width: int = 24
    occupancy_channels: int = 4
    part_feature_dim: int = 26
    pose_feature_dim: int = 17
    spatial_hidden: int = 64
    part_hidden: int = 128
    pose_hidden: int = 64
    arm_hidden: int = 16

    @property
    def action_n(self) -> int:
        return (
            self.n_arms
            * self.max_poses
            * self.grid_height
            * self.grid_width
        )


class ResidualSpatialBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ArmBCPolicy(nn.Module):
    def __init__(self, config: ArmBCConfig) -> None:
        super().__init__()
        self.config = config

        spatial_in = (
            config.occupancy_channels
            + config.n_arms * config.max_poses
        )
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(
                spatial_in,
                config.spatial_hidden,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, config.spatial_hidden),
            nn.GELU(),
            ResidualSpatialBlock(config.spatial_hidden),
            ResidualSpatialBlock(config.spatial_hidden),
        )
        self.spatial_head = nn.Conv2d(
            config.spatial_hidden,
            config.n_arms * config.max_poses,
            kernel_size=1,
        )

        # current + global + parent + dependency + symmetry summaries
        part_context_dim = config.part_feature_dim * 5
        self.part_encoder = nn.Sequential(
            nn.Linear(part_context_dim, config.part_hidden),
            nn.LayerNorm(config.part_hidden),
            nn.GELU(),
            nn.Linear(config.part_hidden, config.pose_hidden),
            nn.GELU(),
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(config.pose_feature_dim + 1, config.pose_hidden),
            nn.LayerNorm(config.pose_hidden),
            nn.GELU(),
            nn.Linear(config.pose_hidden, config.pose_hidden),
            nn.GELU(),
        )
        self.arm_embedding = nn.Embedding(
            config.n_arms,
            config.arm_hidden,
        )
        self.joint_bias = nn.Sequential(
            nn.Linear(
                config.pose_hidden
                + config.pose_hidden
                + config.arm_hidden,
                config.pose_hidden,
            ),
            nn.GELU(),
            nn.Linear(config.pose_hidden, 1),
        )

        # Small learnable direct priors help the smoke model exploit the
        # existing hint channels while still allowing the CNN to override them.
        self.ik_prior_scale = nn.Parameter(torch.tensor(1.0))
        self.grasp_prior_scale = nn.Parameter(torch.tensor(0.25))

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        weights = mask.to(values.dtype)
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        numerator = (values * weights).sum(dim=dim)
        denominator = weights.sum(dim=dim).clamp_min(1.0)
        return numerator / denominator

    @staticmethod
    def _neighbor_summary(
        part_features: torch.Tensor,
        adjacency: torch.Tensor,
        current_part_mask: torch.Tensor,
    ) -> torch.Tensor:
        # current_part_mask: [B,P], adjacency: [B,P,P]
        weights = torch.bmm(
            current_part_mask.to(adjacency.dtype).unsqueeze(1),
            adjacency,
        ).squeeze(1)
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = weights / denominator
        return torch.bmm(
            weights.unsqueeze(1),
            part_features,
        ).squeeze(1)

    def _part_context(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        part_features = batch["part_features"]
        part_mask = batch["part_mask"]
        current = batch["current_part_mask"]

        current_feature = self._masked_mean(
            part_features,
            current,
            dim=1,
        )
        global_feature = self._masked_mean(
            part_features,
            part_mask,
            dim=1,
        )
        parent_feature = self._neighbor_summary(
            part_features,
            batch["parent_adjacency"],
            current,
        )
        dependency_feature = self._neighbor_summary(
            part_features,
            batch["dependency_adjacency"],
            current,
        )
        symmetry_feature = self._neighbor_summary(
            part_features,
            batch["symmetry_adjacency"],
            current,
        )
        return torch.cat(
            [
                current_feature,
                global_feature,
                parent_feature,
                dependency_feature,
                symmetry_feature,
            ],
            dim=-1,
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.config
        occupancy = batch["occupancy"]
        ik_hint = batch["ik_hint_by_arm"]
        batch_size = occupancy.shape[0]

        if occupancy.shape[1:] != (
            cfg.occupancy_channels,
            cfg.grid_height,
            cfg.grid_width,
        ):
            raise ValueError(
                f"occupancy shape={tuple(occupancy.shape)}, "
                f"expected [B,{cfg.occupancy_channels},"
                f"{cfg.grid_height},{cfg.grid_width}]"
            )
        if ik_hint.shape[1:] != (
            cfg.n_arms,
            cfg.max_poses,
            cfg.grid_height,
            cfg.grid_width,
        ):
            raise ValueError(
                f"ik_hint_by_arm shape={tuple(ik_hint.shape)}"
            )

        spatial_input = torch.cat(
            [
                occupancy,
                ik_hint.reshape(
                    batch_size,
                    cfg.n_arms * cfg.max_poses,
                    cfg.grid_height,
                    cfg.grid_width,
                ),
            ],
            dim=1,
        )
        spatial_logits = self.spatial_head(
            self.spatial_encoder(spatial_input)
        ).reshape(
            batch_size,
            cfg.n_arms,
            cfg.max_poses,
            cfg.grid_height,
            cfg.grid_width,
        )

        part_embedding = self.part_encoder(
            self._part_context(batch)
        )
        pose_input = torch.cat(
            [
                batch["pose_features"],
                batch["grasp_hint"].unsqueeze(-1),
            ],
            dim=-1,
        )
        pose_embedding = self.pose_encoder(pose_input)

        arm_ids = torch.arange(
            cfg.n_arms,
            device=occupancy.device,
            dtype=torch.long,
        )
        arm_embedding = self.arm_embedding(arm_ids)

        part_term = part_embedding[:, None, None, :].expand(
            -1,
            cfg.n_arms,
            cfg.max_poses,
            -1,
        )
        pose_term = pose_embedding[:, None, :, :].expand(
            -1,
            cfg.n_arms,
            -1,
            -1,
        )
        arm_term = arm_embedding[None, :, None, :].expand(
            batch_size,
            -1,
            cfg.max_poses,
            -1,
        )
        bias = self.joint_bias(
            torch.cat(
                [part_term, pose_term, arm_term],
                dim=-1,
            )
        ).squeeze(-1)

        logits = (
            spatial_logits
            + bias[..., None, None]
            + self.ik_prior_scale * ik_hint
            + self.grasp_prior_scale
            * batch["grasp_hint"][:, None, :, None, None]
        )
        return logits

    def masked_flat_logits(
        self,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        logits = self(batch).reshape(-1, self.config.action_n)
        mask = batch["action_mask"].to(torch.bool)
        if mask.shape != logits.shape:
            raise ValueError(
                f"action mask shape={tuple(mask.shape)}, "
                f"logits shape={tuple(logits.shape)}"
            )
        if not torch.all(mask.any(dim=1)):
            raise RuntimeError("at least one sample has an empty action mask")
        return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)

    def decode_action(
        self,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        value = action.to(torch.long)
        col = value % cfg.grid_width
        value = value // cfg.grid_width
        row = value % cfg.grid_height
        value = value // cfg.grid_height
        pose = value % cfg.max_poses
        arm = value // cfg.max_poses
        return arm, pose, row, col
