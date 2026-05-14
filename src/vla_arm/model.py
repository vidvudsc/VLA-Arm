from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .env import ACTION_DIM


@dataclass
class PolicyConfig:
    image_size: int = 256
    patch_size: int = 16
    robot_dim: int = 5
    hidden: int = 160
    depth: int = 4
    heads: int = 5
    mlp_ratio: float = 3.0
    dropout: float = 0.05


class VLAArmPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig | None = None):
        super().__init__()
        self.cfg = cfg or PolicyConfig()
        if self.cfg.image_size % self.cfg.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        grid = self.cfg.image_size // self.cfg.patch_size
        self.num_patches = grid * grid
        self.patch_embed = nn.Conv2d(
            3,
            self.cfg.hidden,
            kernel_size=self.cfg.patch_size,
            stride=self.cfg.patch_size,
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, self.cfg.hidden))
        self.robot_proj = nn.Sequential(
            nn.Linear(self.cfg.robot_dim, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
        )
        self.pos = nn.Parameter(torch.randn(1, self.num_patches + 2, self.cfg.hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.hidden,
            nhead=self.cfg.heads,
            dim_feedforward=int(self.cfg.hidden * self.cfg.mlp_ratio),
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.cfg.depth)
        self.norm = nn.LayerNorm(self.cfg.hidden)
        self.head = nn.Sequential(
            nn.Linear(self.cfg.hidden, self.cfg.hidden),
            nn.GELU(),
            nn.Linear(self.cfg.hidden, ACTION_DIM),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor, robot: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(image).flatten(2).transpose(1, 2)
        cls = self.cls.expand(image.shape[0], -1, -1)
        robot_token = self.robot_proj(robot).unsqueeze(1)
        tokens = torch.cat([cls, robot_token, patches], dim=1)
        tokens = tokens + self.pos[:, : tokens.shape[1]]
        tokens = self.transformer(tokens)
        return self.head(self.norm(tokens[:, 0]))


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())
