from __future__ import annotations

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Actor with a trainable backbone and a separate output head for calibration."""

    def __init__(self, state_dim: int = 129, hidden: tuple[int, int] = (256, 128)):
        super().__init__()
        h1, h2 = hidden
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(h2, 1)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(q)
        return self.output_layer(feat).squeeze(-1)

    def features(self, q: torch.Tensor) -> torch.Tensor:
        return self.backbone(q)

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def output_parameters(self):
        return self.output_layer.parameters()

    def backbone_parameters(self):
        return self.backbone.parameters()


class Critic(nn.Module):
    """Q(q, a) critic."""

    def __init__(self, state_dim: int = 129, hidden: tuple[int, int] = (256, 128)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(-1)
        x = torch.cat([q, a], dim=-1)
        return self.net(x).squeeze(-1)
