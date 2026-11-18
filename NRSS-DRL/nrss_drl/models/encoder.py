from __future__ import annotations

import torch
import torch.nn as nn


class SequencePreservingEncoder(nn.Module):
    """Per-depth MLP encoder (dx -> 128 -> dz), keeps sequence length L."""

    def __init__(self, dx: int = 7, dz: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dx, hidden),
            nn.GELU(),
            nn.Linear(hidden, dz),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class SequencePreservingDecoder(nn.Module):
    """Per-depth MLP decoder (dz -> 128 -> dx)."""

    def __init__(self, dx: int = 7, dz: int = 64, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dz, hidden),
            nn.GELU(),
            nn.Linear(hidden, dx),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
