from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, dm: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dm)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dm, 2, dtype=torch.float32) * (-math.log(10000.0) / dm)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DepthContextEncoder(nn.Module):
    """Transformer over the latent depth sequence; returns the last-position state."""

    def __init__(
        self,
        dz: int = 64,
        dm: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Linear(dz, dm)
        self.pos = SinusoidalPositionalEncoding(dm)
        layer = nn.TransformerEncoderLayer(
            d_model=dm,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dm = dm

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h0 = self.pos(self.proj(z))
        h_all = self.encoder(h0)
        return h_all[:, -1, :]
