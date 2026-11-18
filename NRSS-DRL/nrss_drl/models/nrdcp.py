from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import SequencePreservingDecoder, SequencePreservingEncoder
from .transformer import DepthContextEncoder


class NRDCP(nn.Module):
    """Noise-robust depth-context perception module."""

    def __init__(
        self,
        dx: int = 7,
        dz: int = 64,
        dm: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        input_noise_sigma: float = 0.05,
    ):
        super().__init__()
        self.encoder = SequencePreservingEncoder(dx=dx, dz=dz)
        self.decoder = SequencePreservingDecoder(dx=dx, dz=dz)
        self.context = DepthContextEncoder(
            dz=dz,
            dm=dm,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.input_noise_sigma = input_noise_sigma
        self.dm = dm

    def encode_latent(self, s: torch.Tensor, add_noise: bool | None = None) -> torch.Tensor:
        if add_noise is None:
            add_noise = self.training
        if add_noise and self.input_noise_sigma > 0:
            eps = torch.randn_like(s) * self.input_noise_sigma
            s_in = s + eps
        else:
            s_in = s
        return self.encoder(s_in)

    def reconstruct(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode_latent(s, add_noise=True)
        s_hat = self.decoder(z)
        return z, s_hat

    def forward(self, s: torch.Tensor, add_noise: bool | None = None) -> torch.Tensor:
        z = self.encode_latent(s, add_noise=add_noise)
        return self.context(z)

    def reconstruction_loss(
        self,
        s: torch.Tensor,
        weight_decay: float = 1e-4,
    ) -> torch.Tensor:
        _, s_hat = self.reconstruct(s)
        rec = torch.mean((s - s_hat).pow(2))
        if weight_decay > 0:
            l2 = 0.0
            for p in list(self.encoder.parameters()) + list(self.decoder.parameters()):
                l2 = l2 + p.pow(2).sum()
            rec = rec + weight_decay * l2
        return rec
