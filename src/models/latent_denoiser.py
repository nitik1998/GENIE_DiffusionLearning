"""
latent_denoiser.py — MLP Denoiser for Latent Diffusion
=======================================================
Time-conditioned MLP that denoises latent vectors for the DDPM scheduler.
Used in Task 3 latent_diffusion mode.
"""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class LatentDenoiser(nn.Module):
    """
    MLP denoiser for latent-space DDPM.

    Takes a noisy latent vector z_t and timestep t, predicts the noise.
    Architecture: [latent_dim + time_dim] → hidden → hidden → latent_dim
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        time_emb_dim: int = 128,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        layers = []
        in_dim = latent_dim + time_emb_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        x = torch.cat([z_noisy, t_emb], dim=-1)
        return self.net(x)
