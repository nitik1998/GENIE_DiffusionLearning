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


class Block(nn.Module):
    def __init__(self, dim: int, time_emb_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        time_scale = self.time_mlp(t_emb)
        return x + self.mlp(x + time_scale)


class LatentDenoiser(nn.Module):
    """
    MLP denoiser for latent-space DDPM.
    Injects time embeddings at every residual block.
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
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.proj_in = nn.Linear(latent_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            Block(hidden_dim, time_emb_dim) for _ in range(n_layers)
        ])
        
        self.proj_out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        x = self.proj_in(z_noisy)
        
        for block in self.blocks:
            x = block(x, t_emb)
            
        return self.proj_out(x)
