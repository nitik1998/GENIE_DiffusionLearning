import torch
import torch.nn as nn
import torch.nn.functional as F

class Sampling(nn.Module):
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std



class ConvAutoEncoder(nn.Module):
    """
    Simple convolutional autoencoder matching the Common Task 1 notebook
    baseline architecture.
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.decoder(z)
        return x_hat[:, :, :125, :125].contiguous()

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        dummy = torch.zeros((x.size(0), 1), device=x.device, dtype=x.dtype)
        return x_hat, z, dummy

    def reconstruct(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        del use_mean
        return self.decode(self.encode(x))

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

class DeepFalconVAE(nn.Module):
    """
    VAE architecture matching the Evaluation Test DeepFalcon notebook.
    """

    def __init__(self, in_channels: int = 3, latent_dim: int = 256, input_size: tuple[int, int] = (125, 125)) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        self.encoder_output_size = self._get_encoder_output_size(in_channels, input_size)
        self.flatten_size = int(torch.tensor(self.encoder_output_size).prod().item())

        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def _get_encoder_output_size(self, in_channels: int, input_size: tuple[int, int]) -> tuple[int, int, int]:
        del in_channels
        h, w = input_size
        for _ in range(5):
            h = (h + 2 - 4) // 2 + 1
            w = (w + 2 - 4) // 2 + 1
        return (512, h, w)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = x.reshape(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = torch.clamp(self.fc_logvar(x), min=-20.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, logvar, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)
        x = x.reshape(x.size(0), *self.encoder_output_size)
        x = self.decoder(x)
        x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar, z = self.encode(x)
        return self.decode(z), mu, logvar

    def reconstruct(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        mu, _, z = self.encode(x)
        latent = mu if use_mean else z
        return self.decode(latent)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _, _ = self.encode(x)
        return mu

