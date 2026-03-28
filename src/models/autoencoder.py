import torch
import torch.nn as nn
import torch.nn.functional as F

class Sampling(nn.Module):
    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class UpsampleConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


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
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = torch.clamp(self.fc_logvar(x), min=-20.0, max=20.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return mu, logvar, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_input(z)
        x = x.view(x.size(0), *self.encoder_output_size)
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

class ConvVAE(nn.Module):
    """
    Convolutional Variational Autoencoder aligned with the current
    Task 1 reference recipe.
    Input: (B, 3, 125, 125) -> Latent -> Output: (B, 3, 125, 125)
    """

    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 32,
        use_transpose_decoder: bool = True,
        variational: bool = True,
        decoder_batchnorm: bool = True,
        output_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_transpose_decoder = use_transpose_decoder
        self.variational = variational
        self.decoder_batchnorm = decoder_batchnorm
        self.output_bias_init = output_bias_init

        # Encoder Blocks
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.encoder_output_size = self._get_encoder_output_size(input_size=(125, 125))
        self.flatten_size = int(torch.tensor(self.encoder_output_size).prod().item())

        self.fc_mu = nn.Linear(self.flatten_size, embedding_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, embedding_dim) if variational else None
        self.sampling = Sampling()

        # Decoder Blocks
        self.decoder_fc = nn.Linear(embedding_dim, self.flatten_size)
        if self.use_transpose_decoder:
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            )
            self.output_conv = None
        else:
            self.decoder_conv = nn.Sequential(
                UpsampleConvBlock(512, 256, use_batchnorm=decoder_batchnorm),
                UpsampleConvBlock(256, 128, use_batchnorm=decoder_batchnorm),
                UpsampleConvBlock(128, 64, use_batchnorm=decoder_batchnorm),
                UpsampleConvBlock(64, 32, use_batchnorm=decoder_batchnorm),
            )
            self.output_conv = nn.Sequential(
                nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        self._init_weights()

    def _get_encoder_output_size(self, input_size: tuple[int, int]) -> tuple[int, int, int]:
        h, w = input_size
        for _ in range(5):
            h = (h + 2 - 4) // 2 + 1
            w = (w + 2 - 4) // 2 + 1
        return (512, h, w)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Small negative bias keeps the sparse output prior without crushing gradients.
        final_conv = None
        if self.output_conv is not None and isinstance(self.output_conv[0], nn.Conv2d):
            final_conv = self.output_conv[0]
        elif isinstance(self.decoder_conv[-2], nn.ConvTranspose2d):
            final_conv = self.decoder_conv[-2]
        if final_conv is not None and final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, self.output_bias_init)

    def encode(self, x: torch.Tensor):
        x = self.encoder_conv(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        if self.variational:
            logvar = torch.clamp(self.fc_logvar(x), -10.0, 10.0)
            z = self.sampling(mu, logvar)
        else:
            logvar = torch.zeros_like(mu)
            z = mu
        return mu, logvar, z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_fc(z)
        x = x.view(x.size(0), *self.encoder_output_size)
        x = self.decoder_conv(x)
        x = F.interpolate(x, size=(125, 125), mode="bilinear", align_corners=False)
        if self.output_conv is not None:
            x = self.output_conv(x)
        else:
            x = torch.sigmoid(x)
        return x

    def forward(self, x: torch.Tensor):
        mu, logvar, z = self.encode(x)
        return self.decode(z), mu, logvar

    def reconstruct(self, x: torch.Tensor, use_mean: bool = True) -> torch.Tensor:
        mu, logvar, z = self.encode(x)
        latent = mu if use_mean else z
        return self.decode(latent)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        mu, _, _ = self.encode(x)
        return mu
