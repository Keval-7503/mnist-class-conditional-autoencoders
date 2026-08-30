"""Convolutional autoencoder with an explicit, inspectable bottleneck."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AutoencoderConfig:
    """Architecture parameters saved alongside every checkpoint."""

    latent_dim: int = 64
    dropout: float = 0.1
    base_channels: int = 8
    hidden_dim: int = 64

    def __post_init__(self) -> None:
        if self.latent_dim < 2:
            raise ValueError("latent_dim must be at least 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.base_channels < 1:
            raise ValueError("base_channels must be positive")
        if self.hidden_dim < 2:
            raise ValueError("hidden_dim must be at least 2")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ConvAutoencoder(nn.Module):
    """A compact 28x28 autoencoder whose decoder exactly inverts spatial sizes."""

    encoded_channels = 64
    encoded_size = 7

    def __init__(self, config: AutoencoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or AutoencoderConfig()
        self.encoded_channels = 2 * self.config.base_channels
        flat_dim = self.encoded_channels * self.encoded_size * self.encoded_size

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, self.config.base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.config.base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.config.dropout),
            nn.Conv2d(self.config.base_channels, self.encoded_channels, 3, 2, 1),
            nn.BatchNorm2d(self.encoded_channels),
            nn.ReLU(inplace=True),
        )
        self.encoder_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim),
        )
        self.decoder_head = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.hidden_dim, flat_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (self.encoded_channels, self.encoded_size, self.encoded_size)),
            nn.ConvTranspose2d(
                self.encoded_channels, self.config.base_channels, kernel_size=4, stride=2, padding=1
            ),  # 7 -> 14
            nn.BatchNorm2d(self.config.base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.config.base_channels, 1, 4, 2, 1),  # 14 -> 28
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Map a batch of images to a shared latent coordinate system."""

        return self.encoder_head(self.encoder_conv(images))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into normalized grayscale images."""

        return self.decoder_conv(self.decoder_head(latents))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(images)
        return self.decode(latents), latents
