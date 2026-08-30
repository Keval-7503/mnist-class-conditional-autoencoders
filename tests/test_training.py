import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_autoencoder.training import reconstruction_mse


class IdentityModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images


class ZeroModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(images)


class TrainingTests(unittest.TestCase):
    def test_reconstruction_mse_uses_all_elements(self) -> None:
        images = torch.tensor([[[[1.0]]], [[[3.0]]], [[[5.0]]]])
        labels = torch.zeros(3, dtype=torch.long)
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)

        self.assertEqual(reconstruction_mse(IdentityModel(), loader, torch.device("cpu")), 0.0)
        self.assertAlmostEqual(
            reconstruction_mse(ZeroModel(), loader, torch.device("cpu")), 35.0 / 3.0
        )


if __name__ == "__main__":
    unittest.main()
