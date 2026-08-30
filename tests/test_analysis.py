"""Tests for paired metrics and uncertainty helpers."""

import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mnist_autoencoder.analysis import bootstrap_mean_ci, per_example_mse
from mnist_autoencoder.benchmark import parameter_count
from mnist_autoencoder.model import AutoencoderConfig, ConvAutoencoder


class IdentityModel(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images


class AnalysisTests(unittest.TestCase):
    def test_per_example_mse_preserves_labels_and_order(self) -> None:
        images = torch.tensor([[[[0.0]]], [[[1.0]]], [[[0.5]]]])
        labels = torch.tensor([2, 4, 6])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

        errors, observed_labels = per_example_mse(IdentityModel(), loader, torch.device("cpu"))

        np.testing.assert_allclose(errors, np.zeros(3))
        np.testing.assert_array_equal(observed_labels, labels.numpy())

    def test_bootstrap_interval_is_seeded_and_contains_mean(self) -> None:
        values = np.arange(1.0, 11.0)
        first = bootstrap_mean_ci(values, resamples=1_000, seed=7)
        second = bootstrap_mean_ci(values, resamples=1_000, seed=7)

        self.assertEqual(first, second)
        self.assertLess(first[0], values.mean())
        self.assertGreater(first[1], values.mean())

    def test_specialist_budget_is_within_five_percent(self) -> None:
        unified = parameter_count(
            ConvAutoencoder(AutoencoderConfig(base_channels=8, hidden_dim=64))
        )
        specialists = 10 * parameter_count(
            ConvAutoencoder(AutoencoderConfig(base_channels=3, hidden_dim=14))
        )

        self.assertLess(abs(specialists - unified) / unified, 0.05)


if __name__ == "__main__":
    unittest.main()
