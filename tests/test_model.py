import unittest

import torch

from mnist_autoencoder.model import AutoencoderConfig, ConvAutoencoder


class ModelTests(unittest.TestCase):
    def test_autoencoder_round_trip_shapes(self) -> None:
        model = ConvAutoencoder(AutoencoderConfig(latent_dim=16, dropout=0.0)).eval()
        images = torch.rand(4, 1, 28, 28)

        reconstructions, latents = model(images)

        self.assertEqual(reconstructions.shape, images.shape)
        self.assertEqual(latents.shape, (4, 16))
        self.assertTrue(torch.all((0.0 <= reconstructions) & (reconstructions <= 1.0)))

    def test_config_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            AutoencoderConfig(latent_dim=1)
        with self.assertRaises(ValueError):
            AutoencoderConfig(dropout=1.0)


if __name__ == "__main__":
    unittest.main()
