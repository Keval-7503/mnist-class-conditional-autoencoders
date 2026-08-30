from __future__ import annotations

import unittest

import torch

from mnist_autoencoder.visualize import one_example_per_digit


class VisualizationTests(unittest.TestCase):
    def test_selects_one_example_per_digit_in_numeric_order(self) -> None:
        dataset = [(torch.full((1, 2, 2), float(digit)), digit) for digit in reversed(range(10))]

        examples = one_example_per_digit(dataset)

        self.assertEqual(tuple(examples.shape), (10, 1, 2, 2))
        self.assertEqual(examples[:, 0, 0, 0].tolist(), [float(i) for i in range(10)])

    def test_rejects_dataset_with_missing_digit(self) -> None:
        dataset = [(torch.zeros(1, 2, 2), digit) for digit in range(9)]

        with self.assertRaisesRegex(ValueError, "missing digits"):
            one_example_per_digit(dataset)


if __name__ == "__main__":
    unittest.main()
