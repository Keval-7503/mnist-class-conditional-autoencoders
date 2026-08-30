import unittest

import torch

from mnist_autoencoder.data import stratified_split_indices


class DataTests(unittest.TestCase):
    def test_stratified_split_is_seeded_disjoint_and_balanced(self) -> None:
        labels = torch.tensor([0] * 20 + [1] * 20 + [2] * 20)

        train_a, validation_a = stratified_split_indices(labels, 0.2, seed=7)
        train_b, validation_b = stratified_split_indices(labels, 0.2, seed=7)

        self.assertEqual(train_a, train_b)
        self.assertEqual(validation_a, validation_b)
        self.assertTrue(set(train_a).isdisjoint(validation_a))
        self.assertEqual(len(train_a) + len(validation_a), len(labels))
        self.assertEqual(torch.bincount(labels[validation_a]).tolist(), [4, 4, 4])

    def test_split_respects_candidate_indices(self) -> None:
        labels = torch.tensor([0] * 10 + [1] * 10)
        candidates = list(range(10, 20))

        train, validation = stratified_split_indices(labels, 0.2, 11, candidates)

        self.assertEqual(set(train + validation), set(candidates))
        self.assertTrue(all(labels[index].item() == 1 for index in train + validation))


if __name__ == "__main__":
    unittest.main()
