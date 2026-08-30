"""MNIST loading that preserves the canonical test set and a seeded validation split."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    validation: DataLoader
    test: DataLoader
    counts: dict[str, int]


def stratified_split_indices(
    labels: Sequence[int] | torch.Tensor,
    validation_fraction: float,
    seed: int,
    candidate_indices: Sequence[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Create deterministic, disjoint train/validation indices for every label."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    label_array = np.asarray(labels, dtype=np.int64)
    candidates = np.asarray(
        candidate_indices if candidate_indices is not None else np.arange(len(label_array)),
        dtype=np.int64,
    )
    if candidates.size == 0:
        raise ValueError("candidate_indices cannot be empty")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []

    for label in np.unique(label_array[candidates]):
        group = candidates[label_array[candidates] == label].copy()
        rng.shuffle(group)
        validation_count = max(1, int(round(len(group) * validation_fraction)))
        validation_count = min(validation_count, len(group) - 1)
        validation_indices.extend(group[:validation_count].tolist())
        train_indices.extend(group[validation_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _indices_for_digit(labels: torch.Tensor, digit: int | None) -> list[int]:
    if digit is None:
        return list(range(len(labels)))
    if not 0 <= digit <= 9:
        raise ValueError("digit must be in the range 0..9")
    return torch.where(labels == digit)[0].tolist()


def build_mnist_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    validation_fraction: float = 0.1,
    seed: int = 42,
    digit: int | None = None,
    num_workers: int = 0,
    download: bool = True,
) -> LoaderBundle:
    """Build loaders without mixing the official MNIST train and test partitions."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    transform = transforms.ToTensor()
    root = str(Path(data_dir))
    full_train = datasets.MNIST(root=root, train=True, download=download, transform=transform)
    full_test = datasets.MNIST(root=root, train=False, download=download, transform=transform)

    candidates = _indices_for_digit(full_train.targets, digit)
    train_indices, validation_indices = stratified_split_indices(
        full_train.targets,
        validation_fraction=validation_fraction,
        seed=seed,
        candidate_indices=candidates,
    )
    test_indices = _indices_for_digit(full_test.targets, digit)

    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "worker_init_fn": _seed_worker,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        Subset(full_train, train_indices),
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        Subset(full_train, validation_indices), shuffle=False, **common
    )
    test_loader = DataLoader(Subset(full_test, test_indices), shuffle=False, **common)

    return LoaderBundle(
        train=train_loader,
        validation=validation_loader,
        test=test_loader,
        counts={
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
    )


def dataset_targets(dataset: Dataset) -> torch.Tensor:
    """Return labels from an MNIST dataset or nested Subset."""

    if isinstance(dataset, Subset):
        parent = dataset_targets(dataset.dataset)
        return parent[torch.as_tensor(dataset.indices)]
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise TypeError("dataset does not expose a targets attribute")
    return torch.as_tensor(targets)
