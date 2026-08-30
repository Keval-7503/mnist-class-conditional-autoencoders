"""Training and evaluation utilities with complete, sample-weighted metrics."""

from __future__ import annotations

import copy
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import ConvAutoencoder


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_mse: float
    validation_mse: float


@dataclass(frozen=True)
class FitResult:
    history: list[EpochMetrics]
    best_epoch: int
    best_validation_mse: float

    def to_dict(self) -> dict[str, object]:
        return {
            "history": [asdict(item) for item in self.history],
            "best_epoch": self.best_epoch,
            "best_validation_mse": self.best_validation_mse,
        }


def set_reproducibility(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch and optionally request deterministic kernels."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _images_from_batch(batch: object) -> torch.Tensor:
    if isinstance(batch, tuple | list):
        return batch[0]
    if isinstance(batch, torch.Tensor):
        return batch
    raise TypeError("expected a tensor batch or (images, labels) batch")


def reconstruction_mse(
    model: nn.Module,
    batches: Iterable[object],
    device: torch.device,
) -> float:
    """Compute pixel MSE over every element in every batch."""

    model.eval()
    squared_error = 0.0
    element_count = 0
    with torch.no_grad():
        for batch in batches:
            images = _images_from_batch(batch).to(device)
            output = model(images)
            reconstructions = output[0] if isinstance(output, tuple) else output
            squared_error += torch.sum((reconstructions - images) ** 2).item()
            element_count += images.numel()
    if element_count == 0:
        raise ValueError("cannot evaluate an empty data loader")
    return squared_error / element_count


def _train_epoch(
    model: nn.Module,
    batches: Iterable[object],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    squared_error = 0.0
    element_count = 0
    for batch in batches:
        images = _images_from_batch(batch).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        reconstructions = output[0] if isinstance(output, tuple) else output
        loss = torch.mean((reconstructions - images) ** 2)
        loss.backward()
        optimizer.step()
        squared_error += torch.sum((reconstructions.detach() - images) ** 2).item()
        element_count += images.numel()
    if element_count == 0:
        raise ValueError("cannot train on an empty data loader")
    return squared_error / element_count


def fit(
    model: nn.Module,
    train_batches: Iterable[object],
    validation_batches: Iterable[object],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    min_delta: float = 0.0,
) -> FitResult:
    """Train with validation-based early stopping and restore the best state."""

    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")

    model.to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        train_mse = _train_epoch(model, train_batches, optimizer, device)
        validation_mse = reconstruction_mse(model, validation_batches, device)
        history.append(EpochMetrics(epoch, train_mse, validation_mse))

        if validation_mse < best_validation - min_delta:
            best_validation = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return FitResult(history, best_epoch, best_validation)


def save_checkpoint(
    path: str | Path,
    model: ConvAutoencoder,
    metadata: dict[str, object],
) -> None:
    """Save weights plus enough primitive metadata to recreate the architecture."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model.config.to_dict(),
            "metadata": metadata,
        },
        destination,
    )
