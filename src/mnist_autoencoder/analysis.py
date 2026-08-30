"""Analysis helpers that keep cross-class comparisons scientifically well-defined."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn


def per_example_mse(
    model: nn.Module,
    batches: Iterable[object],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one reconstruction MSE and label per image for paired analysis."""

    model.eval()
    errors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            if not isinstance(batch, tuple | list) or len(batch) < 2:
                raise TypeError("per-example evaluation requires (images, labels) batches")
            images = batch[0].to(device)
            output = model(images)
            reconstructions = output[0] if isinstance(output, tuple) else output
            batch_errors = torch.mean((reconstructions - images) ** 2, dim=(1, 2, 3))
            errors.append(batch_errors.cpu().numpy())
            labels.append(torch.as_tensor(batch[1]).cpu().numpy())
    if not errors:
        raise ValueError("cannot evaluate an empty loader")
    return np.concatenate(errors), np.concatenate(labels)


def bootstrap_mean_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a one-dimensional mean."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("values must contain at least two observations")
    if not 0.0 < confidence < 1.0 or resamples < 100:
        raise ValueError("invalid confidence or resample count")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk = 1_000
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        indices = rng.integers(0, array.size, size=(size, array.size))
        means[start : start + size] = array[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def extract_latents(
    model: nn.Module,
    batches: Iterable[object],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract vectors and labels from one encoder (one coordinate system)."""

    encode = getattr(model, "encode", None)
    if encode is None:
        raise TypeError("model must provide an encode method")

    model.eval()
    vectors: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            if not isinstance(batch, tuple | list) or len(batch) < 2:
                raise TypeError("latent extraction requires (images, labels) batches")
            images, batch_labels = batch[0].to(device), batch[1]
            vectors.append(encode(images).cpu().numpy())
            labels.append(torch.as_tensor(batch_labels).cpu().numpy())
    if not vectors:
        raise ValueError("cannot extract latents from an empty loader")
    return np.concatenate(vectors), np.concatenate(labels)


def project_latents(
    vectors: np.ndarray,
    method: str = "pca",
    seed: int = 42,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Project vectors for visualization; t-SNE is exploratory, not a cluster metric."""

    array = np.asarray(vectors)
    if array.ndim != 2 or len(array) < 3:
        raise ValueError("vectors must be a 2D array with at least three rows")
    if method == "pca":
        return PCA(n_components=2).fit_transform(array)
    if method == "tsne":
        effective_perplexity = min(perplexity, float(len(array) - 1))
        return TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(array)
    raise ValueError("method must be 'pca' or 'tsne'")


def mse_by_digit(
    model: nn.Module,
    batches: Iterable[object],
    device: torch.device,
) -> dict[int, float]:
    """Compute complete reconstruction MSE separately for each ground-truth digit."""

    model.eval()
    squared_error: dict[int, float] = defaultdict(float)
    element_count: dict[int, int] = defaultdict(int)
    with torch.no_grad():
        for batch in batches:
            if not isinstance(batch, tuple | list) or len(batch) < 2:
                raise TypeError("per-digit evaluation requires (images, labels) batches")
            images = batch[0].to(device)
            labels = torch.as_tensor(batch[1], device=device)
            output = model(images)
            reconstructions = output[0] if isinstance(output, tuple) else output
            per_image = torch.sum((reconstructions - images) ** 2, dim=(1, 2, 3))
            elements_per_image = images[0].numel()
            for digit in labels.unique():
                mask = labels == digit
                key = int(digit.item())
                squared_error[key] += per_image[mask].sum().item()
                element_count[key] += int(mask.sum().item()) * elements_per_image
    return {digit: squared_error[digit] / element_count[digit] for digit in sorted(squared_error)}
