"""Analysis helpers that keep cross-class comparisons scientifically well-defined."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch import nn


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
