"""Create a one-example-per-digit reconstruction grid from a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .model import AutoencoderConfig, ConvAutoencoder


def one_example_per_digit(dataset: Dataset[tuple[torch.Tensor, int]]) -> torch.Tensor:
    """Return the first canonical test example for each digit, ordered 0 through 9."""

    examples: dict[int, torch.Tensor] = {}
    for image, label in dataset:
        digit = int(label)
        if digit not in examples:
            examples[digit] = image
        if len(examples) == 10:
            break
    missing = sorted(set(range(10)) - examples.keys())
    if missing:
        raise ValueError(f"dataset is missing digits: {missing}")
    return torch.stack([examples[digit] for digit in range(10)])


def load_model(checkpoint_path: Path, device: torch.device) -> ConvAutoencoder:
    """Restore an autoencoder and its saved architecture configuration."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = ConvAutoencoder(AutoencoderConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def save_reconstruction_grid(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    output_path: Path,
) -> None:
    """Save ten originals and ten reconstructions as a compact 2-by-10 figure."""

    figure, axes = plt.subplots(2, 10, figsize=(15, 3.4))
    for digit in range(10):
        axes[0, digit].imshow(originals[digit, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[1, digit].imshow(reconstructions[digit, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        axes[0, digit].set_title(str(digit), fontsize=12, fontweight="bold")
        axes[0, digit].axis("off")
        axes[1, digit].axis("off")
    figure.text(0.012, 0.64, "Original", rotation=90, va="center", fontsize=12)
    figure.text(0.012, 0.25, "Reconstructed", rotation=90, va="center", fontsize=12)
    figure.suptitle(
        "Unified autoencoder: one canonical test example per digit",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.02, 0, 1, 0.91))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/reconstruction_examples.png"),
    )
    parser.add_argument("--no-download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.MNIST(
        root=args.data_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=not args.no_download,
    )
    originals = one_example_per_digit(dataset).to(device)
    model = load_model(args.checkpoint, device)
    with torch.inference_mode():
        reconstructions, _ = model(originals)
    save_reconstruction_grid(originals, reconstructions, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
