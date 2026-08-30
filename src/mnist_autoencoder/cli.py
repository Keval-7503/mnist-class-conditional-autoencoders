"""Command-line entry point for a traceable MNIST experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .analysis import mse_by_digit
from .data import build_mnist_loaders
from .model import AutoencoderConfig, ConvAutoencoder
from .training import fit, reconstruction_mse, save_checkpoint, set_reproducibility


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/unified"))
    parser.add_argument(
        "--digit",
        type=int,
        choices=range(10),
        default=None,
        help="Train one class-conditional model; omit for a shared all-digit encoder.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    set_reproducibility(args.seed, deterministic=args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_mnist_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        digit=args.digit,
        num_workers=args.num_workers,
        download=not args.no_download,
    )
    model = ConvAutoencoder(
        AutoencoderConfig(
            latent_dim=args.latent_dim,
            dropout=args.dropout,
            base_channels=args.base_channels,
            hidden_dim=args.hidden_dim,
        )
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    result = fit(
        model=model,
        train_batches=loaders.train,
        validation_batches=loaders.validation,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
    )
    test_mse = reconstruction_mse(model, loaders.test, device)

    metrics: dict[str, object] = {
        "experiment": "unified" if args.digit is None else f"digit_{args.digit}",
        "seed": args.seed,
        "device": str(device),
        "counts": loaders.counts,
        "test_mse": test_mse,
        "fit": result.to_dict(),
    }
    if args.digit is None:
        metrics["test_mse_by_digit"] = mse_by_digit(model, loaders.test, device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.output_dir / "model.pt", model, metrics)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
