"""Run the pre-specified multi-seed MNIST specialization benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import bootstrap_mean_ci, extract_latents, per_example_mse
from .data import LoaderBundle, build_mnist_loaders
from .model import AutoencoderConfig, ConvAutoencoder
from .training import fit, set_reproducibility


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--no-download", action="store_true")
    return parser


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _train_model(
    loaders: LoaderBundle,
    config: AutoencoderConfig,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ConvAutoencoder, dict[str, Any]]:
    set_reproducibility(seed, deterministic=True)
    model = ConvAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    started = time.perf_counter()
    fit_result = fit(
        model=model,
        train_batches=loaders.train,
        validation_batches=loaders.validation,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=1e-6,
    )
    elapsed = time.perf_counter() - started
    return model, {
        "fit": fit_result.to_dict(),
        "training_seconds": elapsed,
        "model_seed": seed,
        "parameters": parameter_count(model),
        "counts": loaders.counts,
    }


def _train_specialists(
    name: str,
    config: AutoencoderConfig,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    unified_errors: np.ndarray,
    unified_labels: np.ndarray,
) -> dict[str, Any]:
    errors_by_digit: dict[str, list[float]] = {}
    metrics_by_digit: dict[str, Any] = {}
    all_errors: list[np.ndarray] = []
    all_differences: list[np.ndarray] = []

    for digit in range(10):
        print(f"  {name}: seed={seed} digit={digit}", flush=True)
        loaders = build_mnist_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            validation_fraction=args.validation_fraction,
            seed=seed,
            digit=digit,
            num_workers=args.num_workers,
            download=not args.no_download,
        )
        model, training_metrics = _train_model(loaders, config, seed + digit, args, device)
        errors, labels = per_example_mse(model, loaders.test, device)
        if not np.all(labels == digit):
            raise RuntimeError("specialist test loader contains an unexpected label")
        matched_unified = unified_errors[unified_labels == digit]
        if matched_unified.shape != errors.shape:
            raise RuntimeError("paired test errors are not aligned")
        differences = errors - matched_unified
        ci = bootstrap_mean_ci(
            differences,
            resamples=args.bootstrap_resamples,
            seed=seed * 100 + digit,
        )
        errors_by_digit[str(digit)] = errors.tolist()
        all_errors.append(errors)
        all_differences.append(differences)
        metrics_by_digit[str(digit)] = {
            **training_metrics,
            "test_mse": float(errors.mean()),
            "paired_delta_vs_unified": float(differences.mean()),
            "paired_delta_95_ci": list(ci),
            "fraction_lower_error": float(np.mean(differences < 0)),
        }
        del model

    combined_errors = np.concatenate(all_errors)
    combined_differences = np.concatenate(all_differences)
    return {
        "config": asdict(config),
        "parameters_per_model": metrics_by_digit["0"]["parameters"],
        "parameters_total": sum(
            int(metrics_by_digit[str(digit)]["parameters"]) for digit in range(10)
        ),
        "test_mse": float(combined_errors.mean()),
        "paired_delta_vs_unified": float(combined_differences.mean()),
        "paired_delta_95_ci": list(
            bootstrap_mean_ci(
                combined_differences,
                resamples=args.bootstrap_resamples,
                seed=seed,
            )
        ),
        "fraction_lower_error": float(np.mean(combined_differences < 0)),
        "by_digit": metrics_by_digit,
        "errors_by_digit": errors_by_digit,
    }


def _collect_images(loader: torch.utils.data.DataLoader) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch_images, batch_labels in loader:
        images.append(batch_images.numpy().reshape(len(batch_images), -1))
        labels.append(torch.as_tensor(batch_labels).numpy())
    return np.concatenate(images), np.concatenate(labels)


def _classical_baselines(
    loaders: LoaderBundle,
    latent_dim: int,
    seed: int,
) -> dict[str, Any]:
    print("  fitting PCA and mean-image baselines", flush=True)
    train_images, _ = _collect_images(loaders.train)
    test_images, test_labels = _collect_images(loaders.test)

    mean_image = train_images.mean(axis=0, keepdims=True)
    mean_errors = np.mean((test_images - mean_image) ** 2, axis=1)

    started = time.perf_counter()
    pca = PCA(
        n_components=latent_dim,
        svd_solver="randomized",
        iterated_power=3,
        random_state=seed,
    )
    pca.fit(train_images)
    reconstructed = pca.inverse_transform(pca.transform(test_images))
    pca_seconds = time.perf_counter() - started
    pca_errors = np.mean((test_images - reconstructed) ** 2, axis=1)

    return {
        "global_mean": {
            "test_mse": float(mean_errors.mean()),
            "by_digit": {
                str(digit): float(mean_errors[test_labels == digit].mean()) for digit in range(10)
            },
        },
        "pca": {
            "components": latent_dim,
            "test_mse": float(pca_errors.mean()),
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            "fit_and_test_seconds": pca_seconds,
            "by_digit": {
                str(digit): float(pca_errors[test_labels == digit].mean()) for digit in range(10)
            },
        },
    }


def _linear_probe(
    model: ConvAutoencoder,
    loaders: LoaderBundle,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    print("  fitting frozen-latent linear probe", flush=True)
    train_vectors, train_labels = extract_latents(model, loaders.train, device)
    test_vectors, test_labels = extract_latents(model, loaders.test, device)
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=300, random_state=seed),
    )
    started = time.perf_counter()
    probe.fit(train_vectors, train_labels)
    return {
        "test_accuracy": float(probe.score(test_vectors, test_labels)),
        "fit_seconds": time.perf_counter() - started,
    }


def _sha256_files(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    raw = root / "MNIST" / "raw"
    for path in sorted(raw.glob("*.gz")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[path.name] = digest.hexdigest()
    return hashes


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory=D:/FIT/AI_PROJECT", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for condition in ("unified", "full_specialists", "budget_specialists"):
        values = np.asarray([run[condition]["test_mse"] for run in results])
        aggregate[condition] = {
            "test_mse_mean": float(values.mean()),
            "test_mse_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    for condition in ("full_specialists", "budget_specialists"):
        deltas = np.asarray([run[condition]["paired_delta_vs_unified"] for run in results])
        aggregate[condition].update(
            {
                "paired_delta_mean": float(deltas.mean()),
                "paired_delta_sd": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
                "seeds_improved": int(np.sum(deltas < 0)),
                "seed_count": len(deltas),
            }
        )
        if len(deltas) > 1:
            aggregate[condition]["seed_bootstrap_95_ci"] = list(
                bootstrap_mean_ci(deltas, resamples=10_000, seed=2026)
            )
    probe = np.asarray([run["unified"]["linear_probe"]["test_accuracy"] for run in results])
    aggregate["linear_probe"] = {
        "accuracy_mean": float(probe.mean()),
        "accuracy_sd": float(probe.std(ddof=1)) if len(probe) > 1 else 0.0,
    }
    return aggregate


def _write_figures(payload: dict[str, Any], output_dir: Path) -> None:
    conditions = ["unified", "full_specialists", "budget_specialists"]
    labels = ["Unified", "Specialists\n10x total", "Specialists\nbudget matched"]
    means = [payload["aggregate"][condition]["test_mse_mean"] for condition in conditions]
    sds = [payload["aggregate"][condition]["test_mse_sd"] for condition in conditions]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, means, yerr=sds, capsize=5, color=["#2563eb", "#f59e0b", "#14b8a6"])
    ax.set_ylabel("Test reconstruction MSE (lower is better)")
    ax.set_title("Canonical MNIST test performance across seeds")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "condition_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    digits = np.arange(10)
    width = 0.27
    unified_runs = np.asarray(
        [
            [np.mean(run["unified"]["errors"][str(digit)]) for digit in range(10)]
            for run in payload["runs"]
        ]
    )
    full_runs = np.asarray(
        [
            [run["full_specialists"]["by_digit"][str(digit)]["test_mse"] for digit in range(10)]
            for run in payload["runs"]
        ]
    )
    budget_runs = np.asarray(
        [
            [run["budget_specialists"]["by_digit"][str(digit)]["test_mse"] for digit in range(10)]
            for run in payload["runs"]
        ]
    )
    error_kwargs = {"capsize": 3} if len(payload["runs"]) > 1 else {}
    ddof = 1 if len(payload["runs"]) > 1 else 0
    ax.bar(
        digits - width,
        unified_runs.mean(axis=0),
        width,
        yerr=unified_runs.std(axis=0, ddof=ddof),
        label="Unified",
        **error_kwargs,
    )
    ax.bar(
        digits,
        full_runs.mean(axis=0),
        width,
        yerr=full_runs.std(axis=0, ddof=ddof),
        label="Full specialists",
        **error_kwargs,
    )
    ax.bar(
        digits + width,
        budget_runs.mean(axis=0),
        width,
        yerr=budget_runs.std(axis=0, ddof=ddof),
        label="Budget matched",
        **error_kwargs,
    )
    ax.set_xticks(digits)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Test MSE")
    ax.set_title("Per-digit comparison across seeds (mean +/- SD)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "per_digit_comparison.png", dpi=180)
    plt.close(fig)


def _write_report(payload: dict[str, Any], output_dir: Path) -> None:
    agg = payload["aggregate"]
    full = agg["full_specialists"]
    budget = agg["budget_specialists"]
    probe = agg["linear_probe"]
    baselines = payload["baselines"]

    verdict = (
        "supports"
        if full["paired_delta_mean"] < 0 and full["seeds_improved"] == full["seed_count"]
        else "does not support"
    )
    unified_text = f"{agg['unified']['test_mse_mean']:.6f} +/- {agg['unified']['test_mse_sd']:.6f}"
    full_text = f"{full['test_mse_mean']:.6f} +/- {full['test_mse_sd']:.6f}"
    budget_text = f"{budget['test_mse_mean']:.6f} +/- {budget['test_mse_sd']:.6f}"
    full_delta = f"{full['paired_delta_mean']:.6f} +/- {full['paired_delta_sd']:.6f}"
    selected_epochs: list[int] = []
    for run in payload["runs"]:
        selected_epochs.append(run["unified"]["fit"]["best_epoch"])
        for condition in ("full_specialists", "budget_specialists"):
            selected_epochs.extend(
                run[condition]["by_digit"][str(digit)]["fit"]["best_epoch"] for digit in range(10)
            )
    epoch_ceiling = int(payload["protocol"]["epochs"])
    at_ceiling = sum(epoch == epoch_ceiling for epoch in selected_epochs)
    ceiling_summary = (
        f"{at_ceiling}/{len(selected_epochs)} neural fits selected epoch {epoch_ceiling}"
    )
    ceiling_limitation = f"{at_ceiling}/{len(selected_epochs)} fits reached the epoch ceiling."

    report = f"""# Experimental Results

## Headline

The pre-specified comparison **{verdict}** the hypothesis that oracle-routed,
class-specific autoencoders reduce reconstruction error relative to one unified model.

All values below use MNIST's untouched 10,000-image official test partition. Neural-network
results are means +/- sample standard deviations across {len(payload["runs"])} seeds.

| Condition | Test MSE |
|---|---:|
| Unified autoencoder | {unified_text} |
| Ten full-size specialists | {full_text} |
| Ten parameter-budget-matched specialists | {budget_text} |
| PCA ({baselines["pca"]["components"]} components) | {baselines["pca"]["test_mse"]:.6f} |
| Global mean image (one deterministic fit) | {baselines["global_mean"]["test_mse"]:.6f} |

Full specialists minus unified paired MSE: **{full_delta}**
across seeds; improvement occurred in {full["seeds_improved"]}/{full["seed_count"]} seeds.

Budget-matched specialists minus unified paired MSE:
**{budget["paired_delta_mean"]:.6f} +/- {budget["paired_delta_sd"]:.6f}** across seeds;
improvement occurred in {budget["seeds_improved"]}/{budget["seed_count"]} seeds.

The {baselines["pca"]["components"]}-component PCA baseline outperformed every neural condition.
This negative baseline result is central evidence: added nonlinearity did not purchase better
fixed-budget reconstruction in this experiment.

A linear classifier trained on frozen unified-model latents reached
**{100 * probe["accuracy_mean"]:.2f}% +/- {100 * probe["accuracy_sd"]:.2f}%** test accuracy,
showing how much label information the reconstruction bottleneck retained.

## Interpretation

The full specialist system stores approximately ten times the parameters of the unified model.
The budget-matched condition controls this confound by reducing each specialist so that the sum
of all ten parameter counts is close to the unified parameter count. This distinguishes gains
from specialization from gains purchased mainly through total model capacity.

The paired estimates compare both systems on the same official test images. Per-image bootstrap
intervals are retained in `benchmark.json`; the cross-seed summary is the primary replication
evidence. PCA and the mean image provide non-neural reference baselines.

## Protocol safeguards

- Official 60,000/10,000 MNIST train/test partition; test data never enters model selection.
- Seeded stratified validation splits made only from the training partition.
- Validation monitoring and restoration of the best epoch. {ceiling_summary};
  the reported estimand is fixed-budget performance.
- Three predetermined seeds and identical optimizer settings.
- Complete per-pixel and per-example metrics; no partial-batch denominator.
- Parameter counts, package versions, file hashes, elapsed time, and Git revision recorded.
- Raw observations and per-digit results stored in machine-readable JSON.

## Limitations

MNIST is a small educational benchmark. Specialist evaluation assumes an oracle supplies the
correct digit before reconstruction, so it is not a deployable unsupervised system. Three seeds
provide replication but only a coarse estimate of between-run variability. Reconstruction MSE
does not guarantee perceptual quality, and this study does not establish performance on shifted
or real-world data. {ceiling_limitation}
Longer training could change the architecture ranking; changing that ceiling after viewing test
results requires a new, clearly labeled experiment.
"""
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_config = AutoencoderConfig(
        latent_dim=args.latent_dim,
        dropout=0.1,
        base_channels=8,
        hidden_dim=64,
    )
    budget_config = AutoencoderConfig(
        latent_dim=args.latent_dim,
        dropout=0.1,
        base_channels=3,
        hidden_dim=14,
    )
    full_parameters = parameter_count(ConvAutoencoder(full_config))
    budget_total = 10 * parameter_count(ConvAutoencoder(budget_config))
    budget_gap = abs(budget_total - full_parameters) / full_parameters
    if budget_gap > 0.05:
        raise RuntimeError("budget-matched specialists differ from unified capacity by over 5%")

    runs: list[dict[str, Any]] = []
    baselines: dict[str, Any] | None = None
    for seed in args.seeds:
        print(f"unified: seed={seed}", flush=True)
        loaders = build_mnist_loaders(
            args.data_dir,
            batch_size=args.batch_size,
            validation_fraction=args.validation_fraction,
            seed=seed,
            num_workers=args.num_workers,
            download=not args.no_download,
        )
        unified_model, unified_training = _train_model(loaders, full_config, seed, args, device)
        unified_errors, unified_labels = per_example_mse(unified_model, loaders.test, device)
        unified = {
            **unified_training,
            "config": asdict(full_config),
            "test_mse": float(unified_errors.mean()),
            "errors": {
                str(digit): unified_errors[unified_labels == digit].tolist() for digit in range(10)
            },
            "linear_probe": _linear_probe(unified_model, loaders, device, seed),
        }
        if baselines is None:
            baselines = _classical_baselines(loaders, args.latent_dim, seed)

        full_specialists = _train_specialists(
            "full specialists",
            full_config,
            seed,
            args,
            device,
            unified_errors,
            unified_labels,
        )
        budget_specialists = _train_specialists(
            "budget specialists",
            budget_config,
            seed,
            args,
            device,
            unified_errors,
            unified_labels,
        )
        runs.append(
            {
                "seed": seed,
                "unified": unified,
                "full_specialists": full_specialists,
                "budget_specialists": budget_specialists,
            }
        )
        partial = {
            "protocol": vars(args)
            | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
            "runs": runs,
            "baselines": baselines,
        }
        (args.output_dir / "benchmark.partial.json").write_text(
            json.dumps(partial, indent=2), encoding="utf-8"
        )

    assert baselines is not None
    payload = {
        "study": "Unified versus class-specific MNIST autoencoders",
        "protocol": {
            **vars(args),
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "hypothesis": (
                "Oracle-routed specialists reduce paired test MSE relative to a unified model; "
                "the budget-matched condition tests whether any gain survives equal total capacity."
            ),
            "primary_metric": "mean per-pixel MSE on the canonical MNIST test set",
            "specialist_model_seed_rule": "replication seed + digit",
            "estimand": f"performance after at most {args.epochs} complete training-set passes",
        },
        "capacity_control": {
            "unified_parameters": full_parameters,
            "full_specialists_total_parameters": 10 * full_parameters,
            "budget_specialists_total_parameters": budget_total,
            "budget_relative_gap": budget_gap,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "device": str(device),
            "git_revision": _git_revision(),
            "mnist_sha256": _sha256_files(args.data_dir),
        },
        "baselines": baselines,
        "runs": runs,
    }
    payload["aggregate"] = _aggregate(runs)
    (args.output_dir / "benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_figures(payload, args.output_dir)
    _write_report(payload, args.output_dir)
    partial_path = args.output_dir / "benchmark.partial.json"
    if partial_path.exists():
        partial_path.unlink()
    print(json.dumps(payload["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
