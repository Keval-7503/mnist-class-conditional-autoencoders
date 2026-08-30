"""Run a pre-specified exact-epoch sweep for unified and specialist autoencoders."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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

from .analysis import bootstrap_mean_ci, per_example_mse
from .benchmark import _git_revision, _sha256_files, parameter_count
from .data import LoaderBundle, build_mnist_loaders
from .model import AutoencoderConfig, ConvAutoencoder
from .training import _train_epoch, reconstruction_mse, set_reproducibility

CONDITIONS = ("unified", "full_specialists", "budget_specialists")
COLORS = {
    "unified": "#2563eb",
    "full_specialists": "#f59e0b",
    "budget_specialists": "#14b8a6",
}
LABELS = {
    "unified": "Unified",
    "full_specialists": "Full specialists",
    "budget_specialists": "Budget-matched specialists",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/epoch_sweep"))
    parser.add_argument(
        "--reference-benchmark",
        type=Path,
        default=Path("results/benchmark.json"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[12, 30, 60, 120])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore and replace an existing partial result.",
    )
    return parser


def validate_design(seeds: list[int], checkpoints: list[int]) -> tuple[list[int], list[int]]:
    """Validate and normalize the pre-specified replication design."""

    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be non-empty and unique")
    if not checkpoints or any(epoch < 1 for epoch in checkpoints):
        raise ValueError("checkpoints must contain positive epochs")
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("checkpoints must be unique")
    return list(seeds), sorted(checkpoints)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _model_key(seed: int, condition: str, digit: int | None = None) -> str:
    suffix = "all" if digit is None else str(digit)
    return f"{seed}:{condition}:{suffix}"


def _train_trajectory(
    loaders: LoaderBundle,
    config: AutoencoderConfig,
    model_seed: int,
    checkpoints: list[int],
    args: argparse.Namespace,
    device: torch.device,
    include_labels: bool,
) -> dict[str, Any]:
    set_reproducibility(model_seed, deterministic=True)
    model = ConvAutoencoder(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    checkpoint_set = set(checkpoints)
    observations: dict[str, Any] = {}
    started = time.perf_counter()

    for epoch in range(1, checkpoints[-1] + 1):
        train_mse = _train_epoch(model, loaders.train, optimizer, device)
        # Match the original benchmark trajectory exactly: constructing each
        # validation iterator advances PyTorch's RNG stream before the next
        # epoch's dropout masks are sampled.
        validation_mse = reconstruction_mse(model, loaders.validation, device)
        if epoch not in checkpoint_set:
            continue
        errors, labels = per_example_mse(model, loaders.test, device)
        observation: dict[str, Any] = {
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "test_mse": float(errors.mean()),
            "errors": errors.tolist(),
        }
        if include_labels:
            observation["labels"] = labels.tolist()
        observations[str(epoch)] = observation
        print(
            f"    epoch={epoch} validation_mse={validation_mse:.6f} test_mse={errors.mean():.6f}",
            flush=True,
        )

    return {
        "model_seed": model_seed,
        "config": asdict(config),
        "parameters": parameter_count(model),
        "counts": loaders.counts,
        "training_seconds": time.perf_counter() - started,
        "checkpoints": observations,
    }


def _protocol_signature(
    args: argparse.Namespace,
    seeds: list[int],
    checkpoints: list[int],
    full_config: AutoencoderConfig,
    budget_config: AutoencoderConfig,
) -> dict[str, Any]:
    return {
        "seeds": seeds,
        "checkpoints": checkpoints,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "validation_fraction": args.validation_fraction,
        "full_config": asdict(full_config),
        "budget_config": asdict(budget_config),
        "data_dir": str(args.data_dir),
    }


def _load_progress(
    path: Path,
    signature: dict[str, Any],
    fresh: bool,
) -> dict[str, Any]:
    if fresh or not path.exists():
        return {"signature": signature, "models": {}}
    progress = json.loads(path.read_text(encoding="utf-8"))
    if progress.get("signature") != signature:
        raise ValueError(
            "existing partial result has a different protocol; use --fresh "
            "or another output directory"
        )
    return progress


def _paired_summary(
    unified: np.ndarray,
    specialist: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    if unified.shape != specialist.shape:
        raise RuntimeError("paired error arrays are not aligned")
    differences = specialist - unified
    return {
        "test_mse": float(specialist.mean()),
        "paired_delta_vs_unified": float(differences.mean()),
        "paired_delta_95_ci": list(bootstrap_mean_ci(differences, resamples=resamples, seed=seed)),
        "fraction_lower_error": float(np.mean(differences < 0)),
    }


def aggregate_models(
    models: dict[str, Any],
    seeds: list[int],
    checkpoints: list[int],
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Aggregate flat resumable model records into checkpoint-level results."""

    by_checkpoint: dict[str, Any] = {}
    for epoch in checkpoints:
        epoch_key = str(epoch)
        per_seed: list[dict[str, Any]] = []
        for seed in seeds:
            unified_record = models[_model_key(seed, "unified")]
            unified_point = unified_record["checkpoints"][epoch_key]
            unified_errors = np.asarray(unified_point["errors"], dtype=np.float64)
            labels = np.asarray(unified_point["labels"], dtype=np.int64)
            seed_result: dict[str, Any] = {
                "seed": seed,
                "unified": {
                    "test_mse": float(unified_errors.mean()),
                    "validation_mse": unified_point["validation_mse"],
                },
            }
            for condition_index, condition in enumerate(CONDITIONS[1:], start=1):
                specialist_parts: list[np.ndarray] = []
                matched_parts: list[np.ndarray] = []
                by_digit: dict[str, Any] = {}
                for digit in range(10):
                    record = models[_model_key(seed, condition, digit)]
                    point = record["checkpoints"][epoch_key]
                    errors = np.asarray(point["errors"], dtype=np.float64)
                    matched = unified_errors[labels == digit]
                    summary = _paired_summary(
                        matched,
                        errors,
                        resamples=bootstrap_resamples,
                        seed=seed * 1000 + condition_index * 100 + digit,
                    )
                    summary["validation_mse"] = point["validation_mse"]
                    by_digit[str(digit)] = summary
                    specialist_parts.append(errors)
                    matched_parts.append(matched)
                specialist_errors = np.concatenate(specialist_parts)
                matched_unified = np.concatenate(matched_parts)
                seed_result[condition] = {
                    **_paired_summary(
                        matched_unified,
                        specialist_errors,
                        resamples=bootstrap_resamples,
                        seed=seed * 100 + condition_index,
                    ),
                    "by_digit": by_digit,
                }
            per_seed.append(seed_result)

        aggregate: dict[str, Any] = {}
        for condition in CONDITIONS:
            values = np.asarray([row[condition]["test_mse"] for row in per_seed])
            aggregate[condition] = {
                "test_mse_mean": float(values.mean()),
                "test_mse_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
            if condition == "unified":
                continue
            deltas = np.asarray([row[condition]["paired_delta_vs_unified"] for row in per_seed])
            aggregate[condition].update(
                {
                    "paired_delta_mean": float(deltas.mean()),
                    "paired_delta_sd": (float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0),
                    "paired_delta_seed_95_ci": (
                        list(
                            bootstrap_mean_ci(
                                deltas,
                                resamples=bootstrap_resamples,
                                seed=epoch * 10 + CONDITIONS.index(condition),
                            )
                        )
                        if len(deltas) > 1
                        else [float(deltas[0]), float(deltas[0])]
                    ),
                    "seeds_improved": int(np.sum(deltas < 0)),
                    "seed_count": len(deltas),
                }
            )
        by_checkpoint[epoch_key] = {
            "per_seed": per_seed,
            "aggregate": aggregate,
        }
    return by_checkpoint


def supported_crossover(
    by_checkpoint: dict[str, Any],
    checkpoints: list[int],
    condition: str,
) -> int | None:
    """Return the first checkpoint with replicated, interval-supported improvement."""

    for epoch in checkpoints:
        summary = by_checkpoint[str(epoch)]["aggregate"][condition]
        upper = summary["paired_delta_seed_95_ci"][1]
        if summary["seeds_improved"] == summary["seed_count"] and upper < 0:
            return epoch
    return None


def _write_figure(payload: dict[str, Any], output_dir: Path) -> None:
    checkpoints = payload["protocol"]["checkpoints"]
    by_checkpoint = payload["by_checkpoint"]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for condition in CONDITIONS:
        means = [
            by_checkpoint[str(epoch)]["aggregate"][condition]["test_mse_mean"]
            for epoch in checkpoints
        ]
        sds = [
            by_checkpoint[str(epoch)]["aggregate"][condition]["test_mse_sd"]
            for epoch in checkpoints
        ]
        lower = np.asarray(means) - np.asarray(sds)
        upper = np.asarray(means) + np.asarray(sds)
        axes[0].plot(
            checkpoints,
            means,
            marker="o",
            linewidth=2,
            color=COLORS[condition],
            label=LABELS[condition],
        )
        axes[0].fill_between(checkpoints, lower, upper, color=COLORS[condition], alpha=0.14)

    references = payload["reference_baselines"]
    axes[0].axhline(
        references["pca"]["test_mse"],
        color="#7c3aed",
        linestyle="--",
        linewidth=1.5,
        label="PCA (64 components)",
    )
    axes[0].axhline(
        references["global_mean"]["test_mse"],
        color="#64748b",
        linestyle=":",
        linewidth=1.5,
        label="Global mean",
    )
    axes[0].set_xscale("log")
    axes[0].set_xticks(checkpoints, [str(epoch) for epoch in checkpoints])
    axes[0].set_xlabel("Completed training epochs")
    axes[0].set_ylabel("Canonical test MSE (lower is better)")
    axes[0].set_title("Reconstruction performance by training budget")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    for condition in CONDITIONS[1:]:
        means = [
            by_checkpoint[str(epoch)]["aggregate"][condition]["paired_delta_mean"]
            for epoch in checkpoints
        ]
        sds = [
            by_checkpoint[str(epoch)]["aggregate"][condition]["paired_delta_sd"]
            for epoch in checkpoints
        ]
        axes[1].errorbar(
            checkpoints,
            means,
            yerr=sds,
            marker="o",
            capsize=4,
            linewidth=2,
            color=COLORS[condition],
            label=LABELS[condition],
        )
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xticks(checkpoints, [str(epoch) for epoch in checkpoints])
    axes[1].set_xlabel("Completed training epochs")
    axes[1].set_ylabel("Specialist minus unified paired MSE")
    axes[1].set_title("Does specialization cross below parity?")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    figure.suptitle("Pre-specified MNIST epoch-budget sweep", fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "epoch_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def _write_report(payload: dict[str, Any], output_dir: Path) -> None:
    checkpoints = payload["protocol"]["checkpoints"]
    by_checkpoint = payload["by_checkpoint"]
    rows: list[str] = []
    for epoch in checkpoints:
        aggregate = by_checkpoint[str(epoch)]["aggregate"]
        for condition in CONDITIONS:
            summary = aggregate[condition]
            delta = (
                "-"
                if condition == "unified"
                else f"{summary['paired_delta_mean']:.6f} +/- {summary['paired_delta_sd']:.6f}"
            )
            rows.append(
                f"| {epoch} | {LABELS[condition]} | "
                f"{summary['test_mse_mean']:.6f} +/- {summary['test_mse_sd']:.6f} | "
                f"{delta} |"
            )

    crossover_lines: list[str] = []
    for condition in CONDITIONS[1:]:
        crossover = supported_crossover(by_checkpoint, checkpoints, condition)
        if crossover is None:
            crossover_lines.append(
                f"- {LABELS[condition]}: no replicated, interval-supported crossover "
                "at the pre-specified checkpoints."
            )
        else:
            crossover_lines.append(
                f"- {LABELS[condition]}: first supported crossover at epoch {crossover}."
            )

    report = f"""# Epoch-Budget Follow-up

## Question

Does the ranking between one unified autoencoder and ten oracle-routed specialist autoencoders
change as the fixed training budget increases?

This follow-up evaluates exact checkpoints at epochs {", ".join(map(str, checkpoints))}. Every
checkpoint comes from one uninterrupted deterministic trajectory per model. The test set was
evaluated only after the checkpoint schedule and decision rule were fixed.

![Epoch-budget sweep](epoch_sweep.png)

## Results

| Epoch | Condition | Test MSE | Paired delta vs. unified |
|---:|---|---:|---:|
{chr(10).join(rows)}

Positive paired deltas mean specialists are worse. Values are mean +/- sample standard deviation
across seeds {", ".join(map(str, payload["protocol"]["seeds"]))}.

## Crossover decision

A supported crossover requires all seeds to favor the specialist condition and the upper bound of
the across-seed bootstrap interval for specialist-minus-unified MSE to be below zero.

{chr(10).join(crossover_lines)}

## Interpretation guardrails

- The epoch checkpoints were pre-specified before running this follow-up.
- Checkpoints are exact training budgets, not validation-selected best epochs.
- Repeated epochs expose specialists to the same class examples again; they do not create new data.
- Oracle routing, MNIST scope, reconstruction-MSE limitations, and three-seed uncertainty remain.
- A crossover is an interaction with training budget, not evidence that specialization is
  universally superior.
"""
    (output_dir / "RESULTS.md").write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    seeds, checkpoints = validate_design(args.seeds, args.checkpoints)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / "epoch_sweep.partial.json"

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

    signature = _protocol_signature(
        args,
        seeds,
        checkpoints,
        full_config,
        budget_config,
    )
    progress = _load_progress(partial_path, signature, args.fresh)
    models: dict[str, Any] = progress["models"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in seeds:
        unified_key = _model_key(seed, "unified")
        if unified_key not in models:
            print(f"unified: seed={seed}", flush=True)
            loaders = build_mnist_loaders(
                args.data_dir,
                batch_size=args.batch_size,
                validation_fraction=args.validation_fraction,
                seed=seed,
                num_workers=args.num_workers,
                download=not args.no_download,
            )
            models[unified_key] = _train_trajectory(
                loaders,
                full_config,
                seed,
                checkpoints,
                args,
                device,
                include_labels=True,
            )
            _atomic_json(partial_path, progress)

        for condition, config in (
            ("full_specialists", full_config),
            ("budget_specialists", budget_config),
        ):
            for digit in range(10):
                key = _model_key(seed, condition, digit)
                if key in models:
                    continue
                print(
                    f"{LABELS[condition]}: seed={seed} digit={digit}",
                    flush=True,
                )
                loaders = build_mnist_loaders(
                    args.data_dir,
                    batch_size=args.batch_size,
                    validation_fraction=args.validation_fraction,
                    seed=seed,
                    digit=digit,
                    num_workers=args.num_workers,
                    download=not args.no_download,
                )
                models[key] = _train_trajectory(
                    loaders,
                    config,
                    seed + digit,
                    checkpoints,
                    args,
                    device,
                    include_labels=False,
                )
                _atomic_json(partial_path, progress)

    reference_bytes = args.reference_benchmark.read_bytes()
    reference = json.loads(reference_bytes)
    by_checkpoint = aggregate_models(
        models,
        seeds,
        checkpoints,
        args.bootstrap_resamples,
    )
    payload = {
        "study": "Exact-epoch specialization crossover follow-up",
        "protocol": {
            **signature,
            "output_dir": str(args.output_dir),
            "reference_benchmark": str(args.reference_benchmark),
            "bootstrap_resamples": args.bootstrap_resamples,
            "estimand": "performance after exactly each pre-specified number of training passes",
            "crossover_rule": (
                "all seeds favor specialists and the upper across-seed bootstrap "
                "bound is below zero"
            ),
            "test_access": "checkpoint schedule and decision rule fixed before test evaluation",
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
            "reference_benchmark_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        },
        "reference_baselines": reference["baselines"],
        "models": models,
        "by_checkpoint": by_checkpoint,
        "supported_crossover": {
            condition: supported_crossover(by_checkpoint, checkpoints, condition)
            for condition in CONDITIONS[1:]
        },
    }
    _atomic_json(args.output_dir / "epoch_sweep.json", payload)
    _write_figure(payload, args.output_dir)
    _write_report(payload, args.output_dir)
    partial_path.unlink(missing_ok=True)
    print(json.dumps(payload["supported_crossover"], indent=2), flush=True)


if __name__ == "__main__":
    main()
