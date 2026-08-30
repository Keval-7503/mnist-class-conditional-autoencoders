from __future__ import annotations

import unittest

from mnist_autoencoder.sweep import (
    aggregate_models,
    supported_crossover,
    validate_design,
)


def _record(checkpoints: dict[str, dict[str, object]]) -> dict[str, object]:
    return {"checkpoints": checkpoints}


class SweepTests(unittest.TestCase):
    def test_design_requires_unique_positive_checkpoints(self) -> None:
        self.assertEqual(validate_design([11, 22], [60, 12]), ([11, 22], [12, 60]))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_design([11, 11], [12, 30])
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_design([11], [0, 12])

    def test_aggregate_detects_only_replicated_supported_crossover(self) -> None:
        seeds = [11, 22, 33]
        checkpoints = [1, 2]
        labels = [digit for digit in range(10) for _ in range(2)]
        unified_errors = [0.10 + 0.001 * digit for digit in range(10) for _ in range(2)]
        models: dict[str, object] = {}

        for seed in seeds:
            models[f"{seed}:unified:all"] = _record(
                {
                    str(epoch): {
                        "errors": unified_errors,
                        "labels": labels,
                        "validation_mse": 0.1,
                    }
                    for epoch in checkpoints
                }
            )
            for condition in ("full_specialists", "budget_specialists"):
                for digit in range(10):
                    base = 0.10 + 0.001 * digit
                    models[f"{seed}:{condition}:{digit}"] = _record(
                        {
                            "1": {
                                "errors": [base + 0.05, base + 0.05],
                                "validation_mse": 0.2,
                            },
                            "2": {
                                "errors": (
                                    [base - 0.02, base - 0.02]
                                    if condition == "full_specialists"
                                    else [base + 0.03, base + 0.03]
                                ),
                                "validation_mse": 0.1,
                            },
                        }
                    )

        aggregated = aggregate_models(
            models,
            seeds,
            checkpoints,
            bootstrap_resamples=100,
        )

        self.assertAlmostEqual(
            aggregated["1"]["aggregate"]["full_specialists"]["paired_delta_mean"],
            0.05,
        )
        self.assertEqual(
            supported_crossover(aggregated, checkpoints, "full_specialists"),
            2,
        )
        self.assertIsNone(supported_crossover(aggregated, checkpoints, "budget_specialists"))


if __name__ == "__main__":
    unittest.main()
