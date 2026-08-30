"""Verify that committed headline metrics are derivable from raw observations."""

import json
import unittest
from pathlib import Path

import numpy as np

RESULT_PATH = Path(__file__).parents[1] / "results" / "benchmark.json"


class ResultArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    def test_protocol_and_capacity_control(self) -> None:
        self.assertEqual([run["seed"] for run in self.payload["runs"]], [11, 22, 33])
        self.assertEqual(self.payload["protocol"]["epochs"], 12)
        self.assertLess(self.payload["capacity_control"]["budget_relative_gap"], 0.05)
        self.assertEqual(
            self.payload["capacity_control"]["full_specialists_total_parameters"],
            10 * self.payload["capacity_control"]["unified_parameters"],
        )

    def test_raw_errors_reproduce_every_run_metric(self) -> None:
        for run in self.payload["runs"]:
            unified = np.concatenate(
                [np.asarray(run["unified"]["errors"][str(digit)]) for digit in range(10)]
            )
            self.assertEqual(unified.size, 10_000)
            self.assertAlmostEqual(float(unified.mean()), run["unified"]["test_mse"], delta=1e-8)

            for condition in ("full_specialists", "budget_specialists"):
                specialist = np.concatenate(
                    [
                        np.asarray(run[condition]["errors_by_digit"][str(digit)])
                        for digit in range(10)
                    ]
                )
                self.assertEqual(specialist.size, 10_000)
                self.assertAlmostEqual(
                    float(specialist.mean()), run[condition]["test_mse"], delta=1e-8
                )

    def test_aggregate_means_and_sample_deviations(self) -> None:
        for condition in ("unified", "full_specialists", "budget_specialists"):
            values = np.asarray([run[condition]["test_mse"] for run in self.payload["runs"]])
            reported = self.payload["aggregate"][condition]
            self.assertAlmostEqual(float(values.mean()), reported["test_mse_mean"], places=12)
            self.assertAlmostEqual(float(values.std(ddof=1)), reported["test_mse_sd"], places=12)

    def test_data_hashes_and_code_revision_are_recorded(self) -> None:
        hashes = self.payload["environment"]["mnist_sha256"]
        self.assertEqual(len(hashes), 4)
        self.assertTrue(all(len(digest) == 64 for digest in hashes.values()))
        self.assertEqual(len(self.payload["environment"]["git_revision"]), 40)


if __name__ == "__main__":
    unittest.main()
