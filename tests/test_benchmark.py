from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from carno_router.evaluation import BenchmarkConfig, run_benchmark
from carno_router.bundle import load_model_bundle, score_prompt
from carno_router.reporting import write_benchmark_artifacts


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTest(unittest.TestCase):
    def test_benchmark_is_deterministic_and_leak_free(self) -> None:
        config = BenchmarkConfig(
            seed=1729,
            bootstrap_replicates=100,
            selection_bootstrap_replicates=200,
        )
        arguments = (
            str(ROOT / "tests" / "fixtures" / "synthetic_battles.csv"),
            str(ROOT / "configs" / "normalized_costs.json"),
            config,
        )
        first = run_benchmark(*arguments)
        second = run_benchmark(*arguments)
        self.assertEqual(first["report"], second["report"])
        report = first["report"]
        self.assertFalse(report["dataset"]["raw_prompts_in_report"])
        self.assertFalse(report["threshold_selection"]["test_data_used_for_selection"])
        self.assertFalse(report["model"]["selection_receipt"]["test_data_used"])
        self.assertFalse(report["model"]["selection_receipt"]["calibration_data_used"])
        self.assertEqual(report["split"]["group_overlap"], 0)
        self.assertTrue(
            report["threshold_selection"]["selected"]["risk_constraint_pass"]
        )
        self.assertEqual(report["deployment_decision"]["live_autorouting"], "BLOCKED")
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_benchmark_artifacts(directory, first)
            self.assertIn("report.json", manifest["artifacts"])
            text = (Path(directory) / "report.json").read_text(encoding="utf-8")
            self.assertNotIn("Rewrite this short note", text)
            bundle = load_model_bundle(Path(directory) / "model_bundle.json")
            probability = score_prompt(bundle, "A new prompt for offline scoring")
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
