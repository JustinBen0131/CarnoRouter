from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from carno_router.research_eval import content_sha256, load_traces, reference_suite
from carno_router.research_stress import (
    MUTATION_FAMILIES,
    StressConfig,
    build_blinded_campaign,
    build_human_adjudication_packet,
    evaluate_blinded_campaign,
)
from carno_router.research_stress_reporting import write_stress_artifacts


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ResearchStressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = reference_suite()
        cls.config = StressConfig(replicates=2)
        cls.traces, cls.truth, cls.blind_manifest = build_blinded_campaign(cls.suite, cls.config)
        cls.records, cls.evaluation = evaluate_blinded_campaign(
            cls.suite,
            cls.traces,
            cls.truth,
            cls.config,
        )

    def test_campaign_count_is_cartesian_product(self) -> None:
        expected = len(self.suite.tasks) * len(MUTATION_FAMILIES) * self.config.replicates
        self.assertEqual(len(self.records), expected)
        self.assertEqual(len({record["candidate_id"] for record in self.records}), expected)
        self.assertEqual(len({record["receipt_sha256"] for record in self.records}), expected)

    def test_blind_manifest_does_not_expose_mutation_labels(self) -> None:
        serialized = json.dumps(self.blind_manifest, sort_keys=True)
        self.assertNotIn("mutation_family\"", serialized)
        self.assertTrue(self.blind_manifest["blinding"]["mutation_labels_available_to_grader"] is False)

    def test_strict_policy_rejects_all_mutations_and_accepts_all_controls(self) -> None:
        strict = self.evaluation["strict_contract_policy"]
        self.assertEqual(strict["false_accepts"], 0)
        self.assertEqual(strict["false_rejects"], 0)
        self.assertEqual(self.evaluation["stress_evaluator_status"], "PASS")
        self.assertTrue(all(self.evaluation["gates"].values()))
        self.assertTrue(
            all(record["result"]["target_failure_surface_detected"] for record in self.records)
        )

    def test_legacy_policy_blind_spots_are_preserved_as_hardening_evidence(self) -> None:
        legacy = self.evaluation["legacy_score_policy"]
        self.assertGreater(legacy["mutation_false_accepts"], 0)
        self.assertIn("tool_budget_overrun", legacy["blind_spot_families"])
        self.assertIn("unsupported_citation", legacy["blind_spot_families"])

    def test_accounting_is_exact_and_provider_execution_absent(self) -> None:
        accounting = self.evaluation["accounting"]
        self.assertTrue(accounting["complete"])
        self.assertEqual(accounting["actual_provider_execution"], "NOT_RUN")
        self.assertEqual(
            accounting["synthetic_input_tokens"] + accounting["synthetic_output_tokens"],
            accounting["synthetic_total_tokens"],
        )
        self.assertTrue(
            all(record["accounting"]["actual_provider"]["status"] == "NOT_RUN" for record in self.records)
        )

    def test_simulated_raters_are_explicitly_not_humans(self) -> None:
        raters = self.evaluation["simulated_raters"]
        self.assertEqual(raters["rater_count"], 5)
        self.assertEqual(raters["trial_votes"], len(self.records) * 5)
        self.assertEqual(raters["human_grader_agreement"], "NOT_RUN")
        self.assertLess(raters["unanimous_rate"], 1.0)

    def test_human_packet_is_blinded_and_unlabeled(self) -> None:
        packet, answer_key = build_human_adjudication_packet(self.suite, self.traces, self.truth)
        self.assertEqual(packet["status"], "READY_NOT_RUN")
        self.assertEqual(packet["candidate_count"], len(self.suite.tasks) * len(MUTATION_FAMILIES))
        self.assertTrue(all(row["review_label"] is None for row in packet["candidates"]))
        self.assertEqual(answer_key["status"], "SEALED_NOT_HUMAN_LABELS")

    def test_post_seal_trace_swap_is_refused_before_grading(self) -> None:
        control = next(
            trace for trace in self.traces if self.truth[trace.trace_id]["mutation_family"] == "reference_control"
        )
        task = next(task for task in self.suite.tasks if task.task_id == control.task_id)
        swapped = replace(control, actions=(task.forbidden_actions[0],))
        tampered = tuple(swapped if trace.trace_id == control.trace_id else trace for trace in self.traces)
        with self.assertRaisesRegex(ValueError, "sealed trace hash"):
            evaluate_blinded_campaign(self.suite, tampered, self.truth, self.config)

    def test_sealed_identity_drift_is_refused_before_grading(self) -> None:
        first = self.traces[0]
        for field, value, message in (
            ("suite_id", "other-suite", "suite identity"),
            ("seed", first.seed + 1, "seed differs"),
        ):
            drifted = tuple(replace(first, **{field: value}) if trace is first else trace for trace in self.traces)
            with self.assertRaisesRegex(ValueError, message):
                evaluate_blinded_campaign(self.suite, drifted, self.truth, self.config)

    def test_candidate_traces_are_persisted_in_blind_order_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as output:
            manifest = write_stress_artifacts(
                output,
                self.suite,
                self.traces,
                self.truth,
                self.blind_manifest,
                self.records,
                self.evaluation,
            )
            names = [item["path"] for item in manifest["artifacts"]]  # type: ignore[union-attr]
            self.assertIn("candidate_traces.jsonl", names)
            self.assertEqual(manifest["artifact_count"], 14)
            persisted = load_traces(str(Path(output) / "candidate_traces.jsonl"))
            self.assertEqual(len(persisted), len(self.traces))
            for index, (trace, row) in enumerate(zip(persisted, self.blind_manifest["candidates"])):  # type: ignore[arg-type]
                self.assertEqual(row["blind_index"], index)
                self.assertEqual(row["candidate_id"], trace.trace_id)
                self.assertEqual(row["trace_sha256"], content_sha256(trace.to_dict()))

    def test_deterministic_core_artifacts_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            write_stress_artifacts(
                first,
                self.suite,
                self.traces,
                self.truth,
                self.blind_manifest,
                self.records,
                self.evaluation,
            )
            reversed_traces = tuple(reversed(self.traces))
            reversed_records = list(reversed(self.records))
            write_stress_artifacts(
                second,
                self.suite,
                reversed_traces,
                self.truth,
                self.blind_manifest,
                reversed_records,
                self.evaluation,
            )
            # The writer preserves already-sealed blind order; replay uses the same order.
            write_stress_artifacts(
                second,
                self.suite,
                self.traces,
                self.truth,
                self.blind_manifest,
                self.records,
                self.evaluation,
            )
            self.assertEqual(_tree_hashes(Path(first)), _tree_hashes(Path(second)))


if __name__ == "__main__":
    unittest.main()
