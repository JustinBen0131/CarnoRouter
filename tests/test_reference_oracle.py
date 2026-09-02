from __future__ import annotations

import unittest
from dataclasses import replace

from carno_router.research_eval import grade_trace, reference_suite
from carno_router.research_reference_oracle import (
    DECLARED_NAMES,
    HELD_OUT_NAMES,
    SearchConfig,
    audit_frozen_campaign,
    audit_search,
    canonical_trace,
    compare,
    reference_verdict,
)
from carno_router.research_stress import StressConfig, build_blinded_campaign, evaluate_blinded_campaign


class ReferenceOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.suite = reference_suite()
        cls.config = StressConfig(replicates=2)
        cls.traces, cls.truth, _manifest = build_blinded_campaign(cls.suite, cls.config)
        cls.records, cls.evaluation = evaluate_blinded_campaign(cls.suite, cls.traces, cls.truth, cls.config)

    def test_declared_and_held_out_invariant_sets_are_disjoint_and_complete(self) -> None:
        self.assertEqual(len(DECLARED_NAMES), 7)
        self.assertEqual(len(HELD_OUT_NAMES), 6)
        self.assertFalse(set(DECLARED_NAMES) & set(HELD_OUT_NAMES))

    def test_oracle_agrees_with_sealed_truth_and_production_on_campaign(self) -> None:
        expected = {candidate: bool(row["expected_accept"]) for candidate, row in self.truth.items()}
        production = {str(row["candidate_id"]): bool(row["result"]["strict_contract_accept"]) for row in self.records}  # type: ignore[index]
        audit = audit_frozen_campaign(self.suite, self.traces, expected, production)
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["candidates"], len(self.traces))
        self.assertEqual(audit["oracle_vs_truth_disagreements"], 0)
        self.assertEqual(audit["oracle_vs_production_disagreements"], 0)
        self.assertEqual(audit["control_held_out_violations"], {})
        # The sealed campaign exercises few distinct grader outcomes; the number is recorded, not hidden.
        self.assertLessEqual(int(audit["distinct_declared_status_vectors"]), 16)

    def test_generative_search_finds_no_oracle_disagreement_and_no_declared_false_accept(self) -> None:
        audit = audit_search(self.suite, SearchConfig(attacks_per_task=40))
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual(audit["oracle_production_disagreements"], 0)
        self.assertEqual(audit["declared_violation_false_accepts"], 0)
        self.assertEqual(audit["metamorphic_failures"], [])
        self.assertGreater(int(audit["distinct_declared_status_vectors"]), 16)
        self.assertGreater(int(audit["held_out_only_attacks"]), 0)

    def test_search_audit_is_deterministic(self) -> None:
        first = audit_search(self.suite, SearchConfig(attacks_per_task=10))
        second = audit_search(self.suite, SearchConfig(attacks_per_task=10))
        self.assertEqual(first["content_sha256"], second["content_sha256"])

    def test_exact_rational_legacy_boundary_matches_production_float_boundary(self) -> None:
        task = self.suite.tasks[0]
        base = canonical_trace(self.suite, task, "boundary")
        over_budget = replace(base, tokens=task.max_tokens + 1)
        verdict = reference_verdict(task, over_budget)
        self.assertEqual(verdict["legacy_exact_score"], "9/10")
        self.assertTrue(verdict["legacy_accept"])
        self.assertFalse(verdict["strict_accept"])
        production = grade_trace(task, over_budget)
        self.assertTrue(production["legacy_score_policy_passed"])
        self.assertFalse(production["passed"])

    def test_known_contract_silence_is_reported_as_gap_not_detection(self) -> None:
        # The documented contract says nothing about suite identity at grade time,
        # step/decision consistency, or step-level evidence. These are accepted by the
        # production grader and flagged only by the held-out table. Changing this test
        # means the declared contract has been widened on purpose.
        task = self.suite.tasks[0]
        base = canonical_trace(self.suite, task, "gap")
        drifted = replace(base, suite_id="another-suite")
        result = compare(task, drifted, self.suite.suite_id)
        self.assertEqual(result["disagreements"], [])
        self.assertTrue(result["production"]["strict_accept"])  # type: ignore[index]
        self.assertFalse(result["reference"]["held_out"]["suite_identity"])  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
