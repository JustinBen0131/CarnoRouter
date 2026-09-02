from __future__ import annotations

import json
import unittest
from decimal import Decimal
from pathlib import Path

from carno_router.domain import DecisionReason, RouteRequest
from carno_router.policy import RoutingPolicy
from carno_router.pricing import PricingSnapshot
from carno_router.routing import route_request
from carno_router.telemetry import correlated_decision_event, redacted_decision_event


ROOT = Path(__file__).resolve().parents[1]


class RoutingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pricing = PricingSnapshot.load(ROOT / "configs" / "normalized_costs.json")

    def request(self, score: float | None) -> RouteRequest:
        return RouteRequest("request-1", "private prompt text", score, 10, 20)

    def test_threshold_equality_routes_strong(self) -> None:
        decision = route_request(
            self.request(0.5), RoutingPolicy(0.5, Decimal("20")), self.pricing
        )
        self.assertEqual(decision.selected_model, "strong")
        self.assertEqual(decision.reason, DecisionReason.THRESHOLD_STRONG)

    def test_invalid_score_abstains(self) -> None:
        decision = route_request(
            self.request(float("nan")), RoutingPolicy(0.5, Decimal("20")), self.pricing
        )
        self.assertTrue(decision.abstained)
        self.assertEqual(decision.reason, DecisionReason.INVALID_SCORE)

    def test_explicit_fallback_is_visible(self) -> None:
        decision = route_request(
            self.request(None),
            RoutingPolicy(0.5, Decimal("20"), invalid_score_fallback="cheap"),
            self.pricing,
        )
        self.assertEqual(decision.selected_model, "cheap")
        self.assertEqual(decision.reason, DecisionReason.EXPLICIT_FALLBACK)

    def test_budget_fail_closed_is_default_and_downgrade_is_explicit(self) -> None:
        fail_closed = route_request(
            self.request(0.9), RoutingPolicy(0.5, Decimal("1")), self.pricing
        )
        self.assertTrue(fail_closed.abstained)
        self.assertEqual(fail_closed.reason, DecisionReason.BUDGET_EXHAUSTED)
        downgraded = route_request(
            self.request(0.9),
            RoutingPolicy(
                0.5, Decimal("1"), budget_violation_action="downgrade"
            ),
            self.pricing,
        )
        self.assertEqual(downgraded.selected_model, "cheap")
        self.assertEqual(downgraded.reason, DecisionReason.BUDGET_DOWNGRADE)
        abstained = route_request(
            self.request(0.1), RoutingPolicy(0.5, Decimal("0.5")), self.pricing
        )
        self.assertTrue(abstained.abstained)
        self.assertEqual(abstained.reason, DecisionReason.BUDGET_EXHAUSTED)

    def test_telemetry_never_contains_prompt(self) -> None:
        decision = route_request(
            self.request(0.2), RoutingPolicy(0.5, Decimal("20")), self.pricing
        )
        serialized = json.dumps(redacted_decision_event(decision), sort_keys=True)
        self.assertNotIn("private prompt text", serialized)
        self.assertNotIn(decision.prompt_sha256, serialized)
        self.assertNotIn(decision.request_id, serialized)

    def test_correlated_telemetry_requires_a_key_and_uses_hmac(self) -> None:
        decision = route_request(
            self.request(0.2), RoutingPolicy(0.5, Decimal("20")), self.pricing
        )
        with self.assertRaisesRegex(ValueError, "32 bytes"):
            correlated_decision_event(decision, telemetry_key=b"short")
        payload = correlated_decision_event(decision, telemetry_key=b"x" * 32)
        serialized = json.dumps(payload, sort_keys=True)
        self.assertNotIn(decision.prompt_sha256, serialized)
        self.assertNotIn(decision.request_id, serialized)
        self.assertEqual(len(payload["prompt_hmac_sha256"]), 64)

    def test_threshold_policy_is_monotonic(self) -> None:
        policy = RoutingPolicy(0.5, Decimal("20"))
        decisions = [
            route_request(self.request(score), policy, self.pricing).selected_model
            for score in (0.0, 0.49, 0.5, 0.9, 1.0)
        ]
        self.assertEqual(decisions, ["cheap", "cheap", "strong", "strong", "strong"])


if __name__ == "__main__":
    unittest.main()
