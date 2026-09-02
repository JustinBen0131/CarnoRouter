from __future__ import annotations

import hashlib
import hmac

from .domain import RouteDecision


def redacted_decision_event(decision: RouteDecision) -> dict[str, object]:
    """Build unlinkable telemetry without request or prompt fingerprints."""

    probability = decision.strong_needed_probability
    probability_bucket = (
        None if probability is None else min(int(probability * 10), 9) / 10
    )
    length_bucket = min(decision.prompt_length // 128, 16) * 128
    return {
        "schema": "CarnoRedactedDecisionEventV2",
        "selected_model": decision.selected_model,
        "reason": decision.reason.value,
        "abstained": decision.abstained,
        "strong_needed_probability_bucket_low": probability_bucket,
        "prompt_length_bucket_low": length_bucket,
        "pricing_snapshot_id": decision.pricing_snapshot_id,
        "policy_id": decision.policy_id,
        "policy_version": decision.policy_version,
        "evaluation_receipt_sha256": decision.evaluation_receipt_sha256,
    }


def correlated_decision_event(
    decision: RouteDecision, *, telemetry_key: bytes
) -> dict[str, object]:
    """Add key-scoped HMAC identifiers for private incident correlation."""

    if len(telemetry_key) < 32:
        raise ValueError("telemetry_key must contain at least 32 bytes")
    payload = redacted_decision_event(decision)
    payload["schema"] = "CarnoCorrelatedDecisionEventV1"
    payload["request_hmac_sha256"] = hmac.new(
        telemetry_key, decision.request_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    payload["prompt_hmac_sha256"] = hmac.new(
        telemetry_key, decision.prompt_sha256.encode("ascii"), hashlib.sha256
    ).hexdigest()
    return payload
