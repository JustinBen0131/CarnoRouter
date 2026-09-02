from __future__ import annotations

import math

from .domain import DecisionReason, RouteDecision, RouteRequest
from .policy import RoutingPolicy
from .pricing import PricingSnapshot


def _valid_probability(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and 0.0 <= value <= 1.0


def _cost(
    pricing: PricingSnapshot, request: RouteRequest, model_id: str
) -> str:
    return format(
        pricing.estimate(
            model_id, request.input_tokens, request.expected_output_tokens
        ),
        "f",
    )


def route_request(
    request: RouteRequest,
    policy: RoutingPolicy,
    pricing: PricingSnapshot,
) -> RouteDecision:
    """Return a pure routing decision; never invoke a provider."""

    score = request.strong_needed_probability
    selected: str | None
    reason: DecisionReason

    if not _valid_probability(score):
        selected = policy.invalid_score_fallback
        reason = (
            DecisionReason.EXPLICIT_FALLBACK
            if selected is not None
            else DecisionReason.INVALID_SCORE
        )
    elif score >= policy.strong_threshold:
        selected = policy.strong_model
        reason = DecisionReason.THRESHOLD_STRONG
    else:
        selected = policy.cheap_model
        reason = DecisionReason.THRESHOLD_CHEAP

    estimated = None
    if selected is not None:
        selected_cost = pricing.estimate(
            selected, request.input_tokens, request.expected_output_tokens
        )
        if selected_cost > policy.max_cost_per_request:
            cheap_cost = pricing.estimate(
                policy.cheap_model,
                request.input_tokens,
                request.expected_output_tokens,
            )
            if (
                policy.budget_violation_action == "downgrade"
                and selected != policy.cheap_model
                and cheap_cost <= policy.max_cost_per_request
            ):
                selected = policy.cheap_model
                selected_cost = cheap_cost
                reason = DecisionReason.BUDGET_DOWNGRADE
            else:
                selected = None
                reason = DecisionReason.BUDGET_EXHAUSTED
        if selected is not None:
            estimated = format(selected_cost, "f")

    return RouteDecision(
        request_id=request.request_id,
        selected_model=selected,
        reason=reason,
        strong_needed_probability=score if _valid_probability(score) else None,
        threshold=policy.strong_threshold,
        estimated_cost=estimated,
        pricing_snapshot_id=pricing.snapshot_id,
        prompt_sha256=request.prompt_sha256,
        prompt_length=len(request.prompt),
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        evaluation_receipt_sha256=policy.evaluation_receipt_sha256,
        budget_violation_action=policy.budget_violation_action,
    )
