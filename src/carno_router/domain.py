from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256


class DecisionReason(str, Enum):
    THRESHOLD_CHEAP = "threshold_cheap"
    THRESHOLD_STRONG = "threshold_strong"
    BUDGET_DOWNGRADE = "budget_downgrade"
    EXPLICIT_FALLBACK = "explicit_fallback"
    INVALID_SCORE = "invalid_score"
    BUDGET_EXHAUSTED = "budget_exhausted"


@dataclass(frozen=True)
class RouteRequest:
    request_id: str
    prompt: str
    strong_needed_probability: float | None
    input_tokens: int = 0
    expected_output_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("request_id must be non-empty")
        if not isinstance(self.prompt, str):
            raise TypeError("prompt must be text")
        if self.input_tokens < 0 or self.expected_output_tokens < 0:
            raise ValueError("token counts must be non-negative")

    @property
    def prompt_sha256(self) -> str:
        return sha256(self.prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RouteDecision:
    request_id: str
    selected_model: str | None
    reason: DecisionReason
    strong_needed_probability: float | None
    threshold: float
    estimated_cost: str | None
    pricing_snapshot_id: str
    prompt_sha256: str
    prompt_length: int
    policy_id: str
    policy_version: str
    evaluation_receipt_sha256: str | None
    budget_violation_action: str

    @property
    def abstained(self) -> bool:
        return self.selected_model is None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "CarnoRouteDecisionV1",
            "request_id": self.request_id,
            "selected_model": self.selected_model,
            "reason": self.reason.value,
            "abstained": self.abstained,
            "strong_needed_probability": self.strong_needed_probability,
            "threshold": self.threshold,
            "estimated_cost": self.estimated_cost,
            "pricing_snapshot_id": self.pricing_snapshot_id,
            "prompt_sha256": self.prompt_sha256,
            "prompt_length": self.prompt_length,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "evaluation_receipt_sha256": self.evaluation_receipt_sha256,
            "budget_violation_action": self.budget_violation_action,
        }
