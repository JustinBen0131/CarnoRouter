from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re


@dataclass(frozen=True)
class RoutingPolicy:
    strong_threshold: float
    max_cost_per_request: Decimal
    cheap_model: str = "cheap"
    strong_model: str = "strong"
    invalid_score_fallback: str | None = None
    budget_violation_action: str = "abstain"
    policy_id: str = "unversioned"
    policy_version: str = "0"
    evaluation_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.strong_threshold <= 1.0:
            raise ValueError("strong_threshold must be in [0,1]")
        if self.max_cost_per_request < 0:
            raise ValueError("max_cost_per_request must be non-negative")
        if self.invalid_score_fallback not in {None, self.cheap_model, self.strong_model}:
            raise ValueError("fallback must be absent, cheap_model, or strong_model")
        if self.cheap_model == self.strong_model:
            raise ValueError("cheap and strong model identifiers must differ")
        if self.budget_violation_action not in {"abstain", "downgrade"}:
            raise ValueError("budget_violation_action must be abstain or downgrade")
        if not self.policy_id.strip() or not self.policy_version.strip():
            raise ValueError("policy identity and version must be non-empty")
        if self.evaluation_receipt_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.evaluation_receipt_sha256
        ):
            raise ValueError("evaluation receipt must be a lowercase SHA-256")
