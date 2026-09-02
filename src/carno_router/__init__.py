"""Offline-first cost-quality routing and evaluation."""

from .domain import DecisionReason, RouteDecision, RouteRequest
from .policy import RoutingPolicy
from .pricing import PricingSnapshot
from .routing import route_request
from .research_eval import (
    AgentTrace,
    EvaluationConfig,
    ResearchSuite,
    ResearchTask,
    TraceStep,
    evaluate_suite,
    grade_trace,
)

__all__ = [
    "DecisionReason",
    "AgentTrace",
    "EvaluationConfig",
    "PricingSnapshot",
    "ResearchSuite",
    "ResearchTask",
    "RouteDecision",
    "RouteRequest",
    "RoutingPolicy",
    "TraceStep",
    "evaluate_suite",
    "grade_trace",
    "route_request",
]

__version__ = "0.3.0"
