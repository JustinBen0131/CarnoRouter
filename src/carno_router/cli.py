from __future__ import annotations

import argparse
import json
from decimal import Decimal

from .bundle import load_model_bundle, score_prompt
from .domain import RouteRequest
from .evaluation import BenchmarkConfig, run_benchmark
from .policy import RoutingPolicy
from .pricing import PricingSnapshot
from .reporting import write_benchmark_artifacts
from .research_eval import EvaluationConfig, evaluate_suite, load_suite, load_traces
from .research_reporting import write_research_eval_artifacts
from .routing import route_request


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="carno-router",
        description="Offline-first cost-quality routing benchmark",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    benchmark = commands.add_parser("benchmark", help="run a deterministic local benchmark")
    benchmark.add_argument("--data", required=True)
    benchmark.add_argument("--pricing", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--seed", type=int, default=1729)
    benchmark.add_argument("--bootstrap", type=int, default=500)
    benchmark.add_argument("--max-quality-drop", type=float, default=0.02)
    benchmark.add_argument("--train-fraction", type=float, default=0.70)
    benchmark.add_argument("--calibration-fraction", type=float, default=0.15)
    benchmark.add_argument("--selection-bootstrap", type=int, default=1000)

    score = commands.add_parser(
        "score", help="score one prompt from a sealed offline bundle"
    )
    score.add_argument("--bundle", required=True)
    score.add_argument("--prompt", required=True)
    score.add_argument(
        "--policy",
        choices=("risk_controlled", "point_estimate", "safety_recall_95", "balanced_accuracy"),
        default="risk_controlled",
    )

    route = commands.add_parser("route", help="make one pure decision without provider execution")
    route.add_argument("--request-id", required=True)
    route.add_argument("--prompt", required=True)
    route.add_argument("--score", type=float)
    route.add_argument("--threshold", type=float, default=0.5)
    route.add_argument("--max-cost", default="20")
    route.add_argument("--pricing", required=True)
    route.add_argument("--input-tokens", type=int, default=0)
    route.add_argument("--output-tokens", type=int, default=0)
    route.add_argument("--fallback", choices=("cheap", "strong"))
    route.add_argument(
        "--budget-action", choices=("abstain", "downgrade"), default="abstain"
    )
    research_eval = commands.add_parser(
        "research-eval",
        help="grade deterministic research-agent traces without provider execution",
    )
    research_eval.add_argument("--suite", required=True)
    research_eval.add_argument("--traces", required=True)
    research_eval.add_argument("--output", required=True)
    research_eval.add_argument("--min-tasks", type=int, default=24)
    research_eval.add_argument("--min-mean-score", type=float, default=0.95)
    research_eval.add_argument("--min-pass-rate", type=float, default=0.95)
    research_eval.add_argument("--max-ece", type=float, default=0.08)
    research_eval.add_argument("--max-critical-upper", type=float, default=0.15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "benchmark":
        run = run_benchmark(
            args.data,
            args.pricing,
            BenchmarkConfig(
                seed=args.seed,
                bootstrap_replicates=args.bootstrap,
                max_quality_drop=args.max_quality_drop,
                train_fraction=args.train_fraction,
                calibration_fraction=args.calibration_fraction,
                selection_bootstrap_replicates=args.selection_bootstrap,
            ),
        )
        manifest = write_benchmark_artifacts(args.output, run)
        print(json.dumps({"status": "PASS", "output": args.output, "manifest": manifest}, sort_keys=True))
        return 0
    if args.command == "score":
        bundle = load_model_bundle(args.bundle)
        probability = score_prompt(bundle, args.prompt)
        thresholds = bundle.get("thresholds")
        if not isinstance(thresholds, dict):
            raise ValueError("bundle thresholds are invalid")
        threshold = float(thresholds[args.policy])
        print(
            json.dumps(
                {
                    "schema": "CarnoOfflineScoreV1",
                    "bundle_content_sha256": bundle["content_sha256"],
                    "policy": args.policy,
                    "strong_needed_probability": probability,
                    "threshold": threshold,
                    "selected_tier": "strong" if probability >= threshold else "cheap",
                    "provider_execution": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "route":
        pricing = PricingSnapshot.load(args.pricing)
        decision = route_request(
            RouteRequest(
                request_id=args.request_id,
                prompt=args.prompt,
                strong_needed_probability=args.score,
                input_tokens=args.input_tokens,
                expected_output_tokens=args.output_tokens,
            ),
            RoutingPolicy(
                strong_threshold=args.threshold,
                max_cost_per_request=Decimal(args.max_cost),
                invalid_score_fallback=args.fallback,
                budget_violation_action=args.budget_action,
            ),
            pricing,
        )
        print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "research-eval":
        suite = load_suite(args.suite)
        traces = load_traces(args.traces)
        evaluation = evaluate_suite(
            suite,
            traces,
            EvaluationConfig(
                min_tasks=args.min_tasks,
                min_mean_score=args.min_mean_score,
                min_pass_rate=args.min_pass_rate,
                max_ece=args.max_ece,
                max_critical_violation_upper_95=args.max_critical_upper,
            ),
        )
        manifest = write_research_eval_artifacts(args.output, suite, traces, evaluation)
        print(
            json.dumps(
                {
                    "status": evaluation["evaluator_validation"],
                    "output": args.output,
                    "manifest": manifest,
                    "live_agent_deployment": evaluation["live_agent_deployment"],
                },
                sort_keys=True,
            )
        )
        return 0 if evaluation["evaluator_validation"] == "PASS" else 1
    raise AssertionError("unreachable command")
