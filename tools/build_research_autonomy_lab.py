#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from carno_router.research_eval import EvaluationConfig, evaluate_suite, reference_suite, sentinel_traces
from carno_router.research_reporting import write_research_eval_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/research_autonomy_v1"))
    args = parser.parse_args()
    suite = reference_suite()
    traces = sentinel_traces(suite)
    evaluation = evaluate_suite(suite, traces, EvaluationConfig())
    manifest = write_research_eval_artifacts(args.output, suite, traces, evaluation)
    print(
        json.dumps(
            {
                "schema": "CarnoResearchAutonomyBuildReceiptV1",
                "status": evaluation["evaluator_validation"],
                "live_agent_deployment": evaluation["live_agent_deployment"],
                "output": str(args.output.resolve()),
                "manifest_content_sha256": manifest["content_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if evaluation["evaluator_validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
