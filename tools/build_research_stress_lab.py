#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from carno_router.research_eval import reference_suite
from carno_router.research_stress import StressConfig, build_blinded_campaign, evaluate_blinded_campaign
from carno_router.research_stress_reporting import write_execution_receipt, write_stress_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the deterministic Carno blinded stress benchmark.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/research_stress_v1"))
    parser.add_argument("--replicates", type=int, default=100)
    parser.add_argument("--root-seed", type=int, default=2_831_729)
    parser.add_argument("--omit-execution-receipt", action="store_true")
    args = parser.parse_args()

    started_ns = time.perf_counter_ns()
    suite = reference_suite()
    config = StressConfig(replicates=args.replicates, root_seed=args.root_seed)
    traces, truth, blind_manifest = build_blinded_campaign(suite, config)
    records, evaluation = evaluate_blinded_campaign(suite, traces, truth, config)
    manifest = write_stress_artifacts(
        args.output,
        suite,
        traces,
        truth,
        blind_manifest,
        records,
        evaluation,
    )
    elapsed_ns = time.perf_counter_ns() - started_ns
    execution_receipt = None
    if not args.omit_execution_receipt:
        execution_receipt = write_execution_receipt(args.output, evaluation, manifest, elapsed_ns)
    receipt = {
        "schema": "CarnoResearchStressBuildReceiptV1",
        "status": evaluation["stress_evaluator_status"],
        "trials": evaluation["trials"],
        "strict_false_accepts": evaluation["strict_contract_policy"]["false_accepts"],
        "strict_false_rejects": evaluation["strict_contract_policy"]["false_rejects"],
        "legacy_false_accepts_exposed": evaluation["legacy_score_policy"]["mutation_false_accepts"],
        "human_grader_agreement": evaluation["human_grader_agreement"],
        "actual_provider_accounting": evaluation["actual_provider_accounting"],
        "live_agent_deployment": evaluation["live_agent_deployment"],
        "output": str(args.output.resolve()),
        "deterministic_manifest_content_sha256": manifest["content_sha256"],
        "execution_receipt_content_sha256": execution_receipt["content_sha256"] if execution_receipt else None,
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if evaluation["stress_evaluator_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

