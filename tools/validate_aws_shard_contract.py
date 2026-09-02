#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HANDLER_PATH = ROOT / "cloud" / "aws" / "lambda_function.py"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_handler() -> Any:
    spec = importlib.util.spec_from_file_location("carno_aws_lambda_handler", HANDLER_PATH)
    if not spec or not spec.loader:
        raise RuntimeError("cannot load AWS Lambda handler")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def content_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def shard_for(candidate_id: str, shard_count: int) -> int:
    """Mirror of research_stress._partition_for, kept local so the validator is self-contained."""
    digest = hashlib.sha256(f"carno-aws-shard-v1:{candidate_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def local_receipts(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            candidate_id = str(record["candidate_id"])
            if candidate_id in result:
                raise ValueError("duplicate candidate in local trial file")
            result[candidate_id] = str(record["receipt_sha256"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=ROOT / "build" / "aws" / "carno_the283_lambda.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "aws_lambda_shard_contract_validation.json")
    parser.add_argument("--core-output", type=Path, default=ROOT / "artifacts" / "aws_lambda_shard_contract_core.json")
    args = parser.parse_args()
    plan = json.loads((ROOT / "cloud" / "aws" / "execution_plan.json").read_text(encoding="utf-8"))
    evaluation = json.loads((ROOT / "artifacts" / "research_stress_v1" / "stress_evaluation.json").read_text(encoding="utf-8"))
    local = local_receipts(ROOT / "artifacts" / "research_stress_v1" / "trials.jsonl")
    handler = load_handler()
    shard_count = int(plan["shard_count"])
    started = time.perf_counter_ns()
    assembled: dict[str, str] = {}
    shard_counts: list[int] = []
    shard_hashes: list[str] = []
    partition_statuses: list[str] = []
    computation_scopes: list[str] = []
    duplicate_candidates: list[str] = []
    foreign_candidates: list[str] = []
    body_hash_failures: list[int] = []
    expected_shards: list[dict[str, object]] = []
    for shard_index in range(shard_count):
        response = handler.lambda_handler(
            {
                "schema": "CarnoAwsStressShardRequestV2",
                "shard_index": shard_index,
                "shard_count": shard_count,
                "replicates": 100,
                "root_seed": 2_831_729,
                "canonical_evaluation_content_sha256": evaluation["content_sha256"],
            },
            SimpleNamespace(aws_request_id=None, function_version=None),
        )
        result = response["deterministic"]
        shard_counts.append(int(result["trial_count"]))
        shard_hashes.append(str(result["content_sha256"]))
        partition_statuses.append(str(result["partition_evaluator_status"]))
        computation_scopes.append(str(result["computation_scope"]))
        if result["canonical_evaluation_content_sha256"] != evaluation["content_sha256"]:
            raise ValueError("shard evaluation hash differs from canonical evaluation")
        body = {key: value for key, value in result.items() if key != "content_sha256"}
        if content_sha256(body) != result["content_sha256"]:
            body_hash_failures.append(shard_index)
        for item in result["receipts"]:
            candidate_id = str(item["candidate_id"])
            if shard_for(candidate_id, shard_count) != shard_index:
                foreign_candidates.append(candidate_id)
            if candidate_id in assembled:
                duplicate_candidates.append(candidate_id)
            assembled[candidate_id] = str(item["receipt_sha256"])
        expected_shards.append(
            {
                "shard_index": shard_index,
                "trial_count": int(result["trial_count"]),
                "content_sha256": str(result["content_sha256"]),
                "receipts_sha256": content_sha256(result["receipts"]),
            }
        )
    elapsed_ns = time.perf_counter_ns() - started
    missing = sorted(set(local) - set(assembled))
    extra = sorted(set(assembled) - set(local))
    mismatches = sorted(
        candidate_id
        for candidate_id in set(local) & set(assembled)
        if local[candidate_id] != assembled[candidate_id]
    )
    gates = {
        "plan_below_user_cap": float(plan["modeled_total_upper_usd"]) < float(plan["user_absolute_cap_usd"]),
        "exact_shard_count": len(shard_counts) == shard_count == 48,
        "all_shards_nonempty": all(count > 0 for count in shard_counts),
        "no_duplicate_candidates": not duplicate_candidates,
        "no_missing_candidates": not missing,
        "no_extra_candidates": not extra,
        "all_receipts_match": not mismatches,
        "exact_trial_union": len(assembled) == len(local) == 28_800,
        "every_shard_body_hash_recomputes": not body_hash_failures,
        "every_receipt_belongs_to_its_shard": not foreign_candidates,
        "all_shard_payload_hashes_distinct": len(set(shard_hashes)) == shard_count,
        "every_partition_evaluates_only_its_selected_traces": all(
            scope == "selected_partition_traces_only" for scope in computation_scopes
        ),
        "every_partition_passes": all(status == "PASS" for status in partition_statuses),
        "canonical_evaluation_pass": evaluation["stress_evaluator_status"] == "PASS",
        "human_gate_not_run": evaluation["human_grader_agreement"] == "NOT_RUN",
        "provider_gate_not_run": evaluation["actual_provider_accounting"] == "NOT_RUN",
    }
    core: dict[str, object] = {
        "schema": "CarnoAwsLambdaShardContractCoreV2",
        "status": "PASS_LOCAL_CONTRACT_ONLY" if all(gates.values()) else "FAIL",
        "cloud_execution": "NOT_RUN",
        "package": {
            "format": "deterministic_zip",
            "bytes": args.package.stat().st_size,
            "sha256": sha256_file(args.package),
        },
        "plan_sha256": sha256_file(ROOT / "cloud" / "aws" / "execution_plan.json"),
        "canonical_evaluation_content_sha256": evaluation["content_sha256"],
        "local_trial_count": len(local),
        "reassembled_trial_count": len(assembled),
        "shard_count": shard_count,
        "minimum_shard_trials": min(shard_counts),
        "maximum_shard_trials": max(shard_counts),
        "unique_shard_payload_hashes": len(set(shard_hashes)),
        "partition_computation_scope": sorted(set(computation_scopes)),
        "shard_assignment": "sha256('carno-aws-shard-v1:' + candidate_id) first 8 bytes big-endian modulo shard_count",
        "union_receipts_sha256": content_sha256(sorted(assembled.items())),
        "expected_shards": expected_shards,
        "missing_candidates": missing[:20],
        "extra_candidates": extra[:20],
        "receipt_mismatches": mismatches[:20],
        "duplicate_candidates": duplicate_candidates[:20],
        "foreign_candidates": foreign_candidates[:20],
        "cost_envelope": {
            "modeled_total_upper_usd": plan["modeled_total_upper_usd"],
            "user_absolute_cap_usd": plan["user_absolute_cap_usd"],
            "headroom_usd": plan["headroom_to_user_cap_usd"],
            "actual_aws_cost_usd": None,
        },
        "gates": gates,
        "external_state_changed": False,
        "boundary": (
            "This validates package and 48-shard reassembly logic locally and freezes the per-shard hashes a cloud run "
            "must reproduce (verify with tools/verify_aws_shard_responses.py). It is not evidence of AWS execution or cloud billing."
        ),
    }
    core["content_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.core_output.parent.mkdir(parents=True, exist_ok=True)
    args.core_output.write_text(json.dumps(core, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt: dict[str, object] = {
        "schema": "CarnoAwsLambdaShardContractExecutionReceiptV2",
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": core["status"],
        "cloud_execution": "NOT_RUN",
        "aws_sso": "LOGIN_REQUIRED",
        "governed_mutation_route": "NOT_BOUND",
        "local_simulation_elapsed_ns": elapsed_ns,
        "package_path": str(args.package.resolve()),
        "core_path": str(args.core_output.resolve()),
        "core_content_sha256": core["content_sha256"],
        "external_state_changed": False,
        "boundary": "Runtime receipt for local shard-contract validation only; no AWS request was made.",
    }
    receipt["content_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "cloud_execution": receipt["cloud_execution"],
                "shards": shard_count,
                "reassembled_trials": len(assembled),
                "receipt_mismatches": len(mismatches),
                "modeled_total_upper_usd": plan["modeled_total_upper_usd"],
                "core_content_sha256": core["content_sha256"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
