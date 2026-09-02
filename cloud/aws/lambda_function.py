from __future__ import annotations

import hashlib
import json
from typing import Any

from carno_router.research_eval import reference_suite
from carno_router.research_stress import (
    StressConfig,
    build_blinded_partition,
    evaluate_blinded_partition,
)


def _content_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, object]:
    if event.get("schema") != "CarnoAwsStressShardRequestV2":
        raise ValueError("unsupported shard request schema")
    shard_index = int(event["shard_index"])
    shard_count = int(event["shard_count"])
    replicates = int(event.get("replicates", 100))
    root_seed = int(event.get("root_seed", 2_831_729))
    if not 1 <= shard_count <= 256:
        raise ValueError("shard_count must be in [1,256]")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index is outside shard_count")
    if not 1 <= replicates <= 100:
        raise ValueError("replicates must be in [1,100]")
    canonical_evaluation_sha256 = str(event.get("canonical_evaluation_content_sha256", ""))
    if len(canonical_evaluation_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in canonical_evaluation_sha256
    ):
        raise ValueError("canonical_evaluation_content_sha256 must be a lowercase SHA-256 digest")

    suite = reference_suite()
    config = StressConfig(replicates=replicates, root_seed=root_seed)
    traces, truth, partition = build_blinded_partition(
        suite,
        partition_index=shard_index,
        partition_count=shard_count,
        config=config,
    )
    records = evaluate_blinded_partition(suite, traces, truth, config)
    selected = [
        {
            "candidate_id": str(record["candidate_id"]),
            "receipt_sha256": str(record["receipt_sha256"]),
        }
        for record in records
    ]
    selected.sort(key=lambda item: item["candidate_id"])
    false_accepts = sum(bool(record["result"]["false_accept"]) for record in records)  # type: ignore[index]
    false_rejects = sum(bool(record["result"]["false_reject"]) for record in records)  # type: ignore[index]
    intended_surfaces_detected = all(
        bool(record["result"]["target_failure_surface_detected"]) for record in records  # type: ignore[index]
    )
    partition_passed = false_accepts == 0 and false_rejects == 0 and intended_surfaces_detected
    deterministic: dict[str, object] = {
        "schema": "CarnoAwsStressShardResultV2",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "replicates": replicates,
        "root_seed": root_seed,
        "trial_count": len(selected),
        "full_campaign_trial_count": partition["full_campaign_trial_count"],
        "computation_scope": partition["computation_scope"],
        "identity_scope": partition["identity_scope"],
        "partition_manifest_content_sha256": partition["content_sha256"],
        "canonical_evaluation_content_sha256": canonical_evaluation_sha256,
        "canonical_evaluation_binding": "REQUEST_BOUND_EXPECTED_NOT_RECOMPUTED",
        "partition_evaluator_status": "PASS" if partition_passed else "FAIL",
        "strict_false_accepts": false_accepts,
        "strict_false_rejects": false_rejects,
        "intended_failure_surfaces_detected": intended_surfaces_detected,
        "human_grader_agreement": "NOT_RUN",
        "actual_provider_accounting": "NOT_RUN",
        "receipts": selected,
    }
    deterministic["content_sha256"] = _content_sha256(deterministic)
    return {
        "deterministic": deterministic,
        "runtime": {
            "aws_request_id": getattr(context, "aws_request_id", None),
            "function_version": getattr(context, "function_version", None),
        },
    }
