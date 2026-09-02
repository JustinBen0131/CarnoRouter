#!/usr/bin/env python3
"""Verify captured Lambda shard responses against the frozen local shard-contract core.

This is the verification half of the cross-environment coordinator. It never calls
AWS. It reads response payloads captured elsewhere (one JSON file per invocation,
each holding the handler's ``{"deterministic": ..., "runtime": ...}`` return value),
recomputes every shard's content hash, and accepts the run only when every expected
shard is present exactly once, every per-shard hash equals the frozen local
expectation in ``aws_lambda_shard_contract_core.json``, every receipt belongs to its
shard, and the union equals the local trial receipts. Runtime fields (request ids,
function versions) are reported as receipts and never enter the comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SCHEMAS = {"CarnoAwsLambdaShardContractCoreV2"}


def content_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def shard_for(candidate_id: str, shard_count: int) -> int:
    digest = hashlib.sha256(f"carno-aws-shard-v1:{candidate_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def load_local_receipts(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            candidate_id = str(record["candidate_id"])
            if candidate_id in result:
                raise ValueError("duplicate candidate in local trial file")
            result[candidate_id] = str(record["receipt_sha256"])
    return result


def verify_responses(
    expected: dict[str, object],
    responses: list[dict[str, object]],
    local: dict[str, str],
) -> dict[str, object]:
    if expected.get("schema") not in EXPECTED_SCHEMAS:
        raise ValueError("unsupported shard-contract core schema")
    body = {key: value for key, value in expected.items() if key != "content_sha256"}
    if content_sha256(body) != expected["content_sha256"]:
        raise ValueError("shard-contract core content hash does not match its body")
    shard_count = int(expected["shard_count"])
    expected_by_index = {int(item["shard_index"]): item for item in expected["expected_shards"]}  # type: ignore[union-attr]
    if len(expected_by_index) != shard_count:
        raise ValueError("shard-contract core does not freeze one expectation per shard")
    seen: dict[int, int] = {}
    hash_mismatches: list[int] = []
    body_hash_failures: list[int] = []
    foreign: list[str] = []
    duplicates: list[str] = []
    evaluation_drift: list[int] = []
    assembled: dict[str, str] = {}
    runtime: list[dict[str, object]] = []
    for response in responses:
        result = response["deterministic"]
        index = int(result["shard_index"])  # type: ignore[index]
        seen[index] = seen.get(index, 0) + 1
        result_body = {key: value for key, value in result.items() if key != "content_sha256"}  # type: ignore[union-attr]
        if content_sha256(result_body) != result["content_sha256"]:  # type: ignore[index]
            body_hash_failures.append(index)
        anticipated = expected_by_index.get(index)
        if anticipated is None or anticipated["content_sha256"] != result["content_sha256"]:  # type: ignore[index]
            hash_mismatches.append(index)
        if (
            int(result["shard_count"]) != shard_count  # type: ignore[index]
            or result["canonical_evaluation_content_sha256"] != expected["canonical_evaluation_content_sha256"]  # type: ignore[index]
        ):
            evaluation_drift.append(index)
        for item in result["receipts"]:  # type: ignore[index]
            candidate_id = str(item["candidate_id"])
            if shard_for(candidate_id, shard_count) != index:
                foreign.append(candidate_id)
            if candidate_id in assembled:
                duplicates.append(candidate_id)
            assembled[candidate_id] = str(item["receipt_sha256"])
        runtime.append({"shard_index": index, **dict(response.get("runtime", {}))})  # type: ignore[arg-type]
    missing_shards = sorted(set(expected_by_index) - set(seen))
    repeated_shards = sorted(index for index, count in seen.items() if count > 1)
    unexpected_shards = sorted(set(seen) - set(expected_by_index))
    missing = sorted(set(local) - set(assembled))
    extra = sorted(set(assembled) - set(local))
    mismatches = sorted(candidate_id for candidate_id in set(local) & set(assembled) if local[candidate_id] != assembled[candidate_id])
    gates = {
        "every_expected_shard_present_exactly_once": not missing_shards and not repeated_shards and not unexpected_shards,
        "every_shard_body_hash_recomputes": not body_hash_failures,
        "every_shard_hash_matches_frozen_expectation": not hash_mismatches,
        "every_shard_reports_canonical_evaluation": not evaluation_drift,
        "every_receipt_belongs_to_its_shard": not foreign,
        "no_duplicate_candidates": not duplicates,
        "no_missing_candidates": not missing,
        "no_extra_candidates": not extra,
        "all_receipts_match_local": not mismatches,
        "union_hash_matches_frozen_expectation": content_sha256(sorted(assembled.items())) == expected["union_receipts_sha256"],
    }
    return {
        "schema": "CarnoAwsShardResponseVerificationV1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "shard_contract_core_content_sha256": expected["content_sha256"],
        "responses": len(responses),
        "shard_count": shard_count,
        "reassembled_trial_count": len(assembled),
        "local_trial_count": len(local),
        "missing_shards": missing_shards[:20],
        "repeated_shards": repeated_shards[:20],
        "unexpected_shards": unexpected_shards[:20],
        "shard_hash_mismatches": hash_mismatches[:20],
        "shard_body_hash_failures": body_hash_failures[:20],
        "foreign_candidates": foreign[:20],
        "duplicate_candidates": duplicates[:20],
        "missing_candidates": missing[:20],
        "extra_candidates": extra[:20],
        "receipt_mismatches": mismatches[:20],
        "runtime_receipts": sorted(runtime, key=lambda item: int(item["shard_index"])),
        "gates": gates,
        "external_state_changed": False,
        "boundary": "Verifies captured shard responses only. It does not invoke AWS and does not prove who produced the responses.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, required=True, help="aws_lambda_shard_contract_core.json written by validate_aws_shard_contract.py")
    parser.add_argument("--responses", type=Path, required=True, help="directory of captured per-invocation response JSON files")
    parser.add_argument("--trials", type=Path, required=True, help="local trials.jsonl providing canonical receipts")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.core.read_text(encoding="utf-8"))
    responses = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.responses.glob("*.json"))]
    local = load_local_receipts(args.trials)
    verification = verify_responses(expected, responses, local)
    verification["content_sha256"] = content_sha256(verification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": verification["status"], "responses": len(responses), "output": str(args.output.resolve())}, indent=2, sort_keys=True))
    return 0 if verification["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
