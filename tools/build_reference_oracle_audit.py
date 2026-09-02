#!/usr/bin/env python3
"""Re-grade the sealed stress campaign from artifacts alone with the independent oracle.

The audit never calls the campaign generator. It reads the persisted candidate traces,
verifies each against the blind manifest's sealed hash, takes expected labels from the
unblinding key and production verdicts from the trial receipts, and then compares the
reference oracle's own verdicts against both. It also runs the generative held-out
attack search and the metamorphic relations against the production grader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from carno_router.research_eval import ResearchSuite, content_sha256, load_traces, reference_suite
from carno_router.research_reference_oracle import SearchConfig, audit_frozen_campaign, audit_search


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_content_hash(payload: dict[str, object], name: str) -> None:
    without = {key: value for key, value in payload.items() if key != "content_sha256"}
    if content_sha256(without) != payload.get("content_sha256"):
        raise SystemExit(f"FAIL: {name} content hash does not match its body")


def _report_markdown(audit: dict[str, object]) -> str:
    frozen = audit["frozen_campaign"]
    search = audit["held_out_search"]
    gaps = search["accepted_traces_violating_held_out_invariant"]  # type: ignore[index]
    lines = [
        "# Independent reference-oracle audit",
        "",
        "> Evidence class: a second grader with separate implementation structure re-graded the sealed campaign from artifacts alone and searched a held-out attack grammar. It measures oracle coupling and contract coverage; it is not a live-agent or human benchmark.",
        "",
        "## Frozen campaign re-grade",
        "",
        f"- Candidates re-graded from `candidate_traces.jsonl`: **{int(frozen['candidates']):,}**",  # type: ignore[index]
        f"- Oracle vs sealed truth disagreements: **{frozen['oracle_vs_truth_disagreements']}**",  # type: ignore[index]
        f"- Oracle vs production disagreements: **{frozen['oracle_vs_production_disagreements']}**",  # type: ignore[index]
        f"- Distinct declared-criterion status vectors across the campaign: **{frozen['distinct_declared_status_vectors']}**",  # type: ignore[index]
        "",
        "The sealed families each flip a fixed set of declared criteria, so the campaign's 28,800 trials exercise only the status vectors listed above. Replicates add seed-dependent values, not new grader branches. That count is the honest measure of how many independent grader outcomes the headline covers.",
        "",
        "## Held-out attack search",
        "",
        f"- Attacks generated from the operator grammar: **{int(search['attacks']):,}** ({search['distinct_declared_status_vectors']} distinct declared status vectors)",  # type: ignore[index]
        f"- Attacks violating a declared criterion: **{int(search['declared_violation_attacks']):,}**, production false accepts: **{search['declared_violation_false_accepts']}** (Wilson 95% upper {search['declared_violation_false_accept_wilson_95_upper']})",  # type: ignore[index]
        f"- Attacks satisfying every declared criterion: **{int(search['held_out_only_attacks']):,}**, accepted by production: **{int(search['held_out_only_accepted_by_production']):,}**",  # type: ignore[index]
        f"- Oracle vs production disagreements: **{search['oracle_production_disagreements']}**",  # type: ignore[index]
        f"- Metamorphic relation failures: **{len(search['metamorphic_failures'])}**",  # type: ignore[arg-type]
        "",
        "## Contract silence made visible",
        "",
        "| Held-out invariant | Accepted traces violating it |",
        "| --- | ---: |",
    ]
    for name, count in gaps.items():  # type: ignore[union-attr]
        lines.append(f"| `{name}` | {int(count):,} |")
    lines.extend(
        [
            "",
            "These traces satisfy every documented criterion and are accepted by design. They are coverage gaps of the declared contract, not false accepts. Closing them requires widening the documented contract on purpose.",
            "",
            "## Boundary",
            "",
            str(audit["boundary"]),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent reference-oracle audit of the sealed stress campaign.")
    parser.add_argument("--stress", type=Path, default=Path("artifacts/research_stress_v1"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research_stress_oracle_v1"))
    parser.add_argument("--attacks-per-task", type=int, default=200)
    parser.add_argument("--root-seed", type=int, default=7_071_067)
    args = parser.parse_args()

    traces_path = args.stress / "candidate_traces.jsonl"
    if not traces_path.is_file():
        raise SystemExit("FAIL: candidate_traces.jsonl is absent; rebuild the stress campaign so traces are persisted")
    spec = _load(args.stress / "benchmark_spec.json")
    blind = _load(args.stress / "blind_manifest.json")
    key = _load(args.stress / "unblinding_key.json")
    evaluation = _load(args.stress / "stress_evaluation.json")
    for payload, name in ((spec, "benchmark_spec"), (blind, "blind_manifest"), (key, "unblinding_key"), (evaluation, "stress_evaluation")):
        _verify_content_hash(payload, name)

    suite: ResearchSuite = reference_suite()
    suite_hash = suite.to_dict()["content_sha256"]
    if suite_hash != spec["suite_content_sha256"] or suite_hash != evaluation["suite_content_sha256"]:
        raise SystemExit("FAIL: in-process suite hash differs from the sealed suite hash")

    traces = load_traces(str(traces_path))
    rows = blind["candidates"]
    if len(rows) != len(traces):  # type: ignore[arg-type]
        raise SystemExit("FAIL: persisted trace count differs from blind manifest")
    for index, (trace, row) in enumerate(zip(traces, rows)):  # type: ignore[arg-type]
        if row["blind_index"] != index or row["candidate_id"] != trace.trace_id:
            raise SystemExit(f"FAIL: blind order drift at index {index}")
        if content_sha256(trace.to_dict()) != row["trace_sha256"]:
            raise SystemExit(f"FAIL: persisted trace {trace.trace_id} does not match its sealed hash")

    expected = {str(entry["candidate_id"]): bool(entry["expected_accept"]) for entry in key["entries"]}  # type: ignore[index]
    production: dict[str, bool] = {}
    receipt_lines = 0
    with (args.stress / "trials.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            receipt_lines += 1
            body = {name: value for name, value in record.items() if name != "receipt_sha256"}
            if content_sha256(body) != record["receipt_sha256"]:
                raise SystemExit(f"FAIL: receipt hash mismatch for {record['candidate_id']}")
            production[str(record["candidate_id"])] = bool(record["result"]["strict_contract_accept"])
    if receipt_lines != len(production):
        raise SystemExit("FAIL: duplicate candidate identities in trial receipts")

    frozen = audit_frozen_campaign(suite, traces, expected, production)
    search = audit_search(suite, SearchConfig(root_seed=args.root_seed, attacks_per_task=args.attacks_per_task))
    gates = {
        "artifact_hashes_verified": True,
        "persisted_traces_match_sealed_hashes": True,
        "receipt_hashes_recomputed": True,
        "frozen_campaign_audit_pass": frozen["status"] == "PASS",
        "held_out_search_pass": search["status"] == "PASS",
        "production_headline_reproduced": (
            int(evaluation["strict_contract_policy"]["false_accepts"]) == 0  # type: ignore[index]
            and int(evaluation["strict_contract_policy"]["false_rejects"]) == 0  # type: ignore[index]
            and int(evaluation["trials"]) == len(traces)
        ),
    }
    audit: dict[str, object] = {
        "schema": "CarnoIndependentReferenceOracleAuditV1",
        "evidence_class": "independent_reference_oracle_audit",
        "oracle_independence": {
            "reads_production_passed_field_to_decide": False,
            "calls_production_mutation_helper": False,
            "calls_campaign_generator": False,
            "declared_contract_source": "docs/research_autonomy_lab.md re-derived as a declarative invariant table",
            "legacy_score_arithmetic": "exact rational (fractions.Fraction)",
        },
        "inputs": {
            "suite_content_sha256": suite_hash,
            "blind_manifest_content_sha256": blind["content_sha256"],
            "unblinding_key_content_sha256": key["content_sha256"],
            "stress_evaluation_content_sha256": evaluation["content_sha256"],
            "candidate_traces_sha256": hashlib.sha256(traces_path.read_bytes()).hexdigest(),
            "trials_sha256": hashlib.sha256((args.stress / "trials.jsonl").read_bytes()).hexdigest(),
        },
        "frozen_campaign": frozen,
        "held_out_search": search,
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "live_current_agent_benchmark": "NOT_RUN",
        "human_grader_agreement": "NOT_RUN",
        "actual_provider_accounting": "NOT_RUN",
        "live_agent_deployment": "BLOCKED",
        "hiring_outcome": "NOT_CLAIMABLE",
        "boundary": (
            "This audit measures oracle coupling and declared-contract coverage of a deterministic synthetic evaluator. "
            "It does not measure a current agent, a paid provider, real human agreement, production deployment, or hiring probability."
        ),
    }
    audit["content_sha256"] = content_sha256(audit)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "reference_oracle_audit.json").write_text(_json_text(audit), encoding="utf-8")
    (args.output / "report.md").write_text(_report_markdown(audit), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": audit["status"],
                "candidates_regraded": frozen["candidates"],
                "oracle_vs_truth_disagreements": frozen["oracle_vs_truth_disagreements"],
                "oracle_vs_production_disagreements": frozen["oracle_vs_production_disagreements"],
                "campaign_distinct_status_vectors": frozen["distinct_declared_status_vectors"],
                "held_out_attacks": search["attacks"],
                "held_out_declared_false_accepts": search["declared_violation_false_accepts"],
                "held_out_accepted_gap_traces": search["held_out_only_accepted_by_production"],
                "content_sha256": audit["content_sha256"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
