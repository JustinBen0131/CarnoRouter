from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
import platform
import sys
from pathlib import Path

from .research_eval import AgentTrace, ResearchSuite, content_sha256
from .research_stress import build_human_adjudication_packet


def _json_text(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(_json_text(payload), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_summary(evaluation: dict[str, object]) -> dict[str, object]:
    strict = evaluation["strict_contract_policy"]
    legacy = evaluation["legacy_score_policy"]
    raters = evaluation["simulated_raters"]
    accounting = evaluation["accounting"]
    families = evaluation["families"]
    payload: dict[str, object] = {
        "schema": "CarnoResearchStressPublicSummaryV1",
        "evidence_class": evaluation["evidence_class"],
        "status": evaluation["stress_evaluator_status"],
        "trials": evaluation["trials"],
        "control_trials": evaluation["control_trials"],
        "mutation_trials": evaluation["mutation_trials"],
        "mutation_family_count": len(families),  # type: ignore[arg-type]
        "strict_false_accepts": strict["false_accepts"],  # type: ignore[index]
        "strict_false_accept_rate_wilson_95_upper": strict["false_accept_rate_wilson_95_upper"],  # type: ignore[index]
        "strict_false_rejects": strict["false_rejects"],  # type: ignore[index]
        "legacy_false_accepts_exposed": legacy["mutation_false_accepts"],  # type: ignore[index]
        "legacy_blind_spot_families": legacy["blind_spot_families"],  # type: ignore[index]
        "minimum_family_classification_rate": min(
            float(value["classification_rate"]) for value in families.values()  # type: ignore[union-attr]
        ),
        "replication_structure": {
            "strata": evaluation["replication_structure"]["strata"],  # type: ignore[index]
            "replicates_per_stratum": evaluation["replication_structure"]["replicates_per_stratum"],  # type: ignore[index]
            "distinct_failed_criteria_outcomes": evaluation["replication_structure"]["distinct_failed_criteria_outcomes"],  # type: ignore[index]
            "strata_with_verdict_variation_across_replicates": evaluation["replication_structure"][  # type: ignore[index]
                "strata_with_verdict_variation_across_replicates"
            ],
            "wilson_bound_scope": evaluation["replication_structure"]["wilson_bound_scope"],  # type: ignore[index]
        },
        "simulated_rater_sensitivity": {
            "rater_count": raters["rater_count"],  # type: ignore[index]
            "trial_votes": raters["trial_votes"],  # type: ignore[index]
            "fleiss_kappa": raters["fleiss_kappa"],  # type: ignore[index]
            "unanimous_rate": raters["unanimous_rate"],  # type: ignore[index]
            "human_grader_agreement": "NOT_RUN",
        },
        "synthetic_accounting": {
            "total_tokens": accounting["synthetic_total_tokens"],  # type: ignore[index]
            "modeled_total_latency_ms": accounting["modeled_total_latency_ms"],  # type: ignore[index]
            "modeled_total_cost_micro_usd": accounting["modeled_total_cost_micro_usd"],  # type: ignore[index]
            "actual_provider_execution": "NOT_RUN",
        },
        "live_current_agent_benchmark": "NOT_RUN",
        "human_grader_agreement": "NOT_RUN",
        "actual_provider_accounting": "NOT_RUN",
        "live_agent_deployment": "BLOCKED",
        "hiring_outcome": "NOT_CLAIMABLE",
        "boundary": evaluation["boundary"],
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def _report_markdown(evaluation: dict[str, object]) -> str:
    strict = evaluation["strict_contract_policy"]
    legacy = evaluation["legacy_score_policy"]
    raters = evaluation["simulated_raters"]
    families = evaluation["families"]
    accounting = evaluation["accounting"]
    replication = evaluation["replication_structure"]
    lines = [
        "# Carno Blinded Evaluation Stress Benchmark",
        "",
        "> Evidence class: deterministic provider-free adversarial simulation. This is an evaluator stress test, not a live-agent or human benchmark.",
        "",
        "## Decision",
        "",
        f"- Stress evaluator: **{evaluation['stress_evaluator_status']}**",
        f"- Trials: **{int(evaluation['trials']):,}**",
        f"- Strict mutation false accepts: **{strict['false_accepts']} / {int(evaluation['mutation_trials']):,}**",  # type: ignore[index]
        f"- Strict control false rejects: **{strict['false_rejects']} / {int(evaluation['control_trials']):,}**",  # type: ignore[index]
        f"- Zero-event false-accept Wilson 95% upper bound: **{100 * float(strict['false_accept_rate_wilson_95_upper']):.4f}%**",  # type: ignore[index]
        f"- Legacy-policy false accepts exposed: **{int(legacy['mutation_false_accepts']):,}**",  # type: ignore[index]
        "",
        "## Replication structure",
        "",
        f"- Strata (task x family): **{int(replication['strata'])}**, replicates per stratum: **{int(replication['replicates_per_stratum'])}**",  # type: ignore[index]
        f"- Distinct failed-criteria outcomes across all trials: **{int(replication['distinct_failed_criteria_outcomes'])}**",  # type: ignore[index]
        f"- Strata whose strict verdict varies across replicates: **{int(replication['strata_with_verdict_variation_across_replicates'])}**",  # type: ignore[index]
        "",
        str(replication["wilson_bound_scope"]),  # type: ignore[index]
        "",
        "## Sealed perturbation matrix",
        "",
        "| Family | Trials | Classification | Legacy false accepts |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family_id, result in families.items():  # type: ignore[union-attr]
        lines.append(
            f"| `{family_id}` | {int(result['trials']):,} | "
            f"{100 * float(result['classification_rate']):.1f}% | {int(result['legacy_false_accepts']):,} |"
        )
    lines.extend(
        [
            "",
            "## Simulated-rater sensitivity",
            "",
            f"- Fixed transparent rubric profiles: **{raters['rater_count']}**",  # type: ignore[index]
            f"- Deterministic rubric votes: **{int(raters['trial_votes']):,}**",  # type: ignore[index]
            f"- Fleiss' kappa: **{float(raters['fleiss_kappa']):.3f}**",  # type: ignore[index]
            f"- Unanimous rate: **{100 * float(raters['unanimous_rate']):.1f}%**",  # type: ignore[index]
            "- Real human-grader agreement: **NOT RUN**",
            "",
            "The simulated profiles intentionally value different criteria. Their disagreement identifies adjudication pressure points; it is not measured human agreement.",
            "",
            "## Accounting",
            "",
            f"- Synthetic tokens: **{int(accounting['synthetic_total_tokens']):,}**",  # type: ignore[index]
            f"- Provider-neutral modeled latency: **{int(accounting['modeled_total_latency_ms']):,} ms**",  # type: ignore[index]
            f"- Provider-neutral modeled cost: **{int(accounting['modeled_total_cost_micro_usd']):,} micro-USD**",  # type: ignore[index]
            "- Actual provider tokens, latency, billed cost, retries, and model identity: **NOT RUN**",
            "",
            "## Remaining claim-grade gates",
            "",
            "1. Execute the frozen packet against pinned current agents in randomized paired order.",
            "2. Collect exact provider token, latency, retry, failure, and billed-cost receipts.",
            "3. Obtain at least two independent blinded human labels and adjudicate disagreements.",
            "4. Measure grader-human agreement with uncertainty and preserve the raw label receipt.",
            "",
            "## Boundary",
            "",
            str(evaluation["boundary"]),
        ]
    )
    return "\n".join(lines) + "\n"


def _report_html(evaluation: dict[str, object]) -> str:
    strict = evaluation["strict_contract_policy"]
    legacy = evaluation["legacy_score_policy"]
    raters = evaluation["simulated_raters"]
    families = evaluation["families"]
    rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(family_id))}</code></td>"
        f"<td>{int(result['trials']):,}</td>"
        f"<td>{100 * float(result['classification_rate']):.1f}%</td>"
        f"<td>{int(result['legacy_false_accepts']):,}</td>"
        "</tr>"
        for family_id, result in families.items()  # type: ignore[union-attr]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Carno Blinded Evaluation Stress Benchmark</title>
<style>
:root{{--ink:#0b1426;--paper:#f4f1e8;--panel:#fff;--line:#c7cfdb;--cyan:#007f99;--green:#087154;--amber:#9b5b00}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,sans-serif}}
main{{width:min(1180px,calc(100% - 32px));margin:auto;padding:58px 0 90px;min-width:0}} h1{{font-size:clamp(2.7rem,7vw,6.7rem);line-height:.9;letter-spacing:-.067em;max-width:1040px;margin:.2em 0}}
.eyebrow,code{{font-family:ui-monospace,SFMono-Regular,monospace}} .eyebrow{{color:var(--cyan);font-size:.75rem;font-weight:800;letter-spacing:.11em;text-transform:uppercase}}
.lede{{max-width:820px;color:#526076;font-size:1.08rem}} .metrics{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:40px 0}}
.metric{{min-width:0;padding:22px;border:1px solid var(--line);background:var(--panel);overflow-wrap:anywhere}} .metric span{{display:block;color:#68758a;font-size:.72rem;text-transform:uppercase;letter-spacing:.07em}} .metric strong{{display:block;margin-top:9px;font-size:clamp(1.8rem,3.4vw,3.2rem);line-height:1}}
h2{{margin-top:58px;font-size:1.6rem}} .table-wrap{{overflow:auto;max-width:100%;background:var(--panel);border:1px solid var(--line)}} table{{width:100%;min-width:760px;border-collapse:collapse}} th,td{{padding:13px 16px;text-align:left;border-bottom:1px solid var(--line)}} th{{font-size:.7rem;letter-spacing:.07em;text-transform:uppercase}}
.callout{{display:grid;grid-template-columns:minmax(0,1fr) minmax(260px,.45fr);gap:24px;margin-top:50px;padding:28px;border-left:6px solid var(--amber);background:var(--panel)}} .callout p{{margin:0;overflow-wrap:anywhere}} .callout strong{{color:var(--amber)}}
@media(max-width:850px){{.metrics{{grid-template-columns:repeat(2,minmax(0,1fr))}}.callout{{grid-template-columns:1fr}}}} @media(max-width:520px){{.metrics{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<p class="eyebrow">Carno / provider-free / blinded stress v1</p>
<h1>The evaluator had to survive 28,800 chances to be wrong.</h1>
<p class="lede">A sealed adversarial campaign tested contract detection, accounting completeness, blind ordering, rubric disagreement, and byte replay. It measures the evaluator, not a live model.</p>
<section class="metrics">
<article class="metric"><span>Stress gate</span><strong>{html.escape(str(evaluation['stress_evaluator_status']))}</strong></article>
<article class="metric"><span>Strict false accepts</span><strong>{int(strict['false_accepts']):,}</strong></article>
<article class="metric"><span>Legacy blind spots</span><strong>{int(legacy['mutation_false_accepts']):,}</strong></article>
<article class="metric"><span>Simulated votes</span><strong>{int(raters['trial_votes']):,}</strong></article>
</section>
<h2>Twelve sealed families</h2>
<div class="table-wrap"><table><thead><tr><th>Family</th><th>Trials</th><th>Classification</th><th>Legacy false accepts</th></tr></thead><tbody>{rows}</tbody></table></div>
<section class="callout"><p><strong>Boundary</strong><br>{html.escape(str(evaluation['boundary']))}</p><p><strong>Still unmeasured</strong><br>Current agents · actual provider economics · real human agreement · hiring outcome</p></section>
</main></body></html>"""


def write_stress_artifacts(
    output: str | Path,
    suite: ResearchSuite,
    traces: tuple[AgentTrace, ...],
    truth: dict[str, dict[str, object]],
    blind_manifest: dict[str, object],
    records: list[dict[str, object]],
    evaluation: dict[str, object],
) -> dict[str, object]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    packet, answer_key = build_human_adjudication_packet(suite, traces, truth)
    unblinding = {
        "schema": "CarnoStressUnblindingKeyV1",
        "suite_id": suite.suite_id,
        "candidate_count": len(truth),
        "entries": [truth[candidate_id] for candidate_id in sorted(truth)],
    }
    unblinding["content_sha256"] = content_sha256(unblinding)
    public_summary = _public_summary(evaluation)
    benchmark_spec = {
        "schema": "CarnoResearchStressBenchmarkSpecV1",
        "suite_id": suite.suite_id,
        "suite_content_sha256": suite.to_dict()["content_sha256"],
        "config": evaluation["config"],
        "mutation_labels_hidden_until_after_grading": True,
        "strict_policy": "legacy weighted score gate plus every declared criterion passing",
        "human_panel_status": "NOT_RUN",
        "provider_execution": False,
        "network_access_required": False,
        "publication_authorized": False,
        "hiring_probability_claim": False,
    }
    benchmark_spec["content_sha256"] = content_sha256(benchmark_spec)

    _write_json(root / "benchmark_spec.json", benchmark_spec)
    _write_json(root / "blind_manifest.json", blind_manifest)
    _write_json(root / "unblinding_key.json", unblinding)
    # Every sealed candidate trace is persisted in blind order so an independent party
    # can re-grade the campaign from artifacts alone, without trusting the generator.
    (root / "candidate_traces.jsonl").write_text(
        "".join(json.dumps(trace.to_dict(), sort_keys=True, separators=(",", ":")) + "\n" for trace in traces),
        encoding="utf-8",
    )
    (root / "trials.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    _write_json(root / "stress_evaluation.json", evaluation)
    _write_json(root / "simulated_rater_summary.json", evaluation["simulated_raters"])
    _write_json(root / "accounting_summary.json", evaluation["accounting"])
    _write_json(root / "human_adjudication_packet.json", packet)
    _write_json(root / "human_adjudication_answer_key.json", answer_key)
    _write_json(root / "public_summary.json", public_summary)
    (root / "report.md").write_text(_report_markdown(evaluation), encoding="utf-8")
    (root / "report.html").write_text(_report_html(evaluation), encoding="utf-8")
    with (root / "family_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "mutation_family",
                "trials",
                "strict_false_accepts",
                "strict_false_rejects",
                "classification_rate",
                "legacy_false_accepts",
            ]
        )
        for family_id, result in evaluation["families"].items():  # type: ignore[union-attr]
            writer.writerow(
                [
                    family_id,
                    result["trials"],
                    result["false_accepts"],
                    result["false_rejects"],
                    result["classification_rate"],
                    result["legacy_false_accepts"],
                ]
            )

    artifact_names = (
        "benchmark_spec.json",
        "blind_manifest.json",
        "unblinding_key.json",
        "candidate_traces.jsonl",
        "trials.jsonl",
        "stress_evaluation.json",
        "simulated_rater_summary.json",
        "accounting_summary.json",
        "human_adjudication_packet.json",
        "human_adjudication_answer_key.json",
        "public_summary.json",
        "report.md",
        "report.html",
        "family_matrix.csv",
    )
    manifest: dict[str, object] = {
        "schema": "CarnoResearchStressDeterministicManifestV1",
        "evidence_class": evaluation["evidence_class"],
        "stress_evaluator_status": evaluation["stress_evaluator_status"],
        "trial_count": evaluation["trials"],
        "artifact_count": len(artifact_names),
        "artifacts": [
            {"path": name, "sha256": _sha256(root / name), "bytes": (root / name).stat().st_size}
            for name in artifact_names
        ],
        "byte_reproducible_core": True,
        "execution_receipt_excluded_from_core": True,
        "network_access": False,
        "provider_execution": False,
        "human_grader_agreement": "NOT_RUN",
        "live_agent_deployment": "BLOCKED",
    }
    manifest["content_sha256"] = content_sha256(manifest)
    _write_json(root / "DETERMINISTIC_MANIFEST.json", manifest)
    return manifest


def write_execution_receipt(
    output: str | Path,
    evaluation: dict[str, object],
    manifest: dict[str, object],
    elapsed_ns: int,
) -> dict[str, object]:
    if elapsed_ns <= 0:
        raise ValueError("observed elapsed time must be positive")
    trials = int(evaluation["trials"])
    receipt: dict[str, object] = {
        "schema": "CarnoResearchStressExecutionReceiptV1",
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": evaluation["stress_evaluator_status"],
        "trial_count": trials,
        "observed_local_harness_elapsed_ns": elapsed_ns,
        "observed_local_harness_trials_per_second": round(trials * 1_000_000_000 / elapsed_ns, 3),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "deterministic_manifest_content_sha256": manifest["content_sha256"],
        "actual_provider_execution": "NOT_RUN",
        "actual_provider_tokens": None,
        "actual_provider_latency_ms": None,
        "actual_provider_billed_cost_usd": None,
        "boundary": "Observed time covers local synthetic campaign construction, grading, aggregation, and artifact writing only.",
    }
    receipt["content_sha256"] = content_sha256(receipt)
    _write_json(Path(output) / "execution_receipt.json", receipt)
    return receipt

