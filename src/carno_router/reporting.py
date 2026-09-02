from __future__ import annotations

import csv
import hashlib
import html
import json
from pathlib import Path
from typing import Any


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_bytes(_json_bytes(payload))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _report_markdown(report: dict[str, Any]) -> str:
    routed = report["test_policies"]["routed"]
    strong = report["test_policies"]["always_strong"]
    cheap = report["test_policies"]["always_cheap"]
    interval = report["uncertainty"]["modeled_cost_savings_vs_always_strong"]
    point = report["test_policies"]["routed_point_estimate"]
    deployment = report["deployment_decision"]
    shift_warnings = report["distribution_shift"]["warnings"]
    return f"""# Carno cost-quality benchmark

Status: **{report['status']}**

This report evaluates an offline routing policy against frozen paired outcomes. It contains no provider calls and no raw prompts.

## Selected routed policy

| Metric | Routed | Always cheap | Always strong |
|---|---:|---:|---:|
| Mean quality | {routed['mean_quality']:.4f} | {cheap['mean_quality']:.4f} | {strong['mean_quality']:.4f} |
| Mean cost ({report['pricing']['currency']}) | {routed['mean_cost']} | {cheap['mean_cost']} | {strong['mean_cost']} |
| Strong-call rate | {100 * routed['strong_call_rate']:.1f}% | {100 * cheap['strong_call_rate']:.1f}% | {100 * strong['strong_call_rate']:.1f}% |
| Preference accuracy | {100 * routed['preference_accuracy']:.1f}% | {100 * cheap['preference_accuracy']:.1f}% | {100 * strong['preference_accuracy']:.1f}% |

- Calibration-selected threshold: `{report['threshold_selection']['selected']['threshold']}`
- Point-estimate challenger threshold: `{report['threshold_selection']['point_estimate']['threshold']}`
- Quality delta versus always-strong: `{routed['quality_delta_vs_always_strong']:+.4f}`
- Modeled cost savings versus always-strong: `{100 * routed['modeled_cost_savings_vs_always_strong']:.1f}%`
- Paired group-bootstrap 95% savings interval: `{100 * interval['lower_95']:.1f}%` to `{100 * interval['upper_95']:.1f}%`
- Test ROC-AUC: `{report['calibration']['test_metrics']['roc_auc']}`
- Test Brier score: `{report['calibration']['test_metrics']['brier']}`
- Point-estimate challenger savings / quality delta: `{100 * point['modeled_cost_savings_vs_always_strong']:.1f}%` / `{point['quality_delta_vs_always_strong']:+.4f}`

## Decision gate

- Offline research artifact: **{deployment['offline_research_artifact']}**
- Portfolio evidence: **{deployment['portfolio_evidence']}**
- Live autorouting: **{deployment['live_autorouting']}**
- Shift warnings: `{', '.join(shift_warnings) if shift_warnings else 'none'}`

The primary policy is selected with a calibration-only, one-sided 95% prompt-group bootstrap quality-risk bound. The point-estimate policy is retained as a challenger and is not silently promoted.

## Evidence boundary

""" + "\n".join(f"- {line}" for line in report["claim_boundaries"]) + "\n"


def _frontier_svg(report: dict[str, Any]) -> str:
    points = report["test_cost_quality_frontier"]
    costs = [float(point["mean_cost"]) for point in points]
    qualities = [float(point["mean_quality"]) for point in points]
    min_cost, max_cost = min(costs), max(costs)
    min_quality, max_quality = min(qualities), max(qualities)

    def x(value: float) -> float:
        span = max(max_cost - min_cost, 1e-12)
        return 80 + 620 * (value - min_cost) / span

    def y(value: float) -> float:
        span = max(max_quality - min_quality, 1e-12)
        return 360 - 280 * (value - min_quality) / span

    polyline = " ".join(f"{x(cost):.2f},{y(quality):.2f}" for cost, quality in zip(costs, qualities))
    circles = "\n".join(
        f'<circle cx="{x(cost):.2f}" cy="{y(quality):.2f}" r="4" fill="#2b6cb0" />'
        for cost, quality in zip(costs, qualities)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="760" height="430" viewBox="0 0 760 430" role="img" aria-label="Cost quality frontier">
<rect width="760" height="430" fill="#ffffff"/>
<text x="380" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="700">Cost-quality frontier</text>
<line x1="80" y1="360" x2="700" y2="360" stroke="#334155"/>
<line x1="80" y1="80" x2="80" y2="360" stroke="#334155"/>
<text x="390" y="405" text-anchor="middle" font-family="sans-serif" font-size="14">Mean modeled cost</text>
<text x="22" y="220" text-anchor="middle" transform="rotate(-90 22 220)" font-family="sans-serif" font-size="14">Mean quality</text>
<text x="80" y="382" text-anchor="middle" font-family="monospace" font-size="11">{min_cost:.3g}</text>
<text x="700" y="382" text-anchor="middle" font-family="monospace" font-size="11">{max_cost:.3g}</text>
<text x="68" y="364" text-anchor="end" font-family="monospace" font-size="11">{min_quality:.3f}</text>
<text x="68" y="84" text-anchor="end" font-family="monospace" font-size="11">{max_quality:.3f}</text>
<polyline points="{polyline}" fill="none" stroke="#2b6cb0" stroke-width="2"/>
{circles}
</svg>
"""


def _calibration_svg(report: dict[str, Any]) -> str:
    bins = [row for row in report["calibration"]["test_metrics"]["bins"] if row["count"]]
    bars = []
    for index, row in enumerate(bins):
        x = 90 + index * 118
        observed_height = 250 * float(row["observed"])
        predicted_height = 250 * float(row["predicted"])
        bars.append(
            f'<rect x="{x}" y="{350-observed_height:.2f}" width="38" height="{observed_height:.2f}" fill="#2b6cb0"/>'
        )
        bars.append(
            f'<rect x="{x+42}" y="{350-predicted_height:.2f}" width="38" height="{predicted_height:.2f}" fill="#d97706"/>'
        )
        bars.append(
            f'<text x="{x+40}" y="375" text-anchor="middle" font-family="monospace" font-size="10">{row["low"]:.1f}-{row["high"]:.1f}</text>'
        )
    return """<svg xmlns="http://www.w3.org/2000/svg" width="760" height="430" viewBox="0 0 760 430" role="img" aria-label="Calibration bins">
<rect width="760" height="430" fill="#ffffff"/>
<text x="380" y="30" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="700">Held-out calibration</text>
<line x1="70" y1="350" x2="700" y2="350" stroke="#334155"/>
<line x1="70" y1="100" x2="70" y2="350" stroke="#334155"/>
<text x="80" y="60" font-family="sans-serif" font-size="12" fill="#2b6cb0">Observed</text>
<text x="160" y="60" font-family="sans-serif" font-size="12" fill="#d97706">Predicted</text>
""" + "\n".join(bars) + "\n</svg>\n"


def _report_html(report: dict[str, Any]) -> str:
    policies = report["test_policies"]
    rows = []
    for name in (
        "routed",
        "routed_point_estimate",
        "routed_safety_recall_95",
        "routed_balanced_accuracy",
        "prompt_length_logistic",
        "always_cheap",
        "always_strong",
        "deterministic_random_matched_rate",
        "oracle_ceiling",
    ):
        metric = policies[name]
        rows.append(
            "<tr>"
            f"<td>{html.escape(name.replace('_', ' ').title())}</td>"
            f"<td>{metric['mean_quality']:.4f}</td>"
            f"<td>{html.escape(str(metric['mean_cost']))}</td>"
            f"<td>{100 * metric['strong_call_rate']:.1f}%</td>"
            f"<td>{100 * metric['preference_accuracy']:.1f}%</td>"
            f"<td>{100 * metric['modeled_cost_savings_vs_always_strong']:.1f}%</td>"
            "</tr>"
        )
    boundaries = "".join(
        f"<li>{html.escape(line)}</li>" for line in report["claim_boundaries"]
    )
    blockers = "".join(
        f"<li>{html.escape(line)}</li>"
        for line in report["deployment_decision"]["blockers"]
    )
    warnings = report["distribution_shift"]["warnings"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Carno cost-quality benchmark</title>
<style>
*{{box-sizing:border-box}} body{{margin:0;background:#f8fafc;color:#172033;font:15px/1.55 system-ui,-apple-system,sans-serif}} main{{max-width:1100px;margin:auto;padding:36px 22px}} .hero,.card{{background:white;border:1px solid #dbe3ee;border-radius:16px;padding:24px;margin-bottom:18px;min-width:0;max-width:100%;overflow-wrap:anywhere;word-break:break-word;box-shadow:0 4px 18px rgba(15,23,42,.05)}} .eyebrow{{color:#2563eb;font-weight:700;letter-spacing:.08em;text-transform:uppercase;font-size:12px}} h1{{font-size:38px;line-height:1.1;margin:.35em 0}} h2{{font-size:22px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}} .metric{{background:#eff6ff;border-radius:12px;padding:16px;min-width:0;max-width:100%;overflow-wrap:anywhere;word-break:break-word}} .metric strong{{display:block;font-size:26px}} table{{border-collapse:collapse;width:100%;display:block;overflow-x:auto}} th,td{{border-bottom:1px solid #e2e8f0;padding:10px 12px;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} .charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}} img{{width:100%;height:auto;border:1px solid #e2e8f0;border-radius:12px}} code{{overflow-wrap:anywhere;word-break:break-word}} .boundary{{border-left:5px solid #d97706}} .issue-row{{display:block;min-width:0;max-width:100%;overflow-wrap: anywhere;word-break: break-word}} .issue-key{{min-width:0;max-width:100%;overflow-wrap: anywhere;word-break: break-word}} .issue-body{{min-width: 0;max-width:100%;overflow-wrap: anywhere;word-break: break-word}} .issue-title{{min-width:0;max-width:100%;overflow-wrap: anywhere;word-break: break-word}} .chip{{min-width:0;max-width:100%;overflow-wrap: anywhere;word-break: break-word}} .chips{{display:flex;flex-wrap: wrap;}}
</style>
</head>
<body><main>
<section class="hero"><div class="eyebrow">Offline reproducibility receipt</div><h1>Carno cost-quality benchmark</h1><p>Status: <code>{html.escape(report['status'])}</code>. No provider calls. No raw prompts in the report.</p></section>
<section class="grid">
<div class="metric"><span>Routed mean quality</span><strong>{policies['routed']['mean_quality']:.4f}</strong></div>
<div class="metric"><span>Strong-call rate</span><strong>{100 * policies['routed']['strong_call_rate']:.1f}%</strong></div>
<div class="metric"><span>Modeled savings</span><strong>{100 * policies['routed']['modeled_cost_savings_vs_always_strong']:.1f}%</strong></div>
<div class="metric"><span>Quality delta</span><strong>{policies['routed']['quality_delta_vs_always_strong']:+.4f}</strong></div>
</section>
<section class="card boundary"><h2>Deployment decision: {html.escape(report['deployment_decision']['live_autorouting'])}</h2><p>The primary policy uses a calibration-only one-sided 95% quality-risk bound. A lower-cost point estimate remains a challenger, not a production claim.</p><ul>{blockers}</ul><p>Shift warnings: <code>{html.escape(', '.join(warnings) if warnings else 'none')}</code></p></section>
<section class="card"><h2>Held-out policy comparison</h2><table><thead><tr><th>Policy</th><th>Quality</th><th>Mean cost</th><th>Strong calls</th><th>Preference accuracy</th><th>Savings vs strong</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
<section class="charts"><div class="card"><img src="cost_quality_frontier.svg" alt="Cost-quality frontier"></div><div class="card"><img src="calibration_plot.svg" alt="Calibration bins"></div></section>
<section class="card boundary"><h2>Claim boundary</h2><ul>{boundaries}</ul><p>Pricing snapshot: <code>{html.escape(report['pricing']['snapshot_id'])}</code>; dataset SHA-256: <code>{report['dataset']['sha256']}</code>.</p></section>
</main></body></html>
"""


def write_benchmark_artifacts(output: str | Path, run: dict[str, object]) -> dict[str, object]:
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    report = run["report"]
    if not isinstance(report, dict):
        raise TypeError("run report must be a mapping")

    _write_json(root / "report.json", report)
    _write_json(root / "model_bundle.json", run["model_bundle"])
    _write_json(root / "deployment_decision.json", report["deployment_decision"])
    _write_json(
        root / "model_card.json",
        {
            "schema": "CarnoModelCardV1",
            "intended_use": "offline research evaluation and shadow-policy scoring",
            "prohibited_use": "unsupervised live provider routing without a fresh provider evaluation and release review",
            "feature_boundary": "prompt text only; no candidate answers or assistant responses",
            "model": report["model"],
            "calibration": {
                key: report["calibration"][key]
                for key in ("method", "fit_data", "fingerprint", "test_metrics")
            },
            "deployment_decision": report["deployment_decision"],
        },
    )
    _write_json(
        root / "public_claims.json",
        {
            "schema": "CarnoPublicClaimRegistryV1",
            "claims": [
                {
                    "claim": "The audited source contained 33,000 human-preference conversations.",
                    "status": "measured_source_fact",
                    "evidence": "separate private cohort receipt",
                },
                {
                    "claim": f"The strict evaluation cohort contains {report['dataset']['row_count']} rows across {report['dataset']['prompt_group_count']} prompt groups.",
                    "status": "measured",
                    "evidence": "report.json dataset receipt",
                },
                {
                    "claim": "Model and threshold selection do not use test data.",
                    "status": "verified_contract",
                    "evidence": "model and threshold selection receipts",
                },
                {
                    "claim": "Live autorouting is approved.",
                    "status": "blocked",
                    "evidence": "deployment_decision.json",
                },
            ],
        },
    )
    (root / "report.md").write_text(_report_markdown(report), encoding="utf-8")
    (root / "report.html").write_text(_report_html(report), encoding="utf-8")
    (root / "cost_quality_frontier.svg").write_text(
        _frontier_svg(report), encoding="utf-8"
    )
    (root / "calibration_plot.svg").write_text(
        _calibration_svg(report), encoding="utf-8"
    )
    _write_json(root / "bootstrap_intervals.json", run["bootstrap_intervals"])
    _write_csv(
        root / "predictions.csv",
        list(run["predictions"]),
        [
            "example_sha256",
            "group_sha256",
            "model_pair",
            "strong_needed_probability",
            "selected_model",
            "failure_type",
            "strong_is_better",
            "cheap_quality",
            "strong_quality",
        ],
    )
    _write_csv(
        root / "split_manifest.csv",
        list(run["split_manifest"]),
        ["example_sha256", "group_sha256", "split", "model_pair"],
    )
    threshold_rows = [
        {
            "threshold": row["threshold"],
            "mean_quality": row["mean_quality"],
            "mean_cost": row["mean_cost"],
            "strong_call_rate": row["strong_call_rate"],
            "preference_accuracy": row["preference_accuracy"],
            "strong_needed_recall": row["strong_needed_recall"],
            "weak_opportunity_specificity": row["weak_opportunity_specificity"],
            "quality_drop_vs_always_strong": row["quality_drop_vs_always_strong"],
            "quality_drop_upper_pointwise_one_sided_95": row[
                "quality_drop_upper_pointwise_one_sided_95"
            ],
            "quality_drop_upper_simultaneous_one_sided_95": row[
                "quality_drop_upper_simultaneous_one_sided_95"
            ],
            "risk_constraint_pass": row["risk_constraint_pass"],
        }
        for row in run["validation_threshold_curve"]
    ]
    _write_csv(
        root / "threshold_curve.csv",
        threshold_rows,
        [
            "threshold",
            "mean_quality",
            "mean_cost",
            "strong_call_rate",
            "preference_accuracy",
            "strong_needed_recall",
            "weak_opportunity_specificity",
            "quality_drop_vs_always_strong",
            "quality_drop_upper_pointwise_one_sided_95",
            "quality_drop_upper_simultaneous_one_sided_95",
            "risk_constraint_pass",
        ],
    )
    _write_csv(
        root / "slice_diagnostics.csv",
        list(report["slice_diagnostics"]),
        [
            "dimension",
            "value",
            "rows",
            "interpretation",
            "strong_better_prevalence",
            "roc_auc",
            "brier",
            "preference_accuracy",
            "strong_call_rate",
            "strong_needed_recall",
        ],
    )

    artifact_names = sorted(
        path.name for path in root.iterdir() if path.is_file() and path.name != "MANIFEST.json"
    )
    manifest = {
        "schema": "CarnoBenchmarkArtifactManifestV2",
        "benchmark_id": report["benchmark_id"],
        "artifacts": {
            name: {"sha256": _sha256(root / name), "bytes": (root / name).stat().st_size}
            for name in artifact_names
        },
        "reproduction": "PYTHONPATH=src python3 -m carno_router benchmark --data <csv> --pricing <snapshot> --output <dir>",
    }
    _write_json(root / "MANIFEST.json", manifest)
    return manifest
