# Benchmark contract

## Required CSV columns

| Column | Meaning |
|---|---|
| `example_id` | Unique immutable row identifier |
| `group_id` | Duplicate/paraphrase family; one group may appear in only one split |
| `timestamp` | ISO-8601 observation time used for chronological splitting |
| `prompt` | The only model feature |
| `cheap_quality` | Frozen outcome quality for the cheaper model, in `[0,1]` |
| `strong_quality` | Frozen outcome quality for the stronger model, in `[0,1]` |
| `cheap_input_tokens` | Non-negative integer usage |
| `cheap_output_tokens` | Non-negative integer usage |
| `strong_input_tokens` | Non-negative integer usage |
| `strong_output_tokens` | Non-negative integer usage |
| `cheap_latency_ms` | Optional non-negative latency |
| `strong_latency_ms` | Optional non-negative latency |

Unknown columns are rejected. Fields whose names contain `answer`, `response`, `completion`, `winner`, or `provider_key` are rejected explicitly. This prevents the legacy leakage mode in which both candidate answers were concatenated into the router feature.

## Splitting

Groups are ordered by their earliest timestamp and assigned 70%/15%/15% to train, calibration, and test. The loader proves zero group overlap and unique example IDs. Vocabulary, candidate-model selection, model fitting, and Platt-calibrator fitting use train only. Candidate selection uses five-fold group-stratified out-of-fold train predictions. Threshold choice uses calibration only. Final metrics and intervals use test only once.

## Objective

The primary threshold minimizes modeled cost on calibration data subject to a simultaneous one-sided 95% prompt-group bootstrap upper bound on absolute quality drop relative to always-strong. The familywise max-deviation band covers every candidate threshold, so searching the grid cannot silently rely on a pointwise interval. The default budget allows at most `0.02` mean-quality loss. Always-strong remains the exact fail-safe policy. A less conservative point-estimate challenger is reported but never silently promoted.

## Metrics

- mean selected quality;
- modeled normalized or currency cost;
- escalation rate;
- absolute quality delta vs always-strong;
- modeled cost savings vs always-strong;
- regret vs a per-row oracle;
- Brier score and expected calibration error for “strong is better”;
- paired group-bootstrap confidence intervals for quality delta and savings.
- prompt-length and deterministic-random baselines;
- train-to-test model-pair and label-prevalence shift diagnostics;
- prompt-length, time, and model-pair slices with small-slice warnings;
- an explicit deployment decision with blockers.

Dollar savings are valid only when both outcomes and a dated provider pricing snapshot are pinned. Latency claims require measured per-row latency under a declared environment.

The current Arena adapter uses historical human-preference data and normalized costs. It therefore supports evaluation-method claims only. It cannot support a current-provider savings or deployment claim.
