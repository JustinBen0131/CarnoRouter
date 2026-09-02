# Carno cost-quality benchmark

Status: **illustrative_synthetic_smoke**

This report evaluates an offline routing policy against frozen paired outcomes. It contains no provider calls and no raw prompts.

## Selected routed policy

| Metric | Routed | Always cheap | Always strong |
|---|---:|---:|---:|
| Mean quality | 0.8661 | 0.5778 | 0.8661 |
| Mean cost (normalized_cost_units) | 20 | 1 | 20 |
| Strong-call rate | 100.0% | 0.0% | 100.0% |
| Preference accuracy | 77.8% | 22.2% | 77.8% |

- Calibration-selected threshold: `0.04`
- Point-estimate challenger threshold: `0.97`
- Quality delta versus always-strong: `+0.0000`
- Modeled cost savings versus always-strong: `0.0%`
- Paired group-bootstrap 95% savings interval: `0.0%` to `0.0%`
- Test ROC-AUC: `1.0`
- Test Brier score: `0.00108413`
- Point-estimate challenger savings / quality delta: `21.1%` / `+0.0044`

## Decision gate

- Offline research artifact: **PASS**
- Portfolio evidence: **PASS_WITH_EXPLICIT_LIMITS**
- Live autorouting: **BLOCKED**
- Shift warnings: `label_prevalence_shift_train_to_test, small_test_cohort`

The primary policy is selected with a calibration-only, one-sided 95% prompt-group bootstrap quality-risk bound. The point-estimate policy is retained as a challenger and is not silently promoted.

## Evidence boundary

- All savings are modeled under the exact pinned pricing snapshot.
- Synthetic-fixture results are engineering smoke evidence, not provider performance.
- Offline proxy preference is not absolute model quality and cannot establish production parity.
- Provider latency, historical spend, deployment reliability, and customer adoption are not measured.
- The cohort uses historical model generations and is not evidence about current frontier models.
- No demographic fairness conclusion is supported because protected-attribute labels are absent.
