# Case study: rebuilding Carno into a defensible routing benchmark

## Problem

The historical prototype explored routing prompts between models with different capability and cost. A 2026 clean-room audit found that its strongest portfolio idea was buried under unsafe secrets/history, answer leakage, mislabeled comparisons, placeholder fallbacks, non-reproducible evaluation, and hard-coded marketing numbers.

## Intervention

I separated the decision kernel from provider execution and rebuilt the evaluation contract around frozen paired outcomes. The new package uses prompt-only features, group/time splits, train-only model selection, group-stratified out-of-fold calibration, a simultaneous calibration risk bound, fail-closed routing semantics, pinned pricing, deterministic baselines, shift and slice diagnostics, and paired group-bootstrap uncertainty. It emits a byte-reproducible receipt, a sealed scoring bundle, and a static recruiter-readable report without a network call.

## Engineering judgment

The project does not pretend the legacy result was production evidence. It treats credential rotation, provenance, external data licensing, and fresh paired provider evaluation as release gates. Synthetic results are labeled as smoke-test evidence; real cost claims require a new pinned run.

## Current result

The strict Arena proxy contains 576 safe prompt-only battles across 570 prompt groups from a 33,000-row audited source. The untouched chronological test slice found ROC-AUC `0.402`, model-pair total-variation shift `0.351`, and no defensible material savings inside the quality-risk budget. The system therefore marks the offline research artifact `PASS_WITH_EXPLICIT_LIMITS` and live autorouting `BLOCKED`. That negative deployment decision is the result: the measurement system prevented a weak router from becoming a marketing claim.
