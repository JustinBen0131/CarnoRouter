# Evaluation decision: do not deploy this router

## Decision

The offline research artifact passes its reproducibility and evidence-contract gates. Live autorouting is blocked.

## Frozen evaluation

- Audited source: 33,000 human-preference battles.
- Strict private cohort: 576 rows across 570 prompt groups.
- Chronological group split: 403/86/87 rows for train/calibration/test, with zero group overlap.
- Model selection: four prompt-only sparse candidates evaluated with group-stratified out-of-fold train predictions.
- Calibration: five-fold group-stratified out-of-fold Platt scaling.
- Policy selection: 102 thresholds under a simultaneous one-sided 95% max-deviation bootstrap band with 5,000 calibration replicates.
- Final uncertainty: 5,000 paired prompt-group bootstrap replicates on the untouched test slice.

## Measured result

The train-only champion was a 2,048-feature word unigram/bigram logistic model. Its held-out test ROC-AUC was `0.40198864`, below the predeclared `0.65` gate. The train-to-test model-pair total-variation distance was `0.35073158`, above the `0.20` warning threshold.

The risk-controlled policy escalated `98.85%` of test requests, yielding only `1.09%` modeled savings under normalized 20:1 costs and a mean-quality delta of `-0.01149` versus always-strong. The 95% paired-bootstrap interval for the quality delta was `[-0.03488, 0.00000]`. The point-estimate challenger appeared cheaper on calibration but lost `0.10345` mean quality on test.

## Why this is a successful result

The benchmark falsified the deployment story while preserving the underlying research question. It detected weak discrimination, temporal/model-pair shift, threshold-selection risk, and inadequate test size before any provider traffic or public savings claim. The next valid experiment is a larger, task-stratified evaluation with current provider pairs, pinned current pricing, measured latency, and shadow-mode acceptance criteria.

## Claim boundary

This result supports claims about benchmark design, leakage control, calibration, risk-aware policy selection, uncertainty, drift diagnosis, fail-closed routing, and evidence governance. It does not support a current-provider savings, latency, reliability, adoption, or production claim.
