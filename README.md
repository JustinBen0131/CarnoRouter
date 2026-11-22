# Carno Router

Carno Router is a clean-room, offline-first reconstruction of an earlier multi-model routing prototype. It now contains two linked measurement systems: a risk-controlled cost-quality router and a Scientific Research Autonomy Evaluation Lab for grading evidence-grounded agent traces under explicit safety, reproducibility, calibration, and budget contracts.

This repository is intentionally separate from the legacy Git history. It contains no provider calls, database dump, UI, credential storage, raw assistant responses, or historical marketing claims. The only cloud surface is an optional, never-executed AWS Lambda shard contract under `cloud/aws/` that runs the provider-free stress benchmark; it is validated locally and creates no cloud resource. It is a measurement kernel, not a production-service claim.

## What is defensible now

- A pure routing decision function with fail-closed budget behavior, explicit abstention, and policy/evaluation provenance.
- A leakage-checked benchmark contract that uses prompt text only as a feature.
- Group-and-time-aware train/calibration/test splits.
- Train-only champion selection across four deterministic sparse-model specifications.
- Five-fold group-stratified out-of-fold Platt calibration.
- Always-cheap, always-strong, deterministic-random, and oracle baselines.
- Calibration-only threshold selection with a simultaneous one-sided 95% bootstrap quality-risk band across all candidate thresholds.
- Cost, quality, escalation, regret, ROC-AUC, PR-AUC, Brier score, ECE, slice diagnostics, distribution-shift warnings, and paired group-bootstrap intervals.
- A prompt-free, content-hashed scoring bundle that can replay the selected model without a provider call.
- Privacy-safe telemetry: default events omit stable request/prompt identifiers; incident correlation requires a caller-held 32-byte HMAC key.
- Byte-reproducible JSON/HTML reports from a synthetic offline fixture.
- A 24-task, six-slice research-autonomy environment covering evidence integrity, capability safety, scientific judgment, reproducibility, adversarial context, and long-horizon operations.
- Trace graders for decision correctness, citation precision/recall, forbidden actions, replay hashes, budgets, calibration, prompt-injection resistance, and OOD abstention.
- Mutation sentinels that prove the evaluator accepts a contract-conformant reference and rejects reckless, overcautious, and evidence-blind behaviors.
- A sealed 28,800-trial blinded stress campaign whose candidate traces, receipts, and unblinding key are persisted so it can be re-graded from artifacts alone.
- An independent reference oracle (`make oracle-audit`) with its own declarative invariant table, exact-rational legacy scoring, held-out attack grammar, and metamorphic relations; it re-grades every sealed trial and reports how many distinct grader outcomes the campaign actually exercises.

The checked-in reference report is an engineering smoke test. The private Arena proxy run is an evaluation case study. Neither establishes current-provider dollar savings, production reliability, customer adoption, or live-routing fitness. The final deployment receipt blocks live autorouting when the evidence gates do not pass.

## One-command local validation

```bash
make test
make benchmark
make research-lab
make research-stress
make oracle-audit
make aws-contract
make reproduce
make secret-gate
```

The runtime package uses only the Python standard library. No network access or API key is required.

## Evaluate research-agent traces

Generate the deterministic public contract fixture:

```bash
make research-lab
```

Or grade another JSONL trace set against a sealed suite:

```bash
PYTHONPATH=src python3 -m carno_router research-eval \
  --suite path/to/suite.json \
  --traces path/to/traces.jsonl \
  --output artifacts/my_research_eval
```

The checked-in fixture validates evaluator discrimination only. It is not a live-model leaderboard. Even when the evaluator passes, `live_agent_deployment` remains `BLOCKED` until a fresh blinded benchmark provides pinned model identities and exact token, latency, cost, tool, error, and grader receipts. See [the evaluation contract](docs/research_autonomy_lab.md).

## Benchmark your own paired outcomes

Prepare a CSV matching [the benchmark contract](docs/benchmark_contract.md), then run:

```bash
PYTHONPATH=src python3 -m carno_router benchmark \
  --data path/to/paired_outcomes.csv \
  --pricing path/to/pinned_pricing.json \
  --output artifacts/my_run \
  --seed 1729 --bootstrap 5000 --selection-bootstrap 5000 \
  --max-quality-drop 0.02
```

Replay a sealed prompt-only scorer without executing a provider:

```bash
PYTHONPATH=src python3 -m carno_router score \
  --bundle artifacts/my_run/model_bundle.json \
  --policy risk_controlled \
  --prompt "Summarize the experimental design."
```

For dollar claims, the pricing file must be a dated, pinned provider-price snapshot and the paired outcomes must come from a fresh, frozen evaluation. The report labels all savings as modeled under that snapshot.

## Publication status

Public since November 21, 2026 as a clean-room research artifact. Live autorouting remains blocked by the current evidence receipt. The status of each release gate is recorded in [SOURCE_PROVENANCE.md](SOURCE_PROVENANCE.md), and the security policy is in [SECURITY.md](SECURITY.md).

Released under the [MIT License](LICENSE). The license covers the code and the synthetic artifacts in this repository. [NOTICE.md](NOTICE.md) lists what is not redistributed here.

Written by Justin Bennett with the help of AI coding assistants, including ChatGPT.
