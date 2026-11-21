# Source provenance and clean-room boundary

## Historical anchor

- Legacy repository: local archival copy of `JustinBen0131/my-ai-rep`
- Source commit inspected: `1fb5766f47881e633d1d5f59d32e9841eab1aac7`
- Committed tree: `253216b64670bb30b711a26177303691d5878191`
- Inspection date: 2026-09-01

This repository was authored from a clean design contract. It does not copy the legacy `.git` directory, uncommitted working tree, web UI, cloud infrastructure, databases, raw datasets, model artifacts, or secret-bearing configuration.

## Concepts retained

- route requests between lower-cost and stronger model tiers;
- evaluate routing against paired model outcomes;
- make cost and quality tradeoffs visible;
- emit aggregate telemetry suitable for later analysis.

## Deliberately excluded

- `chat-ui/`, `landing-page/`, and `server/` because ownership is mixed and the audited authentication/key-handling design is unsafe;
- AWS pipeline and rendered deployment files;
- tracked `.env`, private-key, SQL dump, bytecode, logs, user/auth records, raw prompts/responses, weights, caches, and legacy Git objects;
- unsafe pickle deserialization, random or constant fallback predictions, simulated training labels, and unvalidated energy telemetry;
- hard-coded savings, customer testimonials, deployment, drift, or production claims.

## External lineage still requiring review

The legacy prototype referenced RouteLLM concepts and public datasets/models including LMSYS Chatbot Arena, MMLU, Nectar, BERT, LLaMA, and Mixtral. This clean package ships no external dataset or model content. Any future data adapter must record the exact source revision, license, redistribution constraints, privacy treatment, and derived-artifact policy.

## Private Arena evaluation adapter

The local evaluation uses a cached LMSYS Chatbot Arena Conversations Arrow file with SHA-256 `cbc3a4a44f8621bbca3f254274beed822e61fa9e2c8672d4b3e2a01b99dedaf1` and 33,000 source rows. `tools/build_arena_proxy.py` fails closed on source drift and constructs a private prompt-only proxy cohort under these rules:

- English battles with a decisive winner;
- one historical strong-tier model and one historical weak-tier model under the pinned heuristic mapping;
- identical normalized user turns in both arms;
- OpenAI-moderation, RoBERTa, and T5 toxicity flags excluded, with T5 score at most 0.5;
- assistant responses never copied into the cohort or used as features;
- source question identifiers are not published in aggregate reports.

The resulting private cohort has 576 rows, 570 prompt groups, 379 strong-tier wins, and 197 weak-tier wins. Its SHA-256 is `6f18f8fea9414c215750c27b0ec696d1c6095f517826c993f2496eb8ec092ebe`. The dataset remains ignored and private. Only prompt-free aggregate reports, model parameters, and hashed split identifiers may enter a release candidate after the external dataset license and derived-artifact policy are reviewed.

## Release gates

1. Justin selects a project license; until then no license is granted.
2. Verify authorship and attribution for every adapted implementation.
3. Audit and rotate any credentials ever committed to the legacy repository.
4. Run a redacted secret scan over files, Git objects, and built distributions.
5. Freeze a redistributable or private evaluation dataset with hash, split, and schema receipts.
6. Run deterministic tests and the reproducibility gate from a clean clone.
7. Pin dependency, model, dataset, and pricing versions for any empirical claim.
8. Manually review the report for personal data, cloud identifiers, raw prompts, and unsupported claims.

### Gate status on November 21, 2026

The repository became public on this date. The gates above are kept as written. Their status is recorded here so that the page stays true for a public reader. A gate is marked open until there is evidence that it is met.

| Gate | Status |
|---|---|
| 1. License | Open. No license has been granted and all rights are reserved. |
| 2. Authorship and attribution | Open. Awaits the author's review of every adapted implementation. |
| 3. Legacy credentials | Open. This concerns the legacy repository, which is not part of this package. This repository has a new Git history and holds none of that material. The check belongs to the credential owner. |
| 4. Secret scan | In part. `tools/secret_gate.py` scans the tracked files in CI on every push. A scan of Git objects and of built distributions is open. |
| 5. Dataset freeze | In part. The private cohort is not redistributed. Its source hash and cohort hash are recorded above. The public reference run is synthetic and ships its manifest and split manifest under `artifacts/synthetic_reference/`. |
| 6. Deterministic tests and reproducibility gate | Run from a clean checkout by the CI workflow on every push, for Python 3.10, 3.11 and 3.12. |
| 7. Version pins | The package has no third-party runtime dependency. Model, dataset and pricing pins stay required for any new empirical claim. |
| 8. Manual report review | Open. Awaits the author's review. |
