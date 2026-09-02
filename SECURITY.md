# Security policy

## Design boundary

The core package is offline and does not execute model providers. Routing consumes a caller-supplied score; benchmarking consumes a local CSV. Default telemetry omits request identifiers and stable prompt hashes, buckets prompt length and score, and records only policy/pricing provenance plus decision metadata. Private incident correlation requires a caller-held key of at least 32 bytes and emits key-scoped HMACs.

## Prohibited material

Do not commit API keys, `.env` files, private keys, certificates, database dumps, cloud identifiers, user records, raw provider logs, model caches, or raw assistant responses. `.env.example`, if added later, may contain variable names and non-routable placeholders only.

## Safe failure behavior

Missing, non-finite, or out-of-range scores never become a random prediction or silent `0.5`. The decision is a typed abstention unless the caller explicitly configured a budget-valid fallback. A cost-cap violation also abstains by default; downgrading a request judged to need the stronger tier is an explicit opt-in action.

## Legacy credentials

This repository does not contain or inspect legacy credential payloads. Before any public release, the credential owner must independently verify that historically committed credentials are revoked or rotated. Deleting them from a new commit would not sanitize old history; this project therefore uses a new Git history.
