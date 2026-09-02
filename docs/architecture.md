# Architecture

```text
local paired outcomes
        |
        v
schema + leakage gate ---> group/time split receipt
        |                         |
        v                         v
four sparse candidates ---> train-only OOF champion
                                  |
                                  v
                     OOF Platt calibration + final fit
                                  |
                                  v
              simultaneous calibration risk-control band
                                  |
                                  v
 always-cheap / strong / random / length / oracle / routed
                                  |
                  +---------------+---------------+
                  |                               |
                  v                               v
       shift + slice + error audit        sealed scoring bundle
                  |                               |
                  +---------------+---------------+
                                  v
              deterministic reports, hashes, and deployment gate
```

The scoring, routing, execution, and reporting concerns are separate. Nothing in the routing path can call a provider. The runtime policy is fail closed on invalid scores and budget violations unless downgrade behavior is explicitly requested. A future live executor must be an optional adapter with an explicit `--live` gate, cost/retry envelope, fresh-provider evidence, shadow evaluation, and separate security review.
