# ADR 001 — Model hydration via init container instead of Triton's native S3 loader

- **Status:** Accepted
- **Date:** Week 3 (Kubernetes/EKS milestone)
- **Context owner:** Infrastructure / DevOps Engineer

## Context

Triton Inference Server can pull model repositories directly from S3, but the
loader runs inside the server process at startup: it adds tens of seconds to pod
ready time, retries opaquely, and a failed fetch leaves Triton in a half-loaded
state that readiness probes cannot distinguish cleanly. Quora's production
experience on EKS (the reference architecture for this project) reported the same
startup latency and flakiness.

## Decision

Hydrate models with a dedicated **init container** (`amazon/aws-cli s3 sync`) into
an `emptyDir` volume mounted by the Triton container. The main container only ever
sees a complete, local model repository.

Consequences:

* Pod startup is deterministic: init either succeeds (container starts) or fails
  (pod restarts with clear `Init:Error` status) — no partial-load states.
* The S3 bucket remains the single source of truth; rollouts pick up new versions
  automatically (`scripts/sync-models-to-s3.sh` publishes them).
* Credentials come from IRSA (`eks.amazonaws.com/role-arn` on the ServiceAccount),
  never from baked-in keys.
* Cost: ~40 s cold start per GPU pod, which we absorb via `maxUnavailable: 0`
  rolling updates and PodDisruptionBudgets.

## Alternatives considered

1. **Native Triton S3 loader** — rejected: opaque failures, slow readiness, races
   with dynamic version policy during rollout.
2. **Bake models into the container image** — rejected: multi-GB images, every
   model refresh triggers a full rebuild/push/pull cycle.
3. **PersistentVolume / EFS share** — rejected for now: cross-AZ latency and cost;
   revisit if model size makes per-pod sync wasteful.
