# CUDA3D Current-Best Frontier Final Report

## Summary

Current formal best:

```text
candidate                     current_best_v_pml_len16
WP speedup vs zmem             1.222023x
Gradient speedup vs zmem       1.206588x
Elapsed speedup vs zmem        1.118261x
max rel L2                    6.384336e-07
```

Estimated total speedup vs the earliest project version is about `2.3x`, but
this depends on the historical `1.8x` anchor and is not a direct same-session
measurement.

## Closed Frontier

- Ordinary exact-CUDA micro prototypes are closed outside the micro-bank policy.
- Direct cooperative-grid K=2 temporal is closed.
- Cluster-local DSM K=2 temporal is closed.
- Same-GPU multi-rank oversubscription is closed.
- Host/setup micro routes are closed without a new `>=5%` measured hotspot.

## New Artifacts

- `docs/current_best_v_pml_len16_release.md`
- `docs/original_vs_current_best_20260609.md`
- `reports/original_vs_current_best_20260609/summary.json`
- `docs/micro_bank_policy.md`
- `docs/multigpu_shot_batching_plan.md`
- `tools/run_multigpu_batching.py`
- `docs/precision_relaxation_policy_proposal.md`
- `docs/next_scope_decision_menu.md`

## Micro-Bank Policy

Small, low-risk, composable changes may be banked if they pass correctness,
repeat performance, resource, and ablation gates.  Heavy ownership/fusion/
temporal/cluster routes remain non-bankable and require a separate model.

## Multi-GPU Plan

True multi-GPU shot batching remains the best route for application throughput,
but it requires `>=2` visible GPUs.  The safe runner refuses to run on a
single-GPU platform and does not fake speedup through same-GPU oversubscription.

## Precision Proposal

Mixed precision is proposal-only.  No CUDA code should be written until the
user explicitly approves Tier 1 or Tier 2 tolerance.

## Recommended Next Action

Choose one:

- wait for `>=2` GPUs and run true multi-GPU batching;
- approve a precision-relaxation tier and start a CPML-memory feasibility study;
- stop the CUDA-core sprint at current-best and use the package in reporting.
