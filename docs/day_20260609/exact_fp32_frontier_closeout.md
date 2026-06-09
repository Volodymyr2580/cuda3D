# Exact-FP32 Frontier Closeout

Date: 2026-06-09

## Summary

The current exact-FP32 single-GPU CUDA-core sprint should stop ordinary micro-prototyping.

The latest compact-state follow-up, `CUDA3D_PML_LEN16_COMPACT_DZ16_OLD_NEXT`, is numerically correct but rejected by the performance gate.  Earlier frontier gates already reject the remaining ordinary exact-CUDA routes.  The next high-precision work must therefore be design/model first, not another CUDA kernel experiment.

## Current Best

Formal current best remains:

```text
current_best_v_pml_len16
```

Formal same-session evidence:

```text
docs/day_20260609/pro_handoff_current_best_frontier.md
reports/day_20260608/formal_vpmlen16_table_20260608_2359/summary.json
```

Key numbers:

```text
WP speedup vs zmem        1.222023x
Gradient speedup vs zmem  1.206588x
Elapsed speedup vs zmem   1.118261x
max rel L2                6.384336e-07
```

## Compact-State Closeout

`CUDA3D_PML_LEN16_COMPACT_STATE` dzz16-only:

- correctness and `perf_1gpu_6shots` output comparisons passed.
- final repeat WP speedup vs current-best: `1.011842x`.
- rejected below the `>=1.02x` disabled-candidate keep gate.

`CUDA3D_PML_LEN16_COMPACT_DZ16_OLD_NEXT`:

- normal build passed.
- debug fill `profile_1gpu` passed and confirmed compact `dz_next16` ownership coverage.
- correctness and `perf_1gpu_6shots` repeat output comparisons passed with max rel L2 `0`.
- repeat WP speedup vs current-best: `1.017787x`.
- repeat Gradient speedup vs current-best: `1.014733x`.
- rejected below the `>=1.02x` small-candidate gate.

Reports:

```text
docs/compact_state/pml_len16_compact_dzz16_commit_result.md
docs/compact_state/pml_len16_compact_dz_old_next_prototype_result.md
reports/compact_state/compact_dzz16_commit_perf6_summary.json
reports/compact_state/compact_dz_old_next_perf6_repeat_summary.json
```

Decision:

```text
compact-state micro route closed
```

Do not continue narrow len16 compact `dzz16`, `dz16`, or old/next state micro-tuning unless a new profiler profile proves that those exact state arrays have become a dominant bottleneck.

## Ordinary Exact-CUDA Frontier

The ordinary exact-CUDA route frontier is already closed:

```text
docs/day_20260608/ownership_frontier_gate.md
reports/day_20260608/ownership_frontier_gate.json
```

Decision:

```text
ordinary_exact_cuda_frontier_exhausted_for_micro_routes
ordinary CUDA allowed prototype count = 0
```

Do not reopen:

- pressure writeback syntax tweaks: `p0 __ldg`, local `new_mem`, cache-policy, z-cache fill.
- residual pressure branch-only split, length-32 split, length-23 descriptor, exact active-point descriptor.
- v-PML descriptor expansion or current-geometry vx/vy component split.
- direct z-face VP fusion or shared-VP retry.
- current p-core block/register/shared-plane family.
- K=2 ordinary CUDA temporal prototype.
- CUDA Graph, launch aggregation, or async-stream scheduling for the current single-GPU case.

## Cluster / Cross-CTA Frontier

RTX 5090 supports cooperative and cluster launch, but the modeled cluster-local K=2 temporal route is also rejected:

```text
docs/day_20260609/cluster_cooperative_frontier_gate.md
docs/day_20260609/cluster_local_ownership_model.md
reports/day_20260609/cluster_cooperative_frontier_gate.json
reports/day_20260609/cluster_local_ownership_model.json
```

Key result:

```text
cooperative grid ceiling         2040 blocks
previous K=2 required blocks     70688 blocks
cooperative over-capacity        34.6510x
best DSM cluster tile estimate   0.9498x sampled-main
```

Decision:

```text
reject_cluster_local_temporal_cuda_prototype
cluster CUDA prototype allowed = false
```

Do not write a cluster CUDA prototype unless a new ownership model beats the DSM byte gate after halo, synchronization, source, receiver, and PML reconciliation costs.

## Current Allowed High-Precision Work

For exact-FP32, the only allowed CUDA-core work is design/model first:

1. A fundamentally different pressure or wave-step ownership representation.
2. A concrete synchronization primitive with a byte/synchronization model.
3. A modeled `>=5%` `perf_1gpu_6shots` repeat speedup ceiling before any CUDA prototype.
4. A correctness proof that preserves the current exact-FP32 tolerance: rel L2 `<=1e-5`, no NaN/Inf.

If no such model exists, stop exact CUDA-core prototyping at `current_best_v_pml_len16`.

## Scope Changes Outside This Gate

These are not part of the current high-precision CUDA-core sprint, but remain possible if the user explicitly chooses them:

- relaxed precision track with a new tolerance policy.
- true multi-GPU batching on a multi-GPU platform.
- application-level multi-shot scheduling.
- packaging and handoff of the current best.

## Next Automation Action

The next heartbeat should not start a new ordinary CUDA micro-prototype.

Recommended next action:

```text
write a short Pro/agent handoff update, or open a design-only model for a fundamentally different ownership representation
```

Any implementation attempt must first name the new ownership primitive and show a modeled `>=5%` repeat-speedup ceiling.
