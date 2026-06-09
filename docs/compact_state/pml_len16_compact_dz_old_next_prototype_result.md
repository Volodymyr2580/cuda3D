# PML Len16 Compact DZ Old/Next Prototype Result

Date: 2026-06-09

## Summary

`CUDA3D_PML_LEN16_COMPACT_DZ16_OLD_NEXT` was implemented as a macro-default-off exact-FP32 prototype, but it is rejected as a performance candidate.

The prototype is numerically correct in the tested cases and passes compact writer coverage, but its `perf_1gpu_6shots` repeat speedup is below the `>=1.02x` small-candidate gate.

## Implementation

The prototype extends the accepted pressure len16 half-warp path:

- Adds compact `dz_old16` / `dz_next16` buffers for accepted pressure len16 central z positions.
- Keeps the residual pressure-PML path on the existing full-array `memory_dz` / `memory_dz_next`.
- Keeps len16 halo z-cache fill on the full-array helper with `write_owned=false`.
- Swaps compact old/next buffers together with the full `d_memory_dz` / `d_memory_dz_next` swap.
- Mirrors compact central writes back to the full array only for debug builds.

The macro depends on:

```text
CUDA3D_PML_LEN16_COMPACT_STATE
CUDA3D_PML_LEN16_COMPACT_DZ16_OLD_NEXT
```

The tested normal build flags also included the current exact-FP32 best line:

```text
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_CPML_VMEM_DISABLE_MPI
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
```

## Validation

Build:

- Normal candidate build: pass.
- Debug fill build with `CUDA3D_PML_LEN16_COMPACT_DZ_DEBUG_FILL` and `CUDA3D_PML_ZMEM_DEBUG_FILL`: pass.

Coverage:

- `profile_1gpu` debug fill run: pass.
- Run directory:
  - `benchmarks/runs/profile_1gpu_compact_dz_old_next_debugfill_profile_20260609_184932`

Correctness:

- Candidate:
  - `benchmarks/runs/correctness_compact_dz_old_next_correctness_probe_20260609_185104`
- Comparison:
  - `reports/compact_state/compare_compact_dz_old_next_correctness_vs_current_best`
- Result:
  - pass
  - 6 output files
  - max rel L2 `0`

`perf_1gpu_6shots` probe:

- Candidate:
  - `benchmarks/runs/perf_1gpu_6shots_compact_dz_old_next_perf6_probe_20260609_185134`
- Comparison:
  - `reports/compact_state/compare_compact_dz_old_next_perf6_probe_vs_current_best`
- Result:
  - pass
  - 6 output files
  - max rel L2 `0`

## Repeat Performance

Repeat summary:

```text
reports/compact_state/compact_dz_old_next_perf6_repeat_summary.json
```

Baseline runs:

```text
benchmarks/runs/perf_1gpu_6shots_compact_current_best_perf6_a2_20260609_163148
benchmarks/runs/perf_1gpu_6shots_compact_current_best_perf6_b2_20260609_163151
benchmarks/runs/perf_1gpu_6shots_compact_current_best_perf6_c2_20260609_163154
```

Candidate runs:

```text
benchmarks/runs/perf_1gpu_6shots_compact_dz_old_next_perf6_a_20260609_185349
benchmarks/runs/perf_1gpu_6shots_compact_dz_old_next_perf6_b_20260609_185353
benchmarks/runs/perf_1gpu_6shots_compact_dz_old_next_perf6_c_20260609_185356
```

| Metric | Baseline Mean | Candidate Mean | Speedup |
|---|---:|---:|---:|
| WP computing time | `2.004982s` | `1.969942s` | `1.017787x` |
| Gradient TIME all | `2.118638s` | `2.087878s` | `1.014733x` |

All three repeat output comparisons passed with max rel L2 `0`.

## Decision

Rejected as a performance candidate.

Reason:

- The repeat WP speedup is `1.017787x`, below the `>=1.02x` disabled-candidate keep gate.
- The route is correct, but compacting `memory_dz` old/next for accepted len16 central z positions does not move enough traffic or latency after the existing direct-fill len16 stack.

The code remains macro-default-off as a documented negative result. It must not be treated as current-best.

## Next Direction

Do not continue narrow `dz16 old/next` compact-state micro-tuning unless new profiler evidence shows `memory_dz` old/next traffic has become a dominant bottleneck.

The next exact-FP32 optimization direction should move to a larger pressure-PML memory ownership model or another profiler-supported route with a modeled `>=5%` repeat speedup ceiling.
