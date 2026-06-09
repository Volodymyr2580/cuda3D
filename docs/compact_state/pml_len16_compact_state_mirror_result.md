# PML Len16 Compact-State Mirror Result

## Scope

This validates:

```text
CUDA3D_PML_LEN16_COMPACT_STATE_MIRROR
```

It is exact-FP32 and default-off.  The full-array CPML path remains
authoritative.  This mirror does not route computation through compact state and
is not a performance candidate.

## Build

Flags:

```text
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DCUDA3D_PML_ZMEM_IN_P
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
-DCUDA3D_CPML_VMEM_DISABLE_MPI
-DCUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
-DCUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
-DCUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
-DCUDA3D_PML_LEN16_COMPACT_STATE_MIRROR
```

Binary SHA256:

```text
3d284fac86d066d8ce09c4d1f0a7126714f198555ecb087350b5e27c6ae636b3
```

## Validation Runs

| case | run | result |
| --- | --- | --- |
| `smoke_1gpu` | `smoke_1gpu_compact_mirror_smoke_20260609_170322` | pass, 3 outputs |
| `correctness` | `correctness_compact_mirror_correctness_20260609_170323` | pass, 6 outputs |
| `perf_1gpu_6shots` probe | `perf_1gpu_6shots_compact_mirror_perf6_probe_20260609_170324` | pass, 6 outputs |

## Mirror Internal Check

The perf probe covered six shots.  For each shot, the mirror checked:

```text
it = 0, 1, 2, 1500
```

All reported:

```text
rel_l2=0.000000e+00
max_abs=0.000000e+00
bad=0
```

## Output Comparison

Mirror vs Phase 0 current-best correctness:

- pass: `true`
- max rel L2 across all outputs: `0`
- max abs across all outputs: `0`

Mirror vs Phase 0 current-best perf probe:

- pass: `true`
- max rel L2 across all outputs: `0`
- max abs across all outputs: `0`

Artifacts:

```text
reports/compact_state/compare_mirror_correctness_vs_current_best/comparison.md
reports/compact_state/compare_mirror_correctness_vs_current_best/comparison.json
reports/compact_state/compare_mirror_perf6_probe_vs_current_best/comparison.md
reports/compact_state/compare_mirror_perf6_probe_vs_current_best/comparison.json
```

## Decision

Phase 3 mirror passes.

Allowed next step:

```text
CUDA3D_PML_LEN16_COMPACT_STATE
```

The commit prototype must route accepted pressure len16 CPML state reads/writes
through compact arrays while keeping residual PML full-array fallback.

Acceptance remains:

- output rel L2 `<= 1e-5`,
- no NaN/Inf,
- `perf_1gpu_6shots` repeat x3,
- WP speedup `>=1.02x` to keep disabled candidate,
- WP speedup `>=1.05x` and positive Gradient speedup to promote.
