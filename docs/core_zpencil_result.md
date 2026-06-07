# Core Z-Pencil Result

Date: 2026-06-07

## Status

Stopped before implementation.

## Summary

`CUDA3D_CORE_ZPENCIL_SHARED` was proposed as the next controlled experiment after the PML fused z-slab prototype failed its continuation threshold. The source-level NCU gate was completed first, as required.

The gate failed the hypothesis: the current `p_core` kernel already implements the proposed z/fast-dimension shared-memory pencil.

## Evidence

Current `cuda_fd3d_p_core_ns` contains:

```cpp
enum { CoreStencilRadius = 7 };
__shared__ float z_tile[PBlockSize3][PBlockSize2][PBlockSize1 + 2 * CoreStencilRadius];
```

The stencil lines for z-neighbor terms read from `z_tile`, and source-level NCU reports them as shared-memory loads:

```text
line 1110: z_tile local1 +/- 1 -> Shared(2), Load(2)
line 1114: z_tile local1 +/- 2 -> Shared(2), Load(2)
line 1118: z_tile local1 +/- 3 -> Shared(2), Load(2)
line 1122: z_tile local1 +/- 4 -> Shared(2), Load(2)
line 1126: z_tile local1 +/- 5 -> Shared(2), Load(2)
line 1130: z_tile local1 +/- 6 -> Shared(2), Load(2)
line 1134: z_tile local1 +/- 7 -> Shared(2), Load(2)
```

The remaining source-level long-scoreboard pressure is mostly:

- center/global fill into the existing shared tile;
- x/y global neighbor loads;
- final pressure update traffic.

The strongest barrier stall is attributed after the existing `__syncthreads()` and before the core-boundary return. This confirms that the current shared-memory path already has a synchronization cost.

See `docs/pcore_source_profile.md` for the detailed source counters.

## Decision

Do not add a `CUDA3D_CORE_ZPENCIL_SHARED` macro that duplicates the current kernel behavior.

Rejected variants:

- macro alias around the existing implementation;
- another z-only shared-memory tile with the same ownership model;
- candidate block shapes that only change `PBlockSize*` without adding new data reuse;
- p_core block-size sweep under the name of z-pencil.

## Validation

No candidate binary was created because the source-profile gate rejected the implementation premise.

After profiling, the server binary was restored to the stable non-`-lineinfo` zmem baseline:

```text
cfc502cf8a27038e54a1bdf1c3665b057a3b306046bd547a1baf70a204e17293  bin/cuda_3D_FM
```

Post-restore sanity run:

```text
case: benchmarks/cases/profile_1gpu
Gradient TIME all = 0.189999 s
WP computing time = 0.161748 s
ALL DONE
```

## Recommendation

Local `p_core` z-pencil work should be closed. Further `p_core` work should only proceed if it targets a genuinely different reuse axis, such as x/y plane reuse, register-streamed stencil restructuring, or a larger temporal/dataflow prototype.

The immediate fallback is recorded in `docs/temporal_blocking_feasibility.md`.
