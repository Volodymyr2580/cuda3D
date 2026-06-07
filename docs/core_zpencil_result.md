# Core Z-Pencil Prototype Result

## Status

Not started.

## Reason

`CUDA3D_CORE_ZPENCIL_SHARED` is gated on Nsight Compute evidence that `p_core` is memory-bound or dominated by z-neighbor reloads. On 2026-06-07, NCU hardware counters were unavailable because `RmProfilingAdminOnly: 1`.

The fallback static resource table shows `cuda_fd3d_p_core_ns` uses 48 registers, no spills, and a small shared-memory tile. That is not enough evidence to justify a deeper p_core rewrite.

## Prototype Definition For Later

Macro:

```text
CUDA3D_CORE_ZPENCIL_SHARED
```

Scope:

- z-pencil shared-memory prototype for `p_core`.
- x/y neighbors remain global loads.
- No full temporal blocking.
- No numerical order change.

Validation required:

- correctness
- `perf_1gpu_6shots repeat`
- kernel-level timing from NCU or Nsight Systems

Acceptance:

- `p_core` kernel itself must improve by `>=10%`.
- whole-job `perf_1gpu_6shots repeat` must improve by `>=2%`.
- Otherwise revert or keep macro-gated and disabled.
