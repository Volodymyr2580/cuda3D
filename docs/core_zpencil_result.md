# Core Z-Pencil Prototype Result

## Status

Deferred.

## Reason

`CUDA3D_CORE_ZPENCIL_SHARED` is gated on Nsight Compute evidence that `p_core` is memory-bound or dominated by z-neighbor reloads. On 2026-06-07, NCU counters became available after the profiler permission fix.

NCU shows `p_core` has very high compute/memory throughput (`~96.94%`), moderate DRAM throughput (`~42.44%`), and long-scoreboard stalls. This makes a z-pencil/shared-memory prototype technically plausible.

However, PML tile kernels remain the first target because `p_pml_tile` is the largest sampled launch and PML fusion has the clearer dataflow-reuse hypothesis. Start `p_core` z-pencil only after the PML z-slab prototype fails its 5% threshold or shifts the bottleneck toward `p_core`.

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
