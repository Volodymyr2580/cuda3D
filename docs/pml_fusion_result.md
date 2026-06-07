# PML Fusion Prototype Result

## Status

Ready to start.

## Reason

`CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE` was originally gated on Nsight Compute evidence. The profiling counter issue has now been fixed and NCU data is available.

NCU shows:

- `p_pml_tile`: largest sampled launch among target kernels, long-scoreboard/wait/short-scoreboard pressure.
- `v_pml_tile`: strong long-scoreboard pressure.
- Both kernels have no spill and moderate DRAM throughput.

This supports a scoped z-slab dataflow/reuse prototype. It does not justify a broad region-wide fusion yet.

## Prototype Definition For Later

Macro:

```text
CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE
```

Scope:

- Only pure z-PML face.
- x/y must be inside the core region.
- edge/corner/x-face/y-face/residual stays on `zmem_reference` generic path.
- No source injection or extraction timing changes.
- Not enabled by default.

Validation required:

- debug dump step 0/1/2
- correctness
- `perf_1gpu`
- `perf_1gpu_6shots`
- `perf_1gpu_6shots repeat`

Acceptance:

- Stop if repeat speedup versus `zmem_reference` is `<5%`.
- If `>=5%`, design x/y residual fusion.
- If `>=10%`, write a region-wide PML fusion design document.
