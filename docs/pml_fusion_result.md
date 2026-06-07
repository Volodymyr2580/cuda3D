# PML Fusion Prototype Result

## Status

Not started.

## Reason

`CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE` is gated on Nsight Compute evidence. On 2026-06-07, NCU failed with `ERR_NVGPUCTRPERM` because `RmProfilingAdminOnly: 1`.

Starting a PML fusion rewrite without knowing whether `p_pml_tile` and `v_pml_tile` are memory-bound, branch-bound, instruction-bound, or occupancy-bound would violate the project architecture gate.

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
