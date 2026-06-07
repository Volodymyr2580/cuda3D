# PML Fusion Prototype Result

## Status

Stopped after implementation and validation.

## Reason

`CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE` was originally gated on Nsight Compute evidence. The prototype has now been implemented and tested on the RTX 5090 server.

NCU shows:

- `p_pml_tile`: largest sampled launch among target kernels, long-scoreboard/wait/short-scoreboard pressure.
- `v_pml_tile`: strong long-scoreboard pressure.
- Both kernels have no spill and moderate DRAM throughput.

The scoped z-slab prototype was correct, but the best validated repeat speedup was below the required continuation threshold:

```text
perf_1gpu_6shots_repeat WP speedup = 0.956846x
threshold to continue              = 1.05x
```

See `docs/pml_fused_zslab_result.md` for the full implementation, validation, NCU, and decision report.

## Prototype Definition Tested

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

Validation completed:

- debug dump step 0/1/2
- correctness
- `perf_1gpu`
- `perf_1gpu_6shots`
- `perf_1gpu_6shots repeat`

Acceptance decision:

- Repeat speedup versus `zmem_reference` was `<5%`.
- `CUDA3D_PML_FUSED_ZSLAB_SKIP_V_OWNED` was not enabled.
- Do not continue this limited PML z-slab split unless a new design removes substantially more velocity/pressure global-memory round-trip work.
