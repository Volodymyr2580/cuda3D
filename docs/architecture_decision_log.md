# Architecture Decision Log

## 2026-06-07

- Accepted `CUDA3D_PML_ZMEM_IN_P` as the RTX 5090 stable baseline.
- Stopped PML fused z-slab: correctness passed, but repeat performance was slower than `zmem_reference`.
- Stopped p_core z-pencil before implementation: source-level NCU showed the baseline already has a z/fast-dimension shared-memory tile.
- Opened the next architecture line: `CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE`.
- Stable tag: `stable-zmem-rtx5090-20260607`.

