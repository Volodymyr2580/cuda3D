# Architecture Decision 2026-06-07

## Decision

Freeze `CUDA3D_PML_ZMEM_IN_P` as the RTX 5090 stable baseline and stop the two local-kernel prototype lines:

- stop `CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE`;
- do not implement `CUDA3D_CORE_ZPENCIL_SHARED`;
- start a controlled `CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE` feasibility path.

## Stable Baseline

The stable RTX 5090 build remains:

```bash
-O3 -arch=sm_120 --use_fast_math \
-DCUDA3D_PML_RECOMPUTE_Z \
-DCUDA3D_PML_TILE_LIST \
-DCUDA3D_PML_ZMEM_IN_P \
-DPmlTileBlockSize1=32 \
-DPmlTileBlockSize2=4 \
-DPmlTileBlockSize3=2
```

All future candidates must compare against `zmem_reference`.

## Rejected Lines

`CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE` was numerically correct but slower than `zmem_reference` in repeat testing:

```text
WP speedup       = 0.956846x
Gradient speedup = 0.961207x
wall speedup     = 0.985772x
```

This line split pressure-side work into another kernel without removing enough global-memory dependency.

`CUDA3D_CORE_ZPENCIL_SHARED` was rejected before implementation by source-level NCU. The current `p_core` already has:

```cpp
__shared__ float z_tile[PBlockSize3][PBlockSize2][PBlockSize1 + 2 * CoreStencilRadius];
```

and source counters show z-neighbor stencil terms are already shared-memory loads. The remaining `p_core` pressure is x/y global neighbor loads, center/global tile fill, final pressure traffic, and existing barrier cost.

## New Prototype Order

Approved order after this decision:

```text
1. stable/zmem baseline consolidation
2. CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE design + debug harness
3. single-GPU core-interior two-step pressure-only prototype
4. if success, source/receiver integration
5. only after that, PML/MPI extension feasibility
```

## Guardrails

Do not continue:

- PML fused z-slab expansion;
- `CUDA3D_PML_FUSED_ZSLAB_SKIP_V_OWNED`;
- `CUDA3D_CORE_ZPENCIL_SHARED`;
- p_core z-only shared-memory duplicate;
- p_core simple block sweep;
- PML tile/block/mask/prune sweep;
- `RECOMPUTE_X`, `RECOMPUTE_Y`, or `RECOMPUTE_XYZ`;
- full-domain temporal blocking;
- MPI temporal blocking;
- any change to source injection or receiver extraction timing.

## Rationale

Local CUDA micro-optimization has reached a point where the remaining plausible gain requires changing dataflow, not adding another shared-memory halo. The first temporal/dataflow experiment must be narrow, debug-first, and single-GPU only. Its first success criterion is not speed; it is proving that a predicted `t+2` strict-interior pressure field can match the baseline interior field.

