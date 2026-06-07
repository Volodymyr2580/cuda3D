# Post Shared-VP Fallback Plan

## Current State

The local PML z-face fusion family has now failed in three forms:

- direct p1 x/y second derivative;
- p-only shared pressure tile;
- staged-V shared velocity intermediate.

All were correctness-clean, so the issue is not numerical stability. The issue is that the pressure critical path gets slower than the saved global-memory traffic.

## Do Not Repeat

Do not continue these without new profiler evidence:

```text
CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY
CUDA3D_PML_ZFACE_SHARED_VP_DEBUG
CUDA3D_PML_ZFACE_SHARED_VP_STAGE_V
```

Do not start another PML z-face block-shape sweep. The failure is structural: large shared tiles reduce occupancy and add staging/synchronization cost.

## Allowed Fallback A: PML Compact-State Audit

Goal:

```text
Measure whether CPML state memory can be compacted or laid out to reduce traffic without fusing pressure math into a heavier kernel.
```

First audit only:

- enumerate `mem_dy`, `mem_dx`, `mem_dz`, `mem_dyy`, `mem_dxx`, `mem_dzz` allocation sizes;
- map which kernels read/write each state under `zmem_reference` and `CPML_VMEM_DOUBLE_BUFFER_ALL`;
- use NCU memory tables to identify whether state traffic is L2/DRAM limited or mostly cache-resident;
- propose layout changes only after evidence.

Stop gate:

- if state traffic is not a top pressure/velocity bottleneck, stop compact-state work.

## Allowed Fallback B: Global-Region Temporal Pipeline

Goal:

```text
Move away from narrow PML z-face local fusion and evaluate whether a global-region multi-step pipeline can reduce full-domain pressure/velocity memory traffic.
```

First phase must be design/profiling only:

- isolate the dominant non-PML kernel time in `perf_1gpu_6shots`;
- quantify full-domain wavefield read/write bytes per timestep;
- audit source injection, receiver extraction, boundary/PML coupling, and pointer swap hazards;
- design a minimal 2-step single-rank prototype only if the byte budget predicts `>=10%` WP speedup.

Stop gate:

- no prototype unless the byte budget and source/receiver audit are both favorable;
- no MPI temporal blocking in the first prototype;
- no full rewrite before a single-GPU meaningful-case gate.

## Current Scaffold To Keep

Keep:

```text
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
```

Reason:

- correctness-clean;
- same-session speedup about `1.0335x` WP;
- useful ownership discipline for future structural prototypes.

Do not promote it as the final target by itself because it does not meet the `>=5%` candidate gate.
