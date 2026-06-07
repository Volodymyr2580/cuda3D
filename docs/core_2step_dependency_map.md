# Core 2-Step Dependency Map

Date: 2026-06-07

Branch: `exp/core-2step-interior-prototype`

## Goal

Map the exact timestep dependencies before implementing any temporal blocking. The first target is a single-GPU, strict-core-interior two-step pressure prototype. It must not change PML, source injection timing, receiver extraction timing, or MPI behavior.

## Main Time Loop

The active order in `src/rem_fd.cu` is:

```text
for it in [0, nt):
  1. V update
     cuda_fd3d_v_pml_tile_ns or cuda_fd3d_v_pml_ns

  2. Core pressure update
     cuda_fd3d_p_core_ns(d_p0, d_p1, d_cw2, ...)

  3. PML pressure update
     cuda_fd3d_p_pml_tile_ns or cuda_fd3d_p_pml_ns

  4. ZMEM_IN_P auxiliary swap
     swap d_memory_dz and d_memory_dz_next

  5. Source injection and receiver extraction
     if nr <= BS:
       lint3d_inject_bell_extract_gpu_zz(d_p0, ...)
     else:
       lint3d_inject_bell_gpu(d_p0, ...)
       lint3d_extract_gpu_zz(d_p0, ...)

  6. Time-level pressure swap
     ptr = d_p0
     d_p0 = d_p1
     d_p1 = ptr
```

The debug harness added for this prototype dumps strict-core-interior `p0` and `p1` after step 5 and before step 6. The dump stage is named:

```text
post_inject_pre_swap
```

## Array Read/Write Map

| Step | Reads | Writes | Notes |
|---|---|---|---|
| V update | `p1`, PML coefficients, `memory_dy/dx/dz` | `vy/vx/vz`, `memory_dy/dx/dz` | PML-only velocity update. |
| Core pressure | `p0`, `p1`, `cw2` | `p0` | Strict core excluding `npml + CorePmlMargin` boundary. |
| PML pressure | `p0`, `p1`, `vy/vx/vz`, PML coefficients, `memory_dyy/dxx/dzz`, `memory_dz` | `p0`, `memory_dyy/dxx/dzz`, `memory_dz_next` under `ZMEM_IN_P` | Runs every timestep. |
| ZMEM swap | `memory_dz`, `memory_dz_next` pointers | pointer swap | Required by `CUDA3D_PML_ZMEM_IN_P`. |
| Source injection | `src`, `d_bell`, source indices/weights | `p0` | Happens after pressure update and before pressure swap. |
| Receiver extraction | `p0`, receiver indices/weights | `d_est` | Records every timestep from post-injection `p0`. |
| Pressure swap | `d_p0`, `d_p1` pointers | pointer swap | Makes post-injection `p0` become next step's `p1`. |

## p_core Stencil Radius

The core pressure kernel uses `CoreStencilRadius`, now tied to:

```cpp
#define CUDA3D_CORE_STENCIL_RADIUS 7
enum { CoreStencilRadius = CUDA3D_CORE_STENCIL_RADIUS };
```

The first safe two-step guard margin is:

```text
2 * CUDA3D_CORE_STENCIL_RADIUS = 14 cells
```

This is a minimum guard for a two-step stencil dependency. More margin may be used for safety.

## Source And Receiver Timing

Source injection is not optional in the normal loop. It occurs after both core and PML pressure updates:

```text
pressure p0 complete -> inject source into p0 -> extract receivers from p0 -> swap p0/p1
```

Receiver extraction also happens every timestep and must not be skipped. Any temporal prototype that skips the intermediate `t+1` extraction changes user-visible output.

## MPI / Halo Status

The first prototype is single-rank only. The debug dump code checks `MPI_COMM_WORLD` size and exits if it is not `1`.

The generated `core_2step_interior_1gpu` case uses:

```text
gpu_count = 1
mpirun -np 1
```

Existing RTX 5090 single-GPU cases:

- `benchmarks/cases/correctness`
- `benchmarks/cases/perf_1gpu`
- `benchmarks/cases/perf_1gpu_6shots`
- `benchmarks/cases/profile_1gpu`
- `benchmarks/cases/core_2step_interior_1gpu`

## Safe Region Definition

Coordinates below are local `nby/nbx/nbz` coordinates, including the `nbd` PML offset but excluding the extra `radius` array padding.

Baseline core region:

```text
z: [nbd + CorePmlMargin, nbz - nbd - CorePmlMargin)
x: [nbd + CorePmlMargin, nbx - nbd - CorePmlMargin)
y: [nbd + CorePmlMargin, nby - nbd - CorePmlMargin)
```

Default strict two-step region:

```text
margin = 2 * CUDA3D_CORE_STENCIL_RADIUS

z: [nbd + CorePmlMargin + margin, nbz - nbd - CorePmlMargin - margin)
x: [nbd + CorePmlMargin + margin, nbx - nbd - CorePmlMargin - margin)
y: [nbd + CorePmlMargin + margin, nby - nbd - CorePmlMargin - margin)
```

The debug harness records the exact chosen region in `*_core_meta.txt`.

## Source/Receiver In Region

The first generated case places source and receivers at shallow `z = dz`, which maps to local `z = nbd + 1`. With default `npml = 8`, `CorePmlMargin = 4`, and margin `14`, the strict interior starts at local `z = 26`, so source and receivers are outside the blocked region.

The debug metadata records:

```text
source_in_region
receivers_in_region
```

The first prototype may proceed only when both are zero.

## Conclusions

- First version can exclude source/receiver on the generated `core_2step_interior_1gpu` case.
- Existing shallow-acquisition cases likely also exclude source/receiver in z, but this must be verified by debug metadata, not assumed.
- If a future case has source/receiver inside the strict region, create a new case or add explicit source/receiver integration before commit mode.
- Minimum guard margin: `2 * CUDA3D_CORE_STENCIL_RADIUS = 14`.
- MPI temporal blocking is out of scope until single-rank debug-only and commit-mode correctness pass.

