# Pressure Triple-Buffer Pipeline Design

Date: 2026-06-07

## Status

Implemented as a macro-gated prototype on branch `exp/pressure-triple-buffer-pipeline`.

The standalone prototype passed correctness but did not clear the meaningful repeat performance gate. It remains default-off and is retained as a dataflow foundation for phase2 temporal pipeline work, not as a promoted baseline.

Final validation report:

```text
reports/triple_buffer_3h/final_3h_report.md
```

## Motivation

Current pressure update uses two buffers:

```text
p0 = p(t-1), overwritten in-place as p(t+1)
p1 = p(t)
```

This creates an old/new dataflow hazard for any future fused or multi-step temporal blocking: one CTA can overwrite `p0` while another CTA still needs old `p0`.

Triple buffer makes the roles explicit:

```text
p_prev = p(t-1), read-only
p_curr = p(t), read-only for pressure update
p_next = p(t+1), write target
```

The first goal is correctness and clean dataflow, not immediate speedup.

## Time-Step Schedule

Baseline-equivalent triple-buffer timestep:

```text
1. v_pml reads p_curr and updates velocities / velocity PML memory.
2. p_core writes core cells in p_next from p_curr and p_prev.
3. p_pml writes PML cells in p_next from p_curr, p_prev, velocity fields, and PML memory.
4. ZMEM_IN_P swaps d_memory_dz / d_memory_dz_next exactly where it does today.
5. source injection writes p_next.
6. receiver extraction reads p_next after source injection.
7. rotate:
       tmp    = p_prev
       p_prev = p_curr
       p_curr = p_next
       p_next = tmp
```

## Required Kernel Signature Change

Current:

```cpp
cuda_fd3d_p_core_ns(float *p0, float *p1, float *cw2, ...)
```

Required macro-gated triple-buffer form:

```cpp
cuda_fd3d_p_core_triple_ns(float *p_next,
                           const float *p_curr,
                           const float *p_prev,
                           const float *cw2,
                           ...)
```

Pressure PML kernels need the same separation:

```cpp
cuda_fd3d_p_pml_triple_ns(float *p_next,
                          const float *p_curr,
                          const float *p_prev,
                          ...)
```

Tile-list and zface pressure variants must either get triple-buffer variants or be disabled under `CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE` until implemented.

## Initialization

Initial conditions are zero in the current code:

```text
d_p0 = 0
d_p1 = 0
```

Triple buffer should initialize:

```text
p_prev = 0
p_curr = 0
p_next = 0
```

No special bootstrap is needed for the zero initial condition.

## Source And Receiver Semantics

Current order:

```text
pressure update -> source injection -> receiver extraction -> pointer swap
```

Triple-buffer mapping:

```text
pressure update writes p_next
source injection modifies p_next
receiver extraction reads p_next
rotate buffers
```

Receiver must not read `p_curr`.

## Does p_next Need Clearing?

In production, `p_next` should not need clearing if:

```text
p_core writes the strict core region
p_pml writes the PML/residual region
source injection only adds after pressure update
receiver extraction only reads written/interpolated cells
```

However, first debug implementation should support:

```text
CUDA3D_PRESSURE_TRIPLE_BUFFER_DEBUG_FILL
```

which fills `p_next` with a signaling pattern or NaN before each timestep, then runs correctness/dump checks. This can reveal stale reads or unwritten cells after rotation.

## PML And ZMEM_IN_P

`CUDA3D_PML_ZMEM_IN_P` uses a separate double buffer:

```text
d_memory_dz
d_memory_dz_next
```

Triple buffer should not change its math or swap timing.

Keep this order:

```text
p_pml writes pressure and d_memory_dz_next
optional ZMEM coverage check
swap d_memory_dz / d_memory_dz_next
source/receiver on p_next
pressure buffer rotation
```

## Debug Dump Mapping

Existing debug dumps use `p0` and `p1` names. For triple buffer:

```text
old p0 at dump point == p_next after pressure update, before source injection
old p1 at dump point == p_curr
new p_prev is not dumped by old tools
```

Recommended triple-buffer dump names:

```text
p_prev
p_curr
p_next
```

Compare tooling can support compatibility mode:

```text
baseline p0 <-> triple p_next
baseline p1 <-> triple p_curr
```

## Memory Footprint

Triple buffer adds one pressure buffer per active rank/shot:

```text
additional bytes = (ny + 2*(npml + radius)) *
                   (nx + 2*(npml + radius)) *
                   (nz + 2*(npml + radius)) *
                   sizeof(float)
```

`radius` here is the allocation pad in `cu_common.h`, currently `4`, not the core stencil radius `7`.

See:

```text
tools/pressure_buffer_memory_estimate.py
docs/pressure_triple_buffer_memory.md
```

## Rollback Plan

All implementation must be macro-gated:

```text
CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE
CUDA3D_PRESSURE_TRIPLE_BUFFER_DEBUG
CUDA3D_PRESSURE_TRIPLE_BUFFER_DISABLE_MPI
CUDA3D_PRESSURE_TRIPLE_BUFFER_DEBUG_FILL
```

Default build remains zmem_reference.

Rollback is:

```text
compile without CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE
```

No baseline kernel math should be removed in the first implementation.

## Acceptance Gates

Correctness:

```text
smoke_1gpu pass
debug dump step 0/1/2 vs zmem_reference pass
correctness rel_l2 <= 1e-5
no NaN/Inf
output file count/size identical
```

Performance:

```text
perf_1gpu_6shots repeat slowdown <= 2%:
    accept as disabled dataflow-clean candidate

slowdown 2% to 5%:
    keep branch, diagnose overhead

slowdown > 5%:
    stop and report memory/dataflow overhead
```

Actual standalone result:

```text
correctness: pass, rel_l2=0
perf_1gpu_6shots repeat: about 1.0045x WP speedup vs zmem_reference
decision: keep default-off, do not promote, use only as phase2 temporal-pipeline foundation
```
