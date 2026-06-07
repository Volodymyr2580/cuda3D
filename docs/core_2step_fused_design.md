# Core 2-Step Fused Interior Design

Date: 2026-06-07

## Status

Design for the next active architecture path.

This document supersedes standalone `CUDA3D_CORE_2STEP_COMMIT_INTERIOR` as a performance direction. The standalone commit path remains useful as a correctness proof, but it is not a speedup path.

## Macros

```text
CUDA3D_CORE_2STEP_FUSED_INTERIOR
CUDA3D_CORE_2STEP_FUSED_DEBUG
CUDA3D_CORE_2STEP_FUSED_COMMIT
CUDA3D_CORE_2STEP_FUSED_DISABLE_MPI
```

`CUDA3D_CORE_2STEP_FUSED_INTERIOR` must stay disabled by default.

## Stencil Radius

`CoreStencilRadius` is derived from:

```text
include/inc3D/cu_common.h:
CUDA3D_CORE_STENCIL_RADIUS = 7

src/single_solver.cu:
enum { CoreStencilRadius = CUDA3D_CORE_STENCIL_RADIUS };
```

Therefore:

```text
R = 7
conservative two-step margin = 2R = 14
```

## Fused Tile Contract

The fused kernel owns a logical tile, not necessarily one CUDA thread block worth of points.

Definitions:

```text
outer tile:
    points for which the fused kernel computes p(t+1)

inner tile:
    a strict subset of the outer tile for which the fused kernel computes p(t+2)
```

First implementation uses:

```text
inner = outer shrunk by 2R in z/x/y
```

This is conservative. It avoids depending on first-step values outside the current fused tile.

Later experiments may test `R` shrink only, but only after a dump comparison proves that all second-step dependencies are satisfied.

## Cross-CTA Dependency Rule

There is no inter-block synchronization inside a CUDA kernel.

Therefore, first fused implementation must obey:

```text
p(t+2) for an inner point may only read p(t+1) values produced inside the same fused logical tile.
```

It must not read another CTA's newly written global `p0(t+1)`.

Consequence:

```text
1. load p1(t) and p0(t) data needed for the outer tile;
2. compute p0_local(t+1) for the outer tile;
3. synchronize inside the CTA or local tile loop stage;
4. compute p2(t+2) only for inner points whose full radius-R p0_local neighborhood is available;
5. write p0(t+1) to global p0;
6. write p2(t+2) to an auxiliary buffer.
```

If a logical tile is processed by multiple CTAs, it cannot use CTA-local shared memory as the only synchronization mechanism for p2. First implementation should keep one fused logical tile per CTA or split the tile so each CTA-owned inner region is self-contained.

## Shared Memory Layout

First prototype should use explicit staging:

```text
p1_outer_halo:
    p1(t) values needed to compute first-step p0_local(t+1)

p0_prev_outer:
    previous p0(t) values for the second-order update

p0_local_outer:
    first-step p(t+1) result for the outer tile
```

The existing baseline core kernel already uses a z-direction shared tile. The fused version must be more deliberate: x/y are no longer tiny dimensions if the tile is intended to have a meaningful inner region.

The first debug implementation may use a simple layout even if it is not optimal. The performance implementation must report shared-memory bytes, occupancy, and inner/outer ratio.

## Global Writes

`p(t+1)` must still be written to global `p0`.

Reason:

- receiver extraction at every timestep must remain exact;
- source injection modifies `p0` after the pressure update;
- PML and guard-region behavior must remain baseline-compatible;
- the existing `p0/p1` swap order must not change.

`p(t+2)` writes to:

```text
d_p2_fused
```

or an equivalent auxiliary buffer.

## Commit And Skip

Commit-mode must only run after debug-only fused p2 comparison passes.

At timestep `t`:

```text
fused kernel computes p(t+1) for outer tile
fused kernel computes p(t+2) for inner tile
p(t+1) is written to global p0
p(t+2) is written to d_p2_fused
```

At timestep `t+1`:

```text
PML/source/receiver ordering remains baseline-compatible
baseline core work skips only the validated inner tile
d_p2_fused is committed to p0 for that same inner tile
```

The current standalone commit prototype copies with a separate kernel. The fused performance path should avoid this if possible. Acceptable first commit implementation may still use a small copy for correctness, but it must be reported separately and should not be tuned as the main route.

## Odd/Even And Final Timestep

Rules:

- timestep 0 may run baseline first-step behavior while the fused kernel generates the first valid `p(t+2)`;
- if no valid prediction exists, fallback baseline;
- final timestep must not produce an unused out-of-range prediction;
- odd/even behavior must preserve existing `p0/p1` swap timing.

## Source And Receiver Exclusion

First version requires:

```text
source_in_fused_region = false
receivers_in_fused_region = false
```

If either is true:

```text
debug mode: reject and report
performance mode: fallback baseline for that shot/tile
```

The existing `Core2StepRegion` helper can be extended for fused regions.

## Cropped Subdomain Fallback

Acquisition cropping can shrink local domains dramatically. Existing evidence:

```text
default region failed on cropped correctness shot:
ERROR invalid CUDA3D_CORE_2STEP commit region z=[26,54) x=[26,1) y=[26,1), n=(80,27,27)
```

Therefore:

- if the fused region is empty, fallback baseline;
- if the eligible ratio is too small, fallback baseline or report as not meaningful;
- meaningful performance testing requires a dedicated case with large x/y/z local domain.

## PML And Guard Compatibility

The fused route only owns strict core interior.

Baseline remains responsible for:

- PML velocity update;
- PML pressure update;
- guard region near PML/core boundary;
- source injection;
- receiver extraction;
- pressure pointer swap.

No fused kernel may update PML memory arrays.

## Difference From Standalone Predict/Copy

The stopped standalone path did this:

```text
baseline p_core
extra p2 prediction kernel
extra copy kernel
skip-region p_core that still paid much of the block/shared-memory load cost
```

The fused path must remove at least part of that overhead:

```text
single fused core kernel computes p(t+1) and p(t+2)
first-step data are reused for second-step work
fully committed blocks can skip before shared-memory fill
standalone p2 prediction launch is removed
copy/commit cost is minimized or fused into skip-aware work
```

## First Benchmark Gate

Add:

```text
benchmarks/cases/core_2step_meaningful_1gpu
```

The case must report:

```text
total core points
fused eligible points
eligible ratio
source/receiver in fused region
```

Gate:

```text
debug-only fused p2-shift correctness must pass first
meaningful case repeat speedup must be >= 5% to continue
perf_1gpu_6shots repeat speedup must be >= 2% to keep as a real candidate
```

If the meaningful case is slower or below 5%, stop the fused route and report.
