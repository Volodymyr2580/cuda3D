# Codex Feedback To Pro: Core 2-Step Fused Debug Stage

Date: 2026-06-07 16:46 +0800

## Executive Summary

Codex implemented and validated `CUDA3D_CORE_2STEP_FUSED_INTERIOR` stage 1-3.

Result:

```text
design doc: done
meaningful 1GPU case: done
debug-only fused p2 predictor: done
p2-shift correctness: pass, rel_l2=0, max_abs=0
receiver output vs zmem_reference: pass, rel_l2=0, max_abs=0
speedup claim: none
```

Commit:

```text
9115859 feat(core2step): validate fused debug predictor
```

GitHub branch:

```text
exp/core-2step-interior-prototype
```

Server `/work/wenzhe/cuda3D` has been synchronized to the same commit.

## What Was Added

Files:

```text
docs/core_2step_fused_design.md
docs/core_2step_fused_result.md
tools/create_core_2step_meaningful_case.py
benchmarks/cases/core_2step_meaningful_1gpu/
```

CUDA changes:

```text
CUDA3D_CORE_2STEP_FUSED_INTERIOR
CUDA3D_CORE_2STEP_FUSED_DEBUG
cuda_fd3d_p_core_2step_fused_predict_ns
```

The debug predictor runs before baseline `p_core`, while `d_p0` is still `p(t-1)` and `d_p1` is still `p(t)`. It locally recomputes first-step values needed by `p(t+2)` and writes only an auxiliary debug buffer. It does not modify the main wavefield.

`CUDA3D_CORE_2STEP_FUSED_DEBUG` gates the predictor to the requested dump timestep, so the meaningful `nt=501` case remains practical for correctness validation.

## Meaningful Case

Case:

```text
benchmarks/cases/core_2step_meaningful_1gpu
```

Properties:

```text
ny=160 nx=160 nz=96 nt=501 npml=12
core_points=2033152
fused_eligible_points=922560
eligible_ratio=0.453758
source_in_fused_region=no
receivers_in_fused_region=0
fused_region=30:90,30:154,30:154
```

This satisfies the requested `>40%` eligible-region gate.

## Validation

Debug binary SHA256:

```text
593e58ccc415e60f9a5700c40280ed8bd4c2a77c945f67588668c582c9f5e42b
```

Run:

```text
benchmarks/runs/core_2step_fused_debug_meaningful_20260607_163942
```

p2-shift report:

```text
benchmarks/reports/core2step_fused_p2_shift_meaningful_20260607_163942/comparison.md
```

Result:

```text
pass=True
count=922560
rel_l2=0.0
max_abs=0.0
max_rel=0.0
rms=0.0
```

Output report:

```text
benchmarks/reports/core2step_fused_output_meaningful_20260607_163942/comparison.md
```

Result:

```text
pass=True
files=1
rel_l2=0.0
max_abs=0.0
max_rel=0.0
```

Restored zmem binary SHA256:

```text
86617a8a4bb549e916c0681d7833b85b8516ceb8293104b9f1b2cd734a6f77ba
```

## Timing Note

Debug timing:

```text
dump it=0: WP=0.078205s, Gradient=0.085542s, elapsed=0:02.66
dump it=1: WP=0.078484s, Gradient=0.085960s, elapsed=0:02.16
```

Restored zmem baseline timing:

```text
WP=0.064357s, Gradient=0.073288s, elapsed=0:02.57
```

The debug build is slower as expected. It is a correctness probe and must not be interpreted as a performance candidate.

## Important Stage-4 Finding

While preparing `CUDA3D_CORE_2STEP_FUSED_COMMIT`, I found a critical correctness hazard:

```text
p0 is both the old field input p(t-1) and the output p(t+1).
```

Therefore, we cannot safely create a true fused commit kernel by simply embedding the current debug helper inside `cuda_fd3d_p_core_ns`.

Reason:

```text
1. p2 needs old p0 values to locally reconstruct first-step neighbor values.
2. baseline p_core writes new p0 in-place.
3. within one kernel there is no grid-wide synchronization.
4. another CTA may overwrite p0 before this CTA finishes reading old p0.
```

So this unsafe shortcut is not allowed:

```text
compute p2 from global p0/p1
write p0(t+1) in the same kernel
```

unless all p2 dependencies are staged from old p0/p1 before any global p0 write can race with them.

## Implication

The real fused commit kernel must use one of these designs:

1. CTA-local shared/register staging:
   - load old `p0(t-1)` and `p1(t)` for the logical outer tile;
   - compute `p0_local(t+1)`;
   - compute inner `p2(t+2)` only from staged/local first-step values;
   - write global `p0(t+1)` after p2 dependencies are safe.

2. Separate read-only old-p0 buffer:
   - keep an immutable copy of `p(t-1)` for the fused kernel;
   - this avoids the in-place race but adds memory traffic and likely weakens performance.

3. Cooperative/grid-synchronized kernel:
   - probably not suitable for current MPI/shot workload unless launch constraints are carefully controlled.

Given `R=7`, a practical shared-memory tile is the hard part. A naive 3D logical tile large enough to retain a meaningful `2R` inner region can exceed shared-memory limits quickly.

## Recommendation

Before writing `CUDA3D_CORE_2STEP_FUSED_COMMIT`, Pro should decide the intended stage-4 architecture:

```text
A. true shared-staged CTA tile, with explicit tile dimensions and shared-memory budget;
B. accept a temporary old-p0 read-only buffer to prove commit schedule first;
C. stop fused commit if the required shared tile is too large for useful occupancy.
```

Codex recommendation:

```text
Do not implement a naive in-place fused commit.
First design a shared-memory budget table for candidate logical tiles.
Only then implement CUDA3D_CORE_2STEP_FUSED_COMMIT.
```

The existing `cuda_fd3d_p_core_ns_skip_region` also still performs its region skip after shared z-tile fill. Any commit path must move fully-covered block skip before shared load, otherwise the next-step skip will not remove the intended cost.

## Next Task Request

Please provide a concrete stage-4 tile plan:

```text
logical outer tile dimensions
inner tile dimensions after 2R margin
shared arrays required
shared bytes per CTA
threads per CTA
estimated occupancy on sm_120
whether old p0 is staged or copied
whether the first commit prototype may use a read-only old-p0 buffer
```

Once this is specified, Codex can implement the commit kernel and run:

```text
meaningful correctness
meaningful repeat speedup
perf_1gpu_6shots repeat
NCU kernel comparison
```
