# Codex Feedback To Pro: Core 2-Step Interior Prototype

Date: 2026-06-07 16:07:53 +0800

Branch: `exp/core-2step-interior-prototype`

Latest commits:

```text
616280f feat(core2step): add commit correctness prototype
248ebba feat(core2step): validate debug p2 predictor
```

## Executive Summary

The core two-step interior line is now correctness-validated, but not yet a speedup.

We proved that a strict core-interior `p(t+2)` prediction is numerically identical to baseline on safe regions, and also proved that a commit schedule can skip the same region on the next timestep and restore the predicted values without changing receiver outputs or interior dumps.

However, the current `CUDA3D_CORE_2STEP_COMMIT_INTERIOR` implementation is slower because it computes `p(t+2)` in a separate prediction kernel and then copies it back. It proves scheduling correctness only. It should not be treated as the final optimization path.

The next useful Pro-level design task is a fused two-step core kernel that computes `p(t+1)` and strict-interior `p(t+2)` in one kernel while reusing loaded data.

## What Was Implemented

1. `CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE`
   - Debug-only strict-interior `p(t+2)` prediction.
   - Dumps `p0_core`, `p1_core`, and `p2_core`.
   - Validates `p2(it)` against next-step baseline `p0(it+1)` using `--mode p2-shift`.

2. `CUDA3D_CORE_2STEP_COMMIT_INTERIOR`
   - Single-rank correctness prototype.
   - Uses `cuda_fd3d_p_core_ns_skip_region` to skip a validated strict interior region on the next timestep.
   - Uses `cuda_core2step_copy_region` to commit previous `p2` into `p0`.
   - Preserves baseline ordering for PML, source injection, receiver extraction, and `p0/p1` swap.

3. Shared region validation helper
   - `Core2StepRegion` centralizes strict-region computation, source-in-region checks, receiver-in-region checks, and cropped-subdomain rejection.

## Validation Evidence

### Debug-only `p(t+2)` predictor

Run:

```text
benchmarks/runs/core_2step_p2_debug_20260607_152450
```

Report:

```text
benchmarks/reports/core2step_p2_shift_compare_20260607_152450/comparison.md
```

Result:

```text
pass = True
files compared = 5
rel_l2 = 0.0 for all compared timesteps
max_abs = 0.0 for all compared timesteps
```

### Commit prototype minimal output correctness

Run:

```text
benchmarks/runs/core_2step_commit_correctness_20260607_154200
```

Report:

```text
benchmarks/reports/core2step_commit_correctness_20260607_154200/comparison.md
```

Result:

```text
pass = True
files compared = 1
rel_l2 = 0.0
max_abs = 0.0
baseline WP = 0.001391 s
candidate WP = 0.001453 s
```

### Commit prototype strict-interior dump correctness

Run:

```text
benchmarks/runs/core_2step_commit_dump_compare_20260607_154800
```

Report:

```text
benchmarks/reports/core2step_commit_dump_compare_20260607_154800/comparison.md
```

Result:

```text
pass = True
files compared = 17
rel_l2 = 0.0 for all compared files
max_abs = 0.0 for all compared files
baseline dump files = 23
candidate dump files = 23
baseline WP = 0.004412 s
candidate WP = 0.004554 s
```

### Full correctness case

The default strict region failed on cropped shots, as designed:

```text
ERROR invalid CUDA3D_CORE_2STEP commit region z=[26,54) x=[26,1) y=[26,1), n=(80,27,27)
```

Rerun with explicit safe region:

```text
CUDA3D_CORE_2STEP_REGION=26:54,12:15,12:15
```

Report:

```text
benchmarks/reports/core2step_commit_correctness_full_region_20260607_155200/comparison.md
```

Result:

```text
pass = True
files compared = 6
rel_l2 = 0.0 for all compared files
max_abs = 0.0 for all compared files
baseline WP = 0.012992 s
candidate WP = 0.015493 s
```

## Key Conclusion

Correctness is solved for the current strict-region scheduling prototype.

Performance is not solved. The current standalone predict+copy design is expected to be slower because:

- it launches an extra prediction kernel;
- it launches an extra copy kernel;
- `cuda_fd3d_p_core_ns_skip_region` still pays much of the original block/shared-memory load cost before returning for skipped points;
- the tested strict regions are small because acquisition cropping can shrink per-shot domains.

Therefore, do not continue tuning the current standalone predict+copy path as a performance optimization.

## Request To Pro

Please design the next fused core two-step temporal blocking kernel.

Required constraints:

1. Single GPU / single MPI rank only for the first version.
2. Strict core interior only.
3. Source and receiver must be outside the fused region, or the kernel must reject the case.
4. PML, guard region, source injection, receiver extraction, and pointer swap must remain baseline-compatible.
5. The fused kernel should compute `p(t+1)` and strict-interior `p(t+2)` together, not by launching a standalone `p2` prediction kernel.
6. It should reuse data from the first-step stencil calculation where possible.
7. It should avoid doing shared-memory loads for blocks that are fully covered by already-committed prediction.
8. It must pass the existing output comparison and strict-interior dump comparison before performance testing.

Suggested design direction:

- Start with a tile/block list of safe strict-interior blocks.
- Build a fused kernel for tiles whose full stencil neighborhood remains inside the strict region.
- Keep boundary/guard tiles on the baseline path.
- For each fused tile, compute `p(t+1)` into global `p0`, compute `p(t+2)` for a smaller inner tile into `p2`, then commit/skip only where the dependency radius is proven safe.
- Measure whether reducing global loads and launch work beats the added complexity. If the fused kernel cannot show at least `>=5%` repeat speedup on a meaningful case, stop this route.

## Files To Inspect

```text
AGENTS.md
docs/core_2step_interior_design.md
docs/core_2step_interior_result.md
include/inc3D/single_solver.h
src/rem_fd.cu
src/single_solver.cu
tools/compare_core_interior_dumps.py
tools/compare_outputs.py
```

## Current State

Remote server:

```text
/work/wenzhe/cuda3D
branch = exp/core-2step-interior-prototype
HEAD = 616280f
```

The server binary was restored to non-debug zmem after testing.

Local note: `AGENT_LOG.md` may contain unrelated uncommitted local log residue from another task; it was intentionally not committed.
