# Core 2-Step Interior Design

Date: 2026-06-07

## Status

Design and debug harness phase. Commit-mode temporal blocking is not implemented yet.

## Prototype Name

```text
CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE
```

Supporting debug macros:

```text
CUDA3D_CORE_2STEP_DEBUG_DUMP
CUDA3D_CORE_2STEP_INTERIOR_COMPARE
CUDA3D_CORE_2STEP_INTERIOR_DEBUG
CUDA3D_CORE_2STEP_DISABLE_MPI
```

The debug path has two levels:

- `CUDA3D_CORE_2STEP_DEBUG_DUMP` dumps post-injection, pre-swap strict-interior state.
- `CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE` additionally computes a debug-only `p(t+2)` prediction into an auxiliary buffer and dumps it as `p2_core`.

The prototype macro must not change the main pressure buffers or receiver output until the shifted dump comparison passes.

## Region Ownership

First version:

- strict core interior only;
- guard margin at least `2 * CUDA3D_CORE_STENCIL_RADIUS`;
- no source or receiver inside the region;
- single GPU / single MPI rank only.

Everything outside the strict region remains baseline:

- PML velocity update;
- PML pressure update;
- core boundary and guard region;
- source injection;
- receiver extraction;
- pressure pointer swap.

## Two-Step Data Contract

The target proof is:

```text
debug-only predicted p(t+2) strict interior == baseline p(t+2) strict interior
```

First step `p(t+1)` must still be written to global `p0` so that:

- receiver extraction at `t+1` is exact;
- source injection at `t+1` is visible;
- PML and boundary behavior remain baseline;
- the normal `p0/p1` swap remains valid.

The second-step strict-interior result may initially be written to an auxiliary buffer such as:

```text
d_p2_core_interior
```

Debug-only mode must not change the main output.

## Commit Mode Prototype

`CUDA3D_CORE_2STEP_COMMIT_INTERIOR` is a correctness prototype, not a performance implementation.

Current behavior:

1. timestep 0 runs the baseline core pressure kernel;
2. after source injection, the prototype predicts strict-interior `p(t+2)` into `d_p2_core_debug`;
3. on the next timestep, `cuda_fd3d_p_core_ns_skip_region` skips that strict region;
4. after PML pressure update, `cuda_core2step_copy_region` commits the predicted values back into `p0`;
5. the normal source injection, receiver extraction, and `p0/p1` swap still run in baseline order.

This proves the commit scheduling contract but does not reduce arithmetic yet, because the prediction is still a separate kernel. A real speedup requires a fused two-step core kernel that reuses loaded data while computing `p(t+1)` and `p(t+2)`.

The next performance implementation must solve:

1. computing `p(t+1)` and strict-interior `p(t+2)` in one fused kernel;
2. reusing shared-memory/global loads across the two temporal steps;
3. skipping whole covered core blocks rather than only returning per point after shared-memory loads;
4. keeping guard-region/PML/source/receiver ordering identical to baseline;
5. handling cropped subdomains where the default strict region is empty or too small.

## Source/Receiver Rule

The first case requires:

```text
source_in_region = 0
receivers_in_region = 0
```

The debug harness computes these values from host-side `src0_indx/rec0_indx` and writes them into each `*_core_meta.txt` file.

If either is non-zero, the first prototype must stop before commit mode.

## Debug Harness

Compile with:

```bash
-DCUDA3D_CORE_2STEP_DEBUG_DUMP
```

Runtime environment:

```bash
CUDA3D_CORE_2STEP_DUMP_DIR=...
CUDA3D_CORE_2STEP_DUMP_STEP=-1
CUDA3D_CORE_2STEP_MARGIN=14
CUDA3D_CORE_2STEP_REGION=
```

`CUDA3D_CORE_2STEP_DUMP_STEP=-1` dumps every timestep. A non-negative value dumps only that timestep.

`CUDA3D_CORE_2STEP_REGION` supports either:

```text
z0:z1,x0:x1,y0:y1
z0,z1,x0,x1,y0,y1
```

The region coordinates are local `nbz/nbx/nby` coordinates including PML but excluding the extra `radius` padding.

Dump files:

```text
rank_<rank>_shot_<shot>_it_<it>_p0_core.bin
rank_<rank>_shot_<shot>_it_<it>_p1_core.bin
rank_<rank>_shot_<shot>_it_<it>_p2_core.bin  # only with CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE and it+1 < nt
rank_<rank>_shot_<shot>_it_<it>_core_meta.txt
```

The dump stage is post-injection and pre-swap.

## Comparison

Use:

```bash
python3 tools/compare_core_interior_dumps.py \
  --baseline <baseline_dump_dir> \
  --candidate <candidate_dump_dir> \
  --out <report_dir>
```

For the debug-only two-step proof, compare the predicted `p2(it)` against the next baseline `p0(it+1)`:

```bash
python3 tools/compare_core_interior_dumps.py \
  --baseline <dump_dir> \
  --candidate <dump_dir> \
  --out <report_dir> \
  --mode p2-shift
```

Metrics:

- finite check;
- relative L2;
- max absolute error;
- max relative error at max absolute error;
- RMS error.

Default tolerance:

```text
rel_l2 <= 1e-5
abs <= 1e-7 for all-zero baseline
```

## Next Implementation Step

After the debug-only `p(t+2)` comparison passes, the next step is to decide whether commit mode is worth implementing and how to gate it so it only covers strict interior timesteps with no source/receiver ownership conflicts.
