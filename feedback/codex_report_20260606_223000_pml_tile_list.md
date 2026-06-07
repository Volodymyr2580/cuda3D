# Codex Feedback Report: pml_tile_list

## Hypothesis

The active PML kernels still launched over the full 3D domain and returned inside the kernel for core points. A host-generated tile list can avoid launching CTAs that are fully inside the inactive core region while preserving the existing full-domain `vx/vy/vz`, CPML memory arrays, and numerical formulas.

The first prototype intentionally avoids changing PML data layout. It only changes CTA scheduling:

```text
old: blockIdx maps directly to full-domain grid
new: blockIdx.x indexes a precomputed PmlTile{z0,x0,y0}
```

## Code Changes

- Modified `include/inc3D/cu_common.h`
  - Added `PmlTileBlockSize1/2/3`, default `32x4x2`, with `#ifndef` guards for compile-time sweeps.
- Modified `include/inc3D/single_solver.h`
  - Added `PmlTile` struct.
  - Added tile-list kernel declarations.
- Modified `src/single_solver.cu`
  - Added `cuda_fd3d_v_pml_tile_ns`.
  - Added `cuda_fd3d_p_pml_tile_ns`.
  - Both kernels copy the existing active formulas and only change `gtid1/2/3` mapping.
- Modified `src/rem_fd.cu`
  - Added host tile-list construction.
  - Added independent compile-time switches:
    - `CUDA3D_PML_TILE_LIST` enables both V and P tile-list paths.
    - `CUDA3D_PML_TILE_LIST_V` enables only `v_pml` tile-list.
    - `CUDA3D_PML_TILE_LIST_P` enables only `p_pml` tile-list.
  - Default build path remains the original kernels.

## Commands

Representative commands:

```bash
cd /work/wenzhe/cuda3D
source ./env_5090.sh

cd src
make -B -f makefile.rtx5090 test

make -B -f makefile.rtx5090 \
  NVFLAGS="-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_TILE_LIST -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2" \
  test

cd ..
python tools/run_benchmark.py --case correctness --tag pml_tile_32x4x2_final
python tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/correctness_rtx5090_baseline_20260606_002850/outputs \
  --candidate benchmarks/runs/correctness_pml_tile_32x4x2_final_20260606_222838/outputs \
  --out benchmarks/reports/correctness_pml_tile_32x4x2_final

python tools/run_benchmark.py --case perf_1gpu --tag pml_tile_32x4x2_final
python tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/perf_1gpu_rtx5090_baseline_20260606_002902/outputs \
  --candidate benchmarks/runs/perf_1gpu_pml_tile_32x4x2_final_20260606_222841/outputs \
  --out benchmarks/reports/perf_1gpu_pml_tile_32x4x2_final
```

## Correctness

| Metric | Value |
|---|---:|
| baseline | `correctness_rtx5090_baseline_20260606_002850` |
| candidate | `correctness_pml_tile_32x4x2_final_20260606_222838` |
| output files | 6 |
| rel L2 | `0.000000e+00` for all files |
| max abs | `0.000000e+00` |
| NaN/Inf | none |
| pass/fail | pass |

## Performance

Baseline:

```text
perf_1gpu_rtx5090_baseline_20260606_002902
WP computing time = 0.545397 s
Gradient TIME all = 0.576524 s
```

Final candidate:

```text
perf_1gpu_pml_tile_32x4x2_final_20260606_222841
v_tiles = 23100
p_tiles = 22188
block = 32x4x2
WP computing time = 0.539543 s
Gradient TIME all = 0.569638 s
```

| Metric | Baseline | Candidate | Speedup |
|---|---:|---:|---:|
| WP computing time | 0.545397s | 0.539543s | 1.0109x |
| Gradient TIME all | 0.576524s | 0.569638s | 1.0121x |

Best observed run during sweep:

```text
perf_1gpu_pml_tile_32x4x2_20260606_222244
WP computing time = 0.538781 s
Gradient TIME all = 0.568863 s
speedup = 1.0123x by WP
```

## Sweep Summary

| Variant | Mode | WP | Result |
|---|---|---:|---|
| `32x4x2` | V+P tile-list | 0.538781s / 0.539543s | best, small gain |
| `32x8x1` | V+P tile-list | 0.542812s | small gain, weaker |
| `16x8x2` | V+P tile-list | 0.578214s | regression |
| `16x4x4` | V+P tile-list | 0.567895s | regression |
| `32x4x1` | V+P tile-list | 0.545382s | neutral |
| `32x2x2` | V+P tile-list | 0.552509s | regression |
| `32x2x4` | V+P tile-list | 0.546216s | neutral/slight regression |
| `24x4x2` | V+P tile-list | 0.558097s | regression |
| `32x4x2` | P-only tile-list | 0.547992s | regression |
| `32x4x2` | V-only tile-list | 0.540711s | small gain |

## Decision

Keep the implementation as an inactive compile-time optimization path.

Do not archive as a speed milestone. The speedup is real and numerically exact, but only about `1.01x`, far below the next `1.5x` threshold.

The current remote binary was rebuilt with:

```text
-DCUDA3D_PML_TILE_LIST -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2
```

The default `make -B -f makefile.rtx5090 test` still builds the original active path.

## Failure / Limitation Mechanism

The tile-list idea reduces inactive CTA scheduling, but it does not reduce the dominant global memory traffic:

```text
p1 -> vx/vy/vz -> p0
```

The `p_pml` P-only experiment regressed, which suggests that fewer CTAs alone is not enough; the smaller block and tile indirection can cost more than the skipped core CTAs save.

The V-only experiment gained a little, which suggests the `v_pml` skip region is slightly more sensitive to tile scheduling. However, this is not a path to multi-x speedup.

## Next Proposed Step

The next meaningful direction is no longer tile scheduling. It should target PML data movement:

1. Build a one-step PML debug harness that can compare `p0`, `vx/vy/vz`, and CPML memory arrays immediately after one time step.
2. Prototype a direction-specific PML update that reduces `vx/vy/vz` global write/read round trips for a single face group, starting with one face or one direction.
3. Keep the old path as baseline and require exact or `<=1e-5` correctness before performance testing.
