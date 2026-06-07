# CUDA3D Architecture Report 2026-06-07

## Decision

`CUDA3D_PML_ZMEM_IN_P` is now the stable engineering baseline for the RTX 5090 server.

Do not compare new experiments against `current_best_reference` as the main baseline. Compare against `zmem_reference`.

Stable build:

```bash
cd /work/wenzhe/cuda3D
source ./env_5090.sh
cd src
make -B -f makefile.rtx5090 test NVFLAGS="-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2"
```

## Evidence

From the overnight RTX 5090 report:

| Metric | current_best_reference | zmem_reference | Speedup |
|---|---:|---:|---:|
| `perf6 WP` | 2.507059 s | 2.393577 s | 1.047411x |
| `perf6 Gradient` | 2.632964 s | 2.514862 s | 1.046962x |
| `perf6_repeat WP` | 2.508503 s | 2.390644 s | 1.049300x |
| `perf6_repeat Gradient` | 2.632298 s | 2.514458 s | 1.046865x |

Rejected routes:

- `CUDA3D_PML_ZMEM_V_TILE_PRUNE`: correct, but pruned 0 tiles and did not improve performance.
- `CUDA3D_PML_TILE_MASK_FASTPATH`: correct, but slower.
- PML tile block sweep: no stable gain over zmem.
- `p_core` simple block sweep: best repeat only about 1.0018x over zmem.
- `-maxrregcount`: consistently slower.
- zface split routes: previously correct in some cases but slower, not worth extending.

## Architecture Gate

The next phase is not another macro/block sweep. It is a profiler-guided data-flow rewrite.

Required before large CUDA rewrites:

```text
Nsight Compute counters for p_pml_tile, v_pml_tile, and p_core.
```

Current status:

```text
NCU installed: yes
NCU counter permission: no
Reason: RmProfilingAdminOnly: 1 / ERR_NVGPUCTRPERM
```

## Approved Prototype Order

1. `CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE`
   - Only after NCU data is available.
   - Only pure z-PML face with x/y in core region.
   - Residual regions stay on zmem generic path.
   - Stop if `perf_1gpu_6shots repeat < 5%` versus zmem.

2. `CUDA3D_CORE_ZPENCIL_SHARED`
   - Only if NCU shows `p_core` is memory-bound or z-neighbor reload dominated.
   - Stop if `p_core` kernel speedup is `<10%` or whole-job repeat speedup is `<2%`.

## Current Outcome

No new structural CUDA code was started in this pass because the profiler gate failed. This is intentional: the current evidence says more blind micro-tuning is low-value.

Next action: enable NVIDIA performance counters, rerun NCU, then decide whether PML z-slab fusion or p_core z-pencil has the stronger evidence.
