# Codex Report - CUDA3D_PML_ZMEM_IN_P

## Summary

??? Pro ???????????? `CUDA3D_PML_ZMEM_IN_P`?

?????

- ??????debug dump step 0/1/2 ? current best ?????correctness/perf ????? `rel_l2 = 0`?
- ?????????`perf_1gpu_6shots` ?? repeat ? WP speedup ??? `1.0349x` ? `1.0329x`?
- ????? `3%~7%` ???????????????????????????
- ??????? `bin/cuda_3D_FM` ???? current best ????? `CUDA3D_PML_ZMEM_IN_P`?

## Implementation

?????

- `include/inc3D/single_solver.h`
- `src/single_solver.cu`
- `src/rem_fd.cu`

?????

- ?? `CUDA3D_PML_ZMEM_IN_P` ????
- ?? `d_memory_dz_next`???? `d_memory_dz` ???
- ?? timestep ? `p_pml` ??? swap `d_memory_dz` / `d_memory_dz_next`?
- `CUDA3D_PML_ZMEM_IN_P` ????`v_pml` / `v_pml_tile` ???? z velocity derivative?`vz` ??? `mem_dz` ???
- `p_pml` / `p_pml_tile` ?? `recompute_vz_after_update_from_old_mem`?? `mem_dz_old` ???? owned z-PML point ? `mem_dz_new`?
- ?? `CUDA3D_PML_ZMEM_DEBUG_FILL`?debug ????? `mem_dz_next` ? NaN?kernel ????? z memory entry ?????
- ????????`CUDA3D_PML_ZMEM_IN_P` ??? `CUDA3D_PML_RECOMPUTE_Z` ?????

## Build Matrix

| Build | Macros | Result | Log |
|---|---|---:|---|
| default | makefile default | pass | `benchmarks/build_logs/default_after_zmem_patch_20260607_001658.log` |
| current best | `RECOMPUTE_Z + TILE_LIST + 32x4x2` | pass | `benchmarks/build_logs/current_best_after_zmem_patch_20260607_001658.log` |
| debug baseline | current best + `DEBUG_DUMP + DEBUG_CHECKS` | pass | `benchmarks/build_logs/current_best_debug_zmem_compare_20260607_001849.log` |
| debug candidate | current best + `ZMEM_IN_P + ZMEM_DEBUG_FILL + DEBUG_DUMP + DEBUG_CHECKS` | pass | `benchmarks/build_logs/zmem_in_p_debug_20260607_001849.log` |
| release candidate | current best + `ZMEM_IN_P` | pass | `benchmarks/build_logs/zmem_in_p_release_20260607_002024.log` |

## Debug Dump Validation

Debug dump baseline: current best.
Debug dump candidate: `CUDA3D_PML_ZMEM_IN_P + CUDA3D_PML_ZMEM_DEBUG_FILL`.

| Step | Report | Result |
|---:|---|---:|
| 0 | `benchmarks/reports/debug_dump_zmem_in_p_vs_best_it0_20260607_001849/comparison.md` | pass |
| 1 | `benchmarks/reports/debug_dump_zmem_in_p_vs_best_it1_20260607_001849/comparison.md` | pass |
| 2 | `benchmarks/reports/debug_dump_zmem_in_p_vs_best_it2_20260607_001849/comparison.md` | pass |

NaN fill ?????????? `mem_dz_new` ? z-PML entries ? smoke case ??? owned thread ???

## Release Correctness

?????? `fixed` ????? timestamp ?? `fixed` ??? compare report ? shell ??????? stdout???????????????

| Case | Report | Result |
|---|---|---:|
| correctness | `benchmarks/reports/zmem_in_p_correctness_vs_best_fixed_20260607_002024/comparison.md` | pass, all rel_l2 = 0 |
| perf_1gpu | `benchmarks/reports/zmem_in_p_perf1_vs_best_fixed_20260607_002024/comparison.md` | pass, all rel_l2 = 0 |
| perf_1gpu_6shots | `benchmarks/reports/zmem_in_p_perf6_vs_best_fixed_20260607_002024/comparison.md` | pass, all rel_l2 = 0 |
| perf_1gpu_6shots repeat | `benchmarks/reports/zmem_in_p_perf6_repeat_vs_best_20260607_002416/comparison.md` | pass, all rel_l2 = 0 |

## Performance

First release pair:

| Case | Version | WP computing time | Gradient TIME all | Wall time |
|---|---|---:|---:|---:|
| correctness | current best | 0.012936s | 0.014592s | 0:02.28 |
| correctness | ZMEM_IN_P | 0.012964s | 0.014649s | 0:02.02 |
| perf_1gpu | current best | 0.508251s | 0.537898s | 0:02.85 |
| perf_1gpu | ZMEM_IN_P | 0.481991s | 0.511625s | 0:02.79 |
| perf_1gpu_6shots | current best | 2.506582s | 2.633927s | 0:04.84 |
| perf_1gpu_6shots | ZMEM_IN_P | 2.421955s | 2.539333s | 0:04.84 |

Speedup:

| Case | WP speedup | Gradient speedup |
|---|---:|---:|
| perf_1gpu | 1.0545x | 1.0514x |
| perf_1gpu_6shots | 1.0349x | 1.0373x |

Repeat `perf_1gpu_6shots`:

| Version | WP computing time | Gradient TIME all | Wall time |
|---|---:|---:|---:|
| current best repeat | 2.499682s | 2.623294s | 0:04.84 |
| ZMEM_IN_P repeat | 2.420177s | 2.531806s | 0:04.83 |

Repeat speedup:

| Metric | Speedup |
|---|---:|
| WP computing time | 1.0329x |
| Gradient TIME all | 1.0361x |

## Profiling Status

Nsight Compute performance counter is still restricted:

```text
RmProfilingAdminOnly: 1
```

Fallback artifacts:

- `benchmarks/profiles/nvidia_params_profiling_20260607_002231.txt`
- `benchmarks/profiles/current_best_ptxas_resource_20260607_002231.log`
- `benchmarks/profiles/current_best_cuobjdump_resource_20260607_002231.txt`

Current best resource snippets from fallback:

| Kernel | Registers | Shared | Spill |
|---|---:|---:|---:|
| `cuda_fd3d_p_pml_tile_ns` | 48 | 0 | 0 |
| `cuda_fd3d_v_pml_tile_ns` | 38 | 0 | 0 |
| `cuda_fd3d_p_core_ns` | 48 | 2160 bytes by cuobjdump | 0 |

## Final State

Final binary restored to current best build:

```text
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
```

Final smoke:

- run: `benchmarks/runs/smoke_1gpu_current_best_final_after_zmem_repeat_20260607_002504`
- result: `ALL DONE`, outputs = 3
- `WP computing time = 0.002406s`
- `Gradient TIME all = 0.003197s`
- final binary sha256: `0b921158eb9b05ffde7f1688b2e641a6371acc8ad485ac7984092f5022166565`

## Decision

`CUDA3D_PML_ZMEM_IN_P` ???????????????????? `3.3%~3.7%` on `perf_1gpu_6shots`?wall-clock ????????

- ??????????
- ????? current best?
- ?? `archives/speedups` ???????
- ??????????????????????? fused/shared-halo ??????? p_pml ? v_pml ???/?????

??????????? z-only ????????????PML fused shared-halo prototype ? core temporal blocking prototype?
