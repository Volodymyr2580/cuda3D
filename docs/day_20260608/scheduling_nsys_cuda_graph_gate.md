# Scheduling Nsight Systems CUDA Graph Gate

## Context

This gate checks whether the current best single-GPU wave-step loop has enough visible launch/scheduling gap to justify a CUDA Graph or launch-aggregation prototype.

Current best candidate:

```text
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_CPML_VMEM_DISABLE_MPI
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
direct-fill pressure z-cache implementation
```

Run location:

```text
/work/wenzhe/cuda3D_codex_day_20260608_68de1a7
```

Case:

```text
benchmarks/cases/perf_1gpu_6shots
```

Nsight Systems command class:

```text
nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none
```

Binary SHA256:

```text
bf719d04f0fa1136af3f1afac54a936ee0d052a18ffd9a9d07863aa7f9dfca28
```

## Program Timing

```text
Gradient TIME all = 2.349826s
WP computing time = 2.238769s
ALL DONE
```

## CUDA Kernel Summary

| Kernel | Total s | Instances | Avg us | Share |
| --- | ---: | ---: | ---: | ---: |
| `cuda_fd3d_p_pml_tile_ns` | 1.251216 | 9006 | 138.931 | 56.0% |
| `cuda_fd3d_p_core_ns` | 0.557985 | 9006 | 61.957 | 25.0% |
| `cuda_fd3d_v_pml_tile_ns` | 0.390576 | 9006 | 43.368 | 17.5% |
| `lint3d_inject_bell_extract_gpu_zz` | 0.032622 | 9006 | 3.622 | 1.5% |

GPU kernel total:

```text
2.232398465s
```

## CUDA API Summary

| API | Total s | Calls | Avg us | Share |
| --- | ---: | ---: | ---: | ---: |
| `cudaLaunchKernel` | 1.845401 | 36024 | 51.227 | 81.5% |
| `cudaMemcpy` | 0.405178 | 145 | 2794.332 | 17.9% |
| `cudaFree` | 0.005528 | 302 | 18.305 | 0.2% |
| `cudaMalloc` | 0.004109 | 230 | 17.867 | 0.2% |

## Launch Gap Bound

Although `cudaLaunchKernel` accumulates `1.845401s` of CPU API time, that time is mostly overlapped with GPU execution.

The visible WP gap relative to GPU kernel time is:

```text
WP computing time - GPU kernel total
= 2.238769s - 2.232398465s
= 0.006370535s
```

Gap fraction:

```text
0.006370535 / 2.238769 = 0.002846
```

Ideal speedup if this entire non-kernel gap vanished:

```text
2.238769 / 2.232398465 = 1.002854x
```

## Decision

Reject a CUDA Graph / launch aggregation CUDA prototype for the current single-GPU `perf_1gpu_6shots` loop.

Reason:

```text
The current best binary is not visibly launch-gap limited in WP time.
Even a perfect removal of the measured non-kernel gap gives only about
1.002854x ideal WP speedup, far below the >=2% small-candidate gate.
```

## Reopen Conditions

This decision only covers the current single-GPU/single-MPI-rank loop. Reopen scheduling work only if at least one condition is met:

```text
1. Nsight Systems shows >2% GPU idle / WP-kernel gap on the same perf case.
2. Multi-rank or multi-shot wall-clock profiling shows launch/scheduling as a real wall-clock bottleneck.
3. The wave-step architecture changes enough that launch count or CPU orchestration becomes newly dominant.
```

Until then, CUDA Graph work should not consume implementation time.

## Artifacts

```text
reports/day_20260608/directfill_scheduling_cuda_api_sum.csv
reports/day_20260608/directfill_scheduling_cuda_gpu_kern_sum.csv
reports/day_20260608/directfill_scheduling_nsys_run.log
reports/day_20260608/directfill_scheduling_nsys_bin.sha256
reports/day_20260608/directfill_scheduling_nsys_summary.md
reports/day_20260608/directfill_scheduling_nsys_summary.json
```
