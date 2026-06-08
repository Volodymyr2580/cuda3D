# Nsight Systems CUDA Summary

## Run

- WP computing time: `2.238769s`
- Gradient TIME all: `2.349826s`
- GPU kernel total: `2.232398s`
- WP minus GPU kernel total: `0.006371s`
- WP minus GPU kernel total fraction: `0.0028`
- Ideal speedup if that gap vanished: `1.002854x`
- cudaLaunchKernel CPU API total: `1.845401s`
- cudaLaunchKernel calls: `36024`
- cudaLaunchKernel avg: `51.227us`

## Kernels

| kernel | total s | instances | avg us | share |
| --- | ---: | ---: | ---: | ---: |
| `cuda_fd3d_p_pml_tile_ns` | 1.251216 | 9006 | 138.931 | 56.0% |
| `cuda_fd3d_p_core_ns` | 0.557985 | 9006 | 61.957 | 25.0% |
| `cuda_fd3d_v_pml_tile_ns` | 0.390576 | 9006 | 43.368 | 17.5% |
| `lint3d_inject_bell_extract_gpu_zz` | 0.032622 | 9006 | 3.622 | 1.5% |

## CUDA API

| API | total s | calls | avg us | share |
| --- | ---: | ---: | ---: | ---: |
| `cudaLaunchKernel` | 1.845401 | 36024 | 51.227 | 81.5% |
| `cudaMemcpy` | 0.405178 | 145 | 2794.332 | 17.9% |
| `cudaFree` | 0.005528 | 302 | 18.305 | 0.2% |
| `cudaMalloc` | 0.004109 | 230 | 17.867 | 0.2% |
| `cuKernelGetName` | 0.001908 | 36024 | 0.053 | 0.1% |
| `cudaMemset` | 0.001086 | 92 | 11.799 | 0.0% |
| `cudaMemcpyToSymbol` | 0.000960 | 72 | 13.330 | 0.0% |
| `cudaEventRecord` | 0.000069 | 12 | 5.769 | 0.0% |
