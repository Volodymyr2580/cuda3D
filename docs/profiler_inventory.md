# Profiler Inventory 2026-06-07

## Summary

Nsight Compute is installed on the RTX 5090 server, but GPU performance counters are currently admin-only. The attempted NCU run connected to the target process and the application completed, but NCU failed to collect hardware counters with `ERR_NVGPUCTRPERM`.

Decision: do not start PML fusion or p_core z-pencil implementation until counters are enabled, unless the user explicitly overrides this architecture gate.

## Environment Probe

Server path:

```text
/work/wenzhe/cuda3D
```

Probe result:

```text
GPU: NVIDIA GeForce RTX 5090, driver 595.71.05, 32607 MiB
Nsight Compute: /usr/local/cuda-13.0/bin/ncu
NCU version: 2025.3.0.0
NVCC: /usr/local/cuda-13.0/bin/nvcc, CUDA 13.0
Profiling parameter: RmProfilingAdminOnly: 1
```

NCU attempt result:

```text
return code: 1
error: ERR_NVGPUCTRPERM
application status: ALL DONE
attempt log: benchmarks/profiles/zmem_ncu_main_attempt_20260607.log
```

## Admin Request

Please enable NVIDIA GPU performance counters for non-admin users on the RTX 5090 server, or run the NCU profile under an account with sufficient permissions.

Diagnostic command:

```bash
cat /proc/driver/nvidia/params | grep -E "RmProfilingAdminOnly|Profiling"
```

Required outcome:

```text
RmProfilingAdminOnly: 0
```

After enabling counters, run:

```bash
cd /work/wenzhe/cuda3D
source ./env_5090.sh
ncu --target-processes all \
  --force-overwrite \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section Occupancy \
  --section LaunchStats \
  --kernel-name regex:".*(p_pml_tile|v_pml_tile|p_core).*" \
  --launch-skip 10 \
  --launch-count 30 \
  -o benchmarks/profiles/zmem_ncu_main \
  bash -lc 'cd benchmarks/cases/profile_1gpu && CUDA_VISIBLE_DEVICES=0 /opt/intel/oneapi/mpi/latest/bin/mpirun -np 1 ../../../bin/cuda_3D_FM < input_profile_1gpu.in'
```

## Fallback Static Resource Table

Because NCU counters are unavailable, a fallback `ptxas -v` and `cuobjdump --dump-resource-usage` inventory was generated. This is not a substitute for memory throughput, occupancy, or stall reasons; it only tells us static resource pressure.

| Kernel | Registers | Shared Memory | Stack | Local | Spill |
|---|---:|---:|---:|---:|---:|
| `cuda_fd3d_p_pml_tile_ns` | 44 | 0 B | 0 B | 0 B | 0 B |
| `cuda_fd3d_v_pml_tile_ns` | 38 | 0 B | 0 B | 0 B | 0 B |
| `cuda_fd3d_p_core_ns` | 48 | 2160 B cuobjdump / 1136 B ptxas | 0 B | 0 B | 0 B |
| `cuda_fd3d_p_pml_ns` | 48 | 0 B | 0 B | 0 B | 0 B |
| `cuda_fd3d_v_pml_ns` | 38 | 0 B | 0 B | 0 B | 0 B |

Static interpretation:

- The target tile kernels show no spill, so register cap sweeping is unlikely to help and was already empirically slower.
- `p_pml_tile` and `v_pml_tile` use no shared memory in the current tile-list implementation, which leaves room for a carefully scoped shared-halo/fusion prototype, but not enough evidence to start it without NCU.
- `p_core` already uses a small shared z-tile and 48 registers; a z-pencil rewrite should wait for NCU evidence that global memory or z-neighbor reloads dominate.

Raw fallback artifacts are on the server:

```text
benchmarks/profiles/ncu_permission_probe_20260607.txt
benchmarks/profiles/zmem_ncu_main_attempt_20260607.log
benchmarks/profiles/zmem_ptxas_verbose_20260607.log
benchmarks/profiles/zmem_cuobjdump_resource_20260607.txt
benchmarks/profiles/zmem_static_resource_inventory_20260607.md
```

## Gate Status

```text
NCU hardware counters available: no
PML fusion allowed to start: no
p_core z-pencil allowed to start: no
Next action: ask admin to enable counters, then rerun NCU command above.
```
