# Profiler Inventory 2026-06-07

## Summary

Nsight Compute hardware counters are now available on the RTX 5090 server after enabling the NVIDIA profiling module option and rebooting.

Current gate status:

```text
RmProfilingAdminOnly: 0
NCU hardware counters available: yes
PML fusion allowed to start: yes
p_core z-pencil allowed to start: conditionally yes, after/if PML z-slab result is insufficient or if p_core becomes the dominant bottleneck
```

## Permission Fix

The server originally reported:

```text
RmProfilingAdminOnly: 1
ERR_NVGPUCTRPERM
```

The following persistent configuration was written with administrator permission:

```text
/etc/modprobe.d/nvidia-profiler.conf
options nvidia NVreg_RestrictProfilingToAdminUsers=0
```

Then `update-initramfs -u` and a server reboot were performed. After reboot:

```text
RmProfilingAdminOnly: 0
```

## Environment

```text
Server path: /work/wenzhe/cuda3D
GPU: NVIDIA GeForce RTX 5090
Driver: 595.71.05
Nsight Compute: /usr/local/cuda-13.0/bin/ncu
NCU version: 2025.3.0.0
NVCC: CUDA 13.0
Git head: 1034ddb before profiler run
```

## NCU Commands

Main throughput/resource inventory:

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
  -o benchmarks/profiles/zmem_ncu_main_20260607 \
  bash -lc 'cd benchmarks/cases/profile_1gpu && CUDA_VISIBLE_DEVICES=0 /opt/intel/oneapi/mpi/latest/bin/mpirun -np 1 ../../../bin/cuda_3D_FM < input_profile_1gpu.in'
```

Warp-state inventory:

```bash
ncu --target-processes all \
  --force-overwrite \
  --section WarpStateStats \
  --kernel-name regex:".*(p_pml_tile|v_pml_tile|p_core).*" \
  --launch-skip 10 \
  --launch-count 30 \
  -o benchmarks/profiles/zmem_ncu_warpstates_20260607 \
  bash -lc 'cd benchmarks/cases/profile_1gpu && CUDA_VISIBLE_DEVICES=0 /opt/intel/oneapi/mpi/latest/bin/mpirun -np 1 ../../../bin/cuda_3D_FM < input_profile_1gpu.in'
```

Both NCU runs completed with return code `0` and the application reached `ALL DONE`.

## Kernel Throughput And Occupancy

| Kernel | Samples | Block | Grid | Time avg | Compute/mem throughput % | DRAM throughput % | SM throughput % | Warps active % | Eligible warps/cycle | Reg/thread | Shared/block | Waves/SM |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `p_pml_tile` | 10 | `(32, 4, 2)` | `(22188, 1, 1)` | 189.366 | 56.15 | 40.43 | 56.15 | 72.82 | 1.143 | 44 | 1.024 KB | 26.10 |
| `v_pml_tile` | 10 | `(32, 4, 2)` | `(23100, 1, 1)` | 71.299 | 55.40 | 45.74 | 42.33 | 82.35 | 1.129 | 38 | 1.024 KB | 22.65 |
| `p_core` | 10 | `(128, 2, 1)` | `(1, 117, 233)` | 93.555 | 96.94 | 42.44 | 58.16 | 66.33 | 1.173 | 48 | 2.160 KB | 32.07 |

`Time avg` is the NCU raw `gpu__time_duration.avg` value for each sampled launch; use it for within-report comparison, not whole-application timing.

## Top Warp Stalls

| Kernel | Stall 1 | Stall 2 | Stall 3 | Stall 4 | Stall 5 | Issue active/cycle |
|---|---:|---:|---:|---:|---:|---:|
| `p_pml_tile` | long_scoreboard 8.70 | wait 1.97 | short_scoreboard 1.91 | not_selected 1.17 | branch_resolving 0.63 | 0.528 |
| `v_pml_tile` | long_scoreboard 15.63 | wait 1.56 | not_selected 1.49 | math_pipe_throttle 0.69 | short_scoreboard 0.49 | 0.457 |
| `p_core` | long_scoreboard 8.64 | barrier 3.49 | not_selected 1.67 | wait 0.90 | mio_throttle 0.84 | 0.444 |

## Static Resource Fallback

The earlier fallback table remains useful for sanity checking:

| Kernel | Registers | Shared Memory | Stack | Local | Spill |
|---|---:|---:|---:|---:|---:|
| `cuda_fd3d_p_pml_tile_ns` | 44 | 0 B static / 1.024 KB launch allocation | 0 B | 0 B | 0 B |
| `cuda_fd3d_v_pml_tile_ns` | 38 | 0 B static / 1.024 KB launch allocation | 0 B | 0 B | 0 B |
| `cuda_fd3d_p_core_ns` | 48 | 2160 B cuobjdump / 1136 B ptxas | 0 B | 0 B | 0 B |

## Architecture Read

- `p_pml_tile` is the largest sampled launch among the three target kernels and shows long-scoreboard, wait, and short-scoreboard pressure. This supports a dataflow/reuse prototype rather than more block-size tuning.
- `v_pml_tile` has the strongest long-scoreboard signal. A fused z-slab prototype is plausible if it reduces global velocity/memory round trips and dependency latency.
- `p_core` has very high compute/memory throughput (`~96.94%`) but only moderate DRAM throughput (`~42.44%`) and long-scoreboard stalls. A controlled z-pencil/shared-memory prototype is justified, but should follow PML z-slab unless PML fails or p_core becomes the dominant limiter.
- Register pressure is not the lead bottleneck: target kernels have no spill, and prior `-maxrregcount` sweeps were slower.

## Artifacts

Tracked summary:

```text
docs/profiler_inventory.md
```

Raw artifacts, ignored by Git but kept on the server:

```text
benchmarks/profiles/zmem_ncu_main_20260607.ncu-rep
benchmarks/profiles/zmem_ncu_main_20260607.log
benchmarks/profiles/zmem_ncu_main_20260607_raw.csv
benchmarks/profiles/zmem_ncu_main_20260607_key_metrics.json
benchmarks/profiles/zmem_ncu_warpstates_20260607.ncu-rep
benchmarks/profiles/zmem_ncu_warpstates_20260607.log
benchmarks/profiles/zmem_ncu_warpstates_20260607_raw.csv
benchmarks/profiles/zmem_ncu_warpstates_20260607_key_metrics.json
```
