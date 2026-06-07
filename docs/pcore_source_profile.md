# p_core Source-Level NCU Profile

Date: 2026-06-07

Branch: `exp/core-zpencil-shared`

## Purpose

This profile tests the premise for a `CUDA3D_CORE_ZPENCIL_SHARED` prototype against the stable RTX 5090 baseline:

```text
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DCUDA3D_PML_ZMEM_IN_P
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
```

The key question was whether `p_core` still repeatedly reloads z-neighbor values from global memory. If yes, a z-pencil shared-memory rewrite could be justified. If no, the proposed first prototype would duplicate the current kernel and should be stopped.

## Commands

The source profile was collected on `/work/wenzhe/cuda3D` with `-lineinfo`:

```bash
cd /work/wenzhe/cuda3D
source ./env_5090.sh

LINEINFO_FLAGS='-O3 -arch=sm_120 --use_fast_math -lineinfo \
-DCUDA3D_PML_RECOMPUTE_Z \
-DCUDA3D_PML_TILE_LIST \
-DCUDA3D_PML_ZMEM_IN_P \
-DPmlTileBlockSize1=32 \
-DPmlTileBlockSize2=4 \
-DPmlTileBlockSize3=2'

(cd src && make -B -f makefile.rtx5090 test NVFLAGS="$LINEINFO_FLAGS")

ncu --target-processes all \
  --force-overwrite \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section Occupancy \
  --section SourceCounters \
  --section WarpStateStats \
  --kernel-name regex:".*p_core.*" \
  --launch-skip 10 \
  --launch-count 20 \
  -o benchmarks/profiles/pcore_zmem_lineinfo_20260607 \
  bash -lc 'cd benchmarks/cases/profile_1gpu && CUDA_VISIBLE_DEVICES=0 /opt/intel/oneapi/mpi/latest/bin/mpirun -np 1 ../../../bin/cuda_3D_FM < input_profile_1gpu.in'

ncu --import benchmarks/profiles/pcore_zmem_lineinfo_20260607.ncu-rep \
  --page source --print-source cuda,sass --csv \
  > benchmarks/profiles/pcore_zmem_lineinfo_20260607_source_cuda_sass.csv
```

The profiling binary SHA256 was:

```text
019e615560c090aa849ac84a9075cc5baff017cfe801eb85af917f1f9ea896ce  bin/cuda_3D_FM
```

After profiling, the server binary was restored to the non-`-lineinfo` zmem baseline:

```text
cfc502cf8a27038e54a1bdf1c3665b057a3b306046bd547a1baf70a204e17293  bin/cuda_3D_FM
```

Post-restore sanity run:

```text
case: benchmarks/cases/profile_1gpu
Gradient TIME all = 0.189999 s
WP computing time = 0.161748 s
ALL DONE
```

## Kernel-Level Summary

`p_core` was sampled over 20 launches.

| Metric | Value |
|---|---:|
| Block size | `(128, 2, 1)` |
| Grid size | `(1, 117, 233)` |
| Duration avg | `93.630 us` |
| DRAM throughput avg | `42.40%` |
| L1/TEX throughput avg | `60.14%` |
| L2 throughput avg | `96.83%` |
| SM throughput avg | `58.42%` |
| Theoretical occupancy | `83.33%` |
| Achieved occupancy avg | `66.51%` |
| Active warps/scheduler | `7.93` |
| Eligible warps/scheduler | `1.20` |
| Registers/thread | `48` |
| Shared memory/block | about `2.16 KB` |

The kernel is not register-spilling. It is mainly latency-limited by memory dependency chains, with L2 already close to saturation for this profile case.

## Existing p_core z-Pencil

The current `cuda_fd3d_p_core_ns` already has a z/fast-dimension shared-memory pencil:

```cpp
enum { CoreStencilRadius = 7 };
__shared__ float z_tile[PBlockSize3][PBlockSize2][PBlockSize1 + 2 * CoreStencilRadius];
```

It loads:

- center point `p1[base]` into `z_tile[...][local1]`;
- left/right halo points for `CoreStencilRadius == 7`;
- all z-neighbor stencil terms from `z_tile`, not from global memory.

The launch geometry is also tied to `PBlockSize*`:

```cpp
dimg_p.x = (core_nz + PBlockSize1 - 1) / PBlockSize1;
dimg_p.y = (core_nx + PBlockSize2 - 1) / PBlockSize2;
dimg_p.z = (core_ny + PBlockSize3 - 1) / PBlockSize3;
dimb_p.x = PBlockSize1;
dimb_p.y = PBlockSize2;
dimb_p.z = PBlockSize3;
```

`PCoreBlockSize*` exists in `include/inc3D/cu_common.h`, but is not used by the current `p_core` launch path.

## Source Hotspots

The following source counters are sums over 20 profiled launches.

| Line | Source | Samples | Long SB | Barrier | Global L1 Tags | L2 Excessive Sectors | Access |
|---:|---|---:|---:|---:|---:|---:|---|
| 1086 | `z_tile[...] = p1[base];` | 31,816 | 25,483 | 0 | 7,735,180 | 3,680,260 | global load + shared store |
| 1101 | core-boundary `return` line after sync | 29,549 | 0 | 28,444 | 0 | 0 | barrier attribution |
| 1116 | `y2 * (p1[base +/- 2 * stride3])` | 20,182 | 17,627 | 0 | 12,214,680 | 5,428,920 | global load |
| 1111 | `x2 * (p1[base +/- stride2])` | 10,152 | 8,713 | 0 | 12,214,660 | 5,428,920 | global load |
| 1092 | left halo shared assignment | 6,567 | 6,432 | 0 | 0 | 0 | shared store |
| 1115 | `x2 * (p1[base +/- 2 * stride2])` | 6,046 | 4,883 | 0 | 12,214,680 | 5,428,920 | global load |
| 1112 | `y2 * (p1[base +/- stride3])` | 4,704 | 3,181 | 0 | 12,214,860 | 5,428,920 | global load |
| 1138 | final pressure write/update line | 1,434 | 0 | 0 | 18,321,900 | 8,143,380 | global read/write |

Grouped by stencil direction:

| Group | Samples | Long SB | Global L1 Tags | L2 Excessive Sectors | Shared Wavefronts |
|---|---:|---:|---:|---:|---:|
| z shared path, including center/halo load | 43,604 | 32,644 | 7,735,180 | 3,680,260 | 55,402,740 |
| x global neighbor loads | 23,855 | 14,722 | 85,504,620 | 38,002,280 | 0 |
| y global neighbor loads | 31,118 | 21,153 | 85,505,160 | 38,002,280 | 0 |
| x + y global neighbor loads | 54,973 | 35,875 | 171,009,780 | 76,004,560 | 0 |

## Interpretation

The source profile does not confirm z-neighbor global reload as the main issue.

The z-neighbor stencil lines use `Address Space = Shared(2)` and have no global L1 tag requests or L2 global sectors. That means the proposed first-version z-pencil cache is already present in the baseline kernel.

The dominant remaining global-load pressure is in x/y neighbor loads. The largest single long-scoreboard line is the center global load into shared memory, but that load is the fill step for the existing shared tile, not a missed z-neighbor reuse opportunity.

The barrier stall is real and comes from the existing shared-memory path. NCU attributes the barrier samples mostly to the first core-boundary return line after:

```cpp
__syncthreads();
if (gtid1 < core1_lo || ... ) return;
```

This makes adding another ordinary shared-memory z-pencil layer unattractive: it would add or preserve the same synchronization cost while caching values already served from shared memory.

## Decision

Do not implement `CUDA3D_CORE_ZPENCIL_SHARED` as originally scoped.

Reason:

- The baseline already contains the proposed z-pencil shared-memory cache.
- Source counters show z-neighbor terms are shared-memory loads.
- The remaining hot global traffic is x/y neighbor traffic and final pressure update traffic.
- A macro-gated duplicate would not be a real optimization candidate and would not have a credible path to the required `p_core >= 10%` kernel speedup.

The appropriate fallback is to analyze temporal/dataflow blocking feasibility instead of continuing p_core block-shape or duplicate shared-memory variants.

