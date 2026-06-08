# Pressure PML Z-Recompute Cache Prototype

## Scope

Implemented macro-default-off prototype:

```text
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

The prototype changes only `cuda_fd3d_p_pml_tile_ns`.  It builds a
CTA-local shared z-line cache for `recompute_vz_after_update_from_old_mem`
values and preserves `memory_dz_next` ownership: only tile-owned active
central z positions write next z CPML memory.

It does not enable `RECOMPUTE_X/Y/XYZ`, tile-mask fastpath, z-face
specialization, or z-face fusion.

## Validation

Test platform:

```text
/work/wenzhe/cuda3D_codex_day_20260608
RTX 5090, sm_120
baseline: zmem_reference flags
candidate combo: zmem_reference + CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
                 + CUDA3D_CPML_VMEM_DISABLE_MPI
                 + CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

Clean-worktree note:

```text
The original /work/wenzhe/cuda3D worktree had uncommitted experiment
changes, so testing used a non-destructive clean worktree under /work/wenzhe.
The perf_1gpu_6shots velocity .dir was linked from the original data
location, and a missing d_obs output directory was created.
```

Correctness:

```text
debug dump step 0/1/2       pass
correctness rel L2          0 for all 6 output files
NaN/Inf                     none detected by compare_outputs.py
perf6 repeat comparisons    pass in all 3 rounds
```

## Performance

Standalone z-cache candidate:

```text
mean WP speedup             1.044955x
mean Gradient speedup       1.045506x
verdict                     useful but below standalone >=5% gate
```

Combined candidate with Phase 1 CPML vmem scaffold:

```text
mean WP speedup             1.083390x
mean Gradient speedup       1.080857x
verdict                     pass meaningful >=5% gate; superseded by direct-fill version below
```

Direct-fill combined candidate:

```text
mean WP speedup             1.100929x
mean Gradient speedup       1.097530x
verdict                     current accepted candidate
```

Perf repeat table:

| round | baseline WP | candidate WP | WP speedup | baseline Gradient | candidate Gradient | Gradient speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.435633 | 2.249627 | 1.082683 | 2.545943 | 2.357701 | 1.079841 |
| 2 | 2.413101 | 2.227910 | 1.083123 | 2.533939 | 2.346707 | 1.079785 |
| 3 | 2.416663 | 2.228645 | 1.084364 | 2.542785 | 2.348029 | 1.082944 |

Direct-fill perf repeat table:

| round | baseline WP | candidate WP | WP speedup | baseline Gradient | candidate Gradient | Gradient speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.438928 | 2.217328 | 1.099940 | 2.549396 | 2.324237 | 1.096874 |
| 2 | 2.417782 | 2.194350 | 1.101821 | 2.535653 | 2.311585 | 1.096933 |
| 3 | 2.415093 | 2.193495 | 1.101025 | 2.541987 | 2.313455 | 1.098784 |

## Failed Aggressive Variant

Tested but removed from code:

```text
shared vx/vy pressure-neighbor cache inside the same p_pml tile
```

Result:

```text
mean WP speedup             0.419906x
mean Gradient speedup       0.426565x
```

Decision:

```text
Do not reopen pressure-PML vx/vy shared-neighbor cache in the current
block shape.  It is correctness-safe but performance-catastrophic because
the shared-memory fill/sync overhead dominates the saved global loads.
```

## Next Gate

The current accepted candidate is the combination:

```text
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_CPML_VMEM_DISABLE_MPI
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

The accepted implementation uses direct cache fill: each CTA thread fills
its own central z entry, while `threadIdx.x < 4` fills left halo entries and
`threadIdx.x < 3` fills right halo entries.  This avoids the first prototype's
linear cache-fill loop with division/modulo indexing.

Next work should profile this direct-fill combined candidate against zmem and
decompose the remaining `cuda_fd3d_p_pml_tile_ns` time.  Do not expand to
shared vx/vy caches unless new profiler evidence changes the conclusion above.

## NCU Follow-Up

Short Nsight Compute profile:

```text
profile                       docs/day_20260608/zrecomp_cache_cpml_combo_ncu_summary.md
sections                      SpeedOfLight, MemoryWorkloadAnalysis,
                              SchedulerStats, WarpStateStats, Occupancy
launch skip/count             10 / 30
kernel filter                 cuda_fd3d_(p_core|v_pml_tile|p_pml_tile)
```

Kernel duration summary:

| kernel | zmem | combo | speedup |
| --- | ---: | ---: | ---: |
| `cuda_fd3d_p_core_ns` | 76.061 us | 75.306 us | 1.010x |
| `cuda_fd3d_p_pml_tile_ns` | 158.291 us | 142.902 us | 1.108x |
| `cuda_fd3d_v_pml_tile_ns` | 58.320 us | 53.101 us | 1.098x |

Profile read:

```text
p_core remains L2/memory-throughput limited and nearly unchanged.
v_pml improves from CPML vmem ownership.
p_pml improves from z-cache, but combo p_pml still has low eligible
warps/scheduler (0.798) and high No Eligible (60.879%), so the next
pressure-PML work should target issue/latency overhead rather than raw
DRAM bandwidth.
```

## Direct-Fill NCU Follow-Up

Short Nsight Compute profile of the accepted direct-fill implementation:

```text
profile                       reports/day_20260608/directfill_combo_ncu_20260608_120449_summary.md
sections                      SpeedOfLight, MemoryWorkloadAnalysis,
                              SchedulerStats, WarpStateStats, Occupancy
launch skip/count             10 / 30
kernel filter                 cuda_fd3d_(p_core|v_pml_tile|p_pml_tile)
```

Kernel duration summary:

| kernel | zmem | direct-fill combo | speedup |
| --- | ---: | ---: | ---: |
| `cuda_fd3d_p_core_ns` | 75.942 us | 75.270 us | 1.009x |
| `cuda_fd3d_p_pml_tile_ns` | 158.438 us | 134.099 us | 1.181x |
| `cuda_fd3d_v_pml_tile_ns` | 58.794 us | 53.590 us | 1.097x |

Profile read:

```text
direct-fill improves p_pml_tile beyond the first linear-loop combo
(134.099 us vs 142.902 us in the previous short profile).  p_core remains
unchanged and should not be the next local target.  direct-fill p_pml_tile
still has high No Eligible (59.885%) and low eligible warps/scheduler
(0.820), so the remaining opportunity is control/latency in the pressure
PML z-cache path rather than shared vx/vy caching.

Next allowed micro-structure candidate:
  CUDA3D_PML_PRESSURE_ZCACHE_WARP_RANGE

Candidate intent:
  compute the active z range once per 32-thread z-line with warp broadcast,
  then reuse it for central and halo z-cache fills.  This targets repeated
  active-range branch/control work inside fill_pml_pressure_vz_cache_entry
  without changing memory_dz_next ownership or pressure math.
```

## Rejected Warp-Range Candidate

Tested:

```text
CUDA3D_PML_PRESSURE_ZCACHE_WARP_RANGE
```

Result:

```text
correctness                    pass, rel L2 = 0 for 6 outputs
perf6 repeat compares          pass
mean WP speedup vs direct-fill 0.997223x
mean Gradient speedup          0.997502x
```

Perf repeat table:

| round | direct WP | warp WP | WP speedup | direct Gradient | warp Gradient | Gradient speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.196203 | 2.207274 | 0.994984 | 2.308980 | 2.314959 | 0.997417 |
| 2 | 2.182848 | 2.187369 | 0.997933 | 2.298789 | 2.305196 | 0.997221 |
| 3 | 2.180586 | 2.183309 | 0.998753 | 2.298663 | 2.303571 | 0.997869 |

Decision:

```text
Do not keep or retry warp-broadcast active-range caching.  It is
correctness-safe, but the shuffle/control overhead is not repaid.  The
accepted source was restored to direct-fill z-cache after this test.
```

## SourceCounters Follow-Up

Direct-fill `p_pml_tile` was profiled with `-lineinfo` and SourceCounters:

```text
raw csv      reports/day_20260608/directfill_p_pml_source_ncu.csv
summary      reports/day_20260608/directfill_source_profile_summary.md
kernel       cuda_fd3d_p_pml_tile_ns
launches     skip 10 / count 10
```

Profile read:

```text
No Eligible                         ~60%
Eligible warps/scheduler            ~0.81
L1TEX scoreboard stall               ~14.4 cycles/warp
Uncoalesced global accesses          ~19% excessive sectors
Avg active threads/warp              ~19.84
Avg not-predicated-off threads/warp  ~18.69
```

Dominant source lines:

```text
mem_dzz / mem_dxx / mem_dyy CPML memory updates
final p0[outIndex] pressure writeback
```

The z-cache shared loads are visible in the source page but are not the
dominant remaining issue.

## Rejected Local-Mem Accum Candidate

Tested:

```text
pml_local_mem_accum
```

The candidate rewrote:

```text
mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
c1+=mem_dzz[pind];
```

into explicit local accumulation:

```text
const float new_mem = mem_dzz[pind]*coef+c1*(coef-1);
mem_dzz[pind]=new_mem;
c1+=new_mem;
```

Result:

```text
correctness                    pass, rel L2 = 0 for 6 outputs
perf6 repeat compares          pass
mean WP speedup vs direct-fill 1.000647x
mean Gradient speedup          0.998957x
```

Decision:

```text
Reject local new_mem accumulation.  It is numerically safe but performance
neutral, so it fails the >=2% small-candidate gate.
```

## Rejected P0 LDG Candidate

Tested:

```text
pml_p0_ldg
```

The candidate changed the old pressure read in the final pressure-PML update
from:

```text
p0[outIndex]
```

to:

```text
__ldg(p0+outIndex)
```

Result:

```text
correctness                    pass, rel L2 = 0 for 6 outputs
perf6 repeat compares          pass
mean WP speedup vs direct-fill 1.000054x
mean Gradient speedup          1.000694x
```

Perf repeat table:

| round | direct WP | candidate WP | WP speedup | direct Gradient | candidate Gradient | Gradient speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.208778 | 2.208617 | 1.000073 | 2.321201 | 2.317286 | 1.001689 |
| 2 | 2.191094 | 2.190333 | 1.000347 | 2.307412 | 2.307792 | 0.999835 |
| 3 | 2.188848 | 2.189415 | 0.999741 | 2.308949 | 2.307660 | 1.000559 |

Decision:

```text
Reject pml_p0_ldg.  The final p0 update line is hot in SourceCounters, but
changing only the old p0 operand to a read-only-cache load is noise-level
neutral and fails the >=2% small-candidate gate.
```
