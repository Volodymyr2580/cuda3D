# Len16 Half-Warp NCU Profile

## Decision

The NCU profile confirms that `CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK`
improves the pressure-PML kernel path by changing active-segment ownership.
The next CUDA prototype should not be a simple length-23 special case.  It
needs a fresh design gate for either exact active-point/descriptor ownership or
a source-level drill-down inside the packed len16 kernel.

## Run

Remote isolated worktree:

```text
/work/wenzhe/cuda3D/.codex_worktrees/sprint_0648
```

Profile case:

```text
benchmarks/cases/profile_1gpu
```

Nsight Compute command shape:

```bash
ncu --target-processes all --csv --page details \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  --section WarpStateStats \
  --section Occupancy \
  --section SourceCounters \
  --launch-skip 10 \
  --launch-count 12 \
  --kernel-name 'regex:.*cuda_fd3d_(p_core|v_pml_tile|p_pml_tile|p_pml_len16_halfwarp).*'
```

Both profile binaries were built with `-lineinfo` for source/counter
visibility.  After profiling, the remote binary was rebuilt back to release
len16 candidate flags without `-lineinfo` and smoke-tested.

Artifacts:

```text
reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.md
reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json
reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/directfill_profile_ncu_details.csv
reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/len16_profile_ncu_details.csv
```

Final release binary:

```text
SHA256 2dd6c588c41f206adcb0121a755e17857ef1a862fc28d59d72c7434e64685b3a
```

Post-profile smoke:

```text
benchmarks/runs/smoke_1gpu_len16_after_ncu_restore_20260608_160633
ALL DONE
```

## Main Metrics

| kernel path | direct-fill duration | len16 duration | note |
| --- | ---: | ---: | --- |
| `cuda_fd3d_p_core_ns` | `93.752us` | `93.547us` | unchanged |
| `cuda_fd3d_v_pml_tile_ns` | `65.528us` | `65.248us` | unchanged |
| pressure PML total | `164.328us` | `138.453us` | `1.187x` kernel-path speedup |
| pressure residual tile | `164.328us` | `72.683us` | residual after len16 split |
| pressure len16 packed | absent | `65.771us` | new packed kernel |

Sampled main-kernel total:

```text
direct-fill: 323.608us
len16:       297.248us
speedup:     1.0887x
```

This matches the independent `perf_1gpu_6shots` repeat result direction:

```text
mean WP speedup vs direct-fill:       1.082719x
mean Gradient speedup vs direct-fill: 1.072448x
```

## Bottleneck Movement

The direct-fill pressure-PML kernel before len16:

```text
duration:                         164.328us
No Eligible:                      61.170%
Eligible warps/scheduler:          0.775
Avg active threads/warp:          19.680
Avg not-predicated threads/warp:  18.550
Branch efficiency:                75.530%
```

After len16, the residual pressure-PML kernel:

```text
duration:                          72.683us
No Eligible:                       63.497%
Eligible warps/scheduler:           0.733
Avg active threads/warp:           22.950
Avg not-predicated threads/warp:   21.650
Branch efficiency:                 83.320%
```

The new packed len16 kernel:

```text
duration:                          65.771us
No Eligible:                       73.827%
Eligible warps/scheduler:           0.433
Warp cycles/issued instruction:    34.210
Avg active threads/warp:           26.380
Avg not-predicated threads/warp:   24.910
Branch efficiency:                 65.220%
Block limit registers:             10
Block limit shared memory:         18
```

Interpretation:

- The split substantially reduces total pressure-PML duration.
- Removing len16 tiles improves residual pressure-PML branch efficiency and
  active-thread shape.
- The packed len16 kernel is now a latency/issue-limited kernel with very low
  eligible warps per scheduler, not a simple occupancy problem.
- `p_core` is unchanged and still memory-throughput limited.
- `v_pml` is unchanged and remains a separate coalescing/memory-latency problem.

## Length-23 Gate

The earlier active-segment model measured:

```text
length-23 active line slots: 87,776
length-23 active lanes:      2,018,848
inactive lanes in those warps: 789,984
```

After len16 packing, a length-23-only special case could at best remove about
`0.790M` inactive lanes from the pressure-PML path.  That is much smaller than
the length-16 opportunity already taken, and it cannot pack two lines into one
warp.  A naive length-23 kernel would still use one warp per line with only 23
active lanes, while adding another launch, another tile list, and more host-side
split logic.

Decision:

```text
Do not open a simple CUDA3D_PML_PRESSURE_LEN23_* prototype.
```

Allowed reopening condition:

```text
Only reopen length-23 work as part of a broader exact active-point or compact
descriptor design that demonstrates >=5% repeat speedup ceiling before code.
```

## Next Gate

Recommended next gate:

```text
Phase 4.11 exact active-point / compact descriptor budget
```

Questions it must answer before CUDA code:

- Can active z-points across length-16, length-23, and edge/corner residual
  regions be packed without descriptor traffic overwhelming lane savings?
- Can the descriptor be generated once per shot/domain rather than per time
  step?
- Can the kernel preserve direct-fill z-cache math and CPML memory ownership?
- Does the model predict at least `>=5%` `perf_1gpu_6shots` repeat speedup?

If that gate fails, the next more promising route is not another pressure-PML
micro-specialization.  It should shift to either:

- source-level drill-down of `cuda_fd3d_p_pml_len16_halfwarp_ns`, especially
  branch/control and final `p0/mem_dzz` update latency, or
- a separate v-PML memory layout/coalescing design.
