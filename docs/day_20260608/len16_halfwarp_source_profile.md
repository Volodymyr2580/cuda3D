# Len16 Half-Warp Source Profile

## Decision

Do not open a len16-only syntax micro-optimization prototype for `p0` read
syntax, local `new_mem` accumulation, or branch-only specialization.

The source-level Nsight Compute profile shows that the accepted len16 packed
kernel is dominated by final pressure writeback and z-CPML memory update
dependency stalls.  This is not a z-cache fill problem anymore.

## Run

Remote isolated worktree:

```text
/work/wenzhe/cuda3D/.codex_worktrees/sprint_0648
```

Kernel:

```text
cuda_fd3d_p_pml_len16_halfwarp_ns
```

Profile command shape:

```bash
ncu --target-processes all --force-overwrite \
  --section SourceCounters \
  --section SchedulerStats \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --launch-skip 10 \
  --launch-count 10 \
  --kernel-name 'regex:.*cuda_fd3d_p_pml_len16_halfwarp.*'
```

The binary was built with `-lineinfo` for this profile.  After profiling, the
remote binary was rebuilt back to release len16 flags and smoke-tested.

Artifacts:

```text
reports/day_20260608/len16_source_profile_20260608_1646/details.csv
reports/day_20260608/len16_source_profile_20260608_1646/details_summary.md
reports/day_20260608/len16_source_profile_20260608_1646/details_summary.json
reports/day_20260608/len16_source_profile_20260608_1646/source_hotlines.md
reports/day_20260608/len16_source_profile_20260608_1646/source_hotlines.json
reports/day_20260608/len16_source_profile_20260608_1646/lineinfo_bin.sha256
reports/day_20260608/len16_source_profile_20260608_1646/final_release_bin.sha256
reports/day_20260608/len16_source_profile_20260608_1646/ncu_run.log
```

Large raw artifacts left only on the server:

```text
reports/day_20260608/len16_source_profile_20260608_1646/len16_source.ncu-rep
reports/day_20260608/len16_source_profile_20260608_1646/source_page.txt
reports/day_20260608/len16_source_profile_20260608_1646/source_page_cuda_sass.txt
```

Final release binary after restore:

```text
SHA256 77ba44c3f94fc5992b07b01ee786bfadf6c2a4671fc8e755dace2bcef9b31c58
```

Post-profile smoke:

```text
benchmarks/runs/smoke_1gpu_len16_after_source_profile_restore_20260608_165211
ALL DONE
```

## Kernel-Level Signals

```text
No Eligible                         73.545%
Eligible warps/scheduler             0.427
Issued warp/scheduler                0.264
Active warps/scheduler               8.986
Warp cycles/issued instruction      33.970
Avg active threads/warp             26.380
Avg not-predicated threads/warp     24.910
Branch efficiency                   65.220%
Avg divergent branches             316.610
Achieved occupancy                  74.912%
L1/TEX hit rate                     61.537%
L2 hit rate                         54.157%
```

Nsight Compute reports a CPI stall dominated by L1TEX scoreboard dependency,
about `24.6` cycles per warp.

## Source Hot Lines

The parsed source-correlated CUDA+SASS page produced `15,712` source-line
samples in the len16 kernel.  Top hot lines:

| source line | samples | not-issued | share | source |
| ---: | ---: | ---: | ---: | --- |
| 1813 | 5660 | 4643 | 36.02% | `p0[base]=2*__ldg(p1+base)-p0[base]` |
| 1814 | 3890 | 3161 | 24.76% | `+__ldg(cw2+base)*dt*(c1+c2+c3);` |
| 1810 | 3287 | 2612 | 20.92% | `mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);` |
| 1804 | 927 | 660 | 5.90% | `mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);` |
| 1778 | 266 | 156 | 1.69% | `const size_t ts2 = (size_t)(gtid2 + radius) * stride2;` |
| 1738 | 180 | 109 | 1.15% | `const PmlTile tile = tiles[blockIdx.x];` |
| 1784 | 111 | 59 | 0.71% | `vz_line_cache[cbase-2])` |

The final pressure update lines account for about `60.78%` of parsed samples.
The z-CPML `mem_dzz` update accounts for about `26.82%`.  The z-cache shared
load lines are visible but no longer dominant.

## Rejected Len16 Micro Routes

Do not implement these as len16-only micro prototypes:

- `p0` read syntax change such as `__ldg(p0 + base)`.
- explicit local `new_mem` accumulation for `mem_dzz`.
- branch-only lower/upper or margin specialization without a model showing
  `>=5%` repeat speedup ceiling.
- additional z-cache/shared-memory tuning inside the current len16 kernel.

Reasons:

- The same `p0` read-only-load and `new_mem` ideas were already rejected on the
  direct-fill pressure-PML path with noise-level gains.
- The source hot lines show memory dependency stalls on unavoidable pressure
  writeback and CPML state traffic, not a clear syntax-codegen miss.
- The branch line itself is not a dominant sample line; branch-only
  specialization would add tile lists and launches without a proven ceiling.

## Next Direction

The pressure-PML len16 route remains accepted as current best, but the next
meaningful CUDA route should move away from small source syntax changes.

Allowed next gates:

- v-PML memory layout / coalescing design.
- A broader pressure-PML memory-ownership design that reduces final `p0/cw2`
  traffic or CPML z-state dependency, with a modeled `>=5%` repeat ceiling
  before CUDA code.
