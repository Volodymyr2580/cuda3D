# Direct-Fill Pressure PML Source Profile

## Run

```text
worktree       /work/wenzhe/cuda3D_codex_day_20260608_68de1a7
commit         68de1a7 source with direct-fill z-cache
flags          direct-fill combo + -lineinfo
kernel         cuda_fd3d_p_pml_tile_ns
sections       SourceCounters, SchedulerStats, WarpStateStats
launches       skip 10, count 10
raw csv        reports/day_20260608/directfill_p_pml_source_ncu.csv
```

The full Nsight Compute source page was generated on the server as
`reports/day_20260608/directfill_source_page_20260608_122754/source_page.txt`
but was not committed because it is about 19 MB.

## Main Signals

```text
No Eligible                         about 60%
Eligible warps/scheduler            about 0.81
L1TEX scoreboard stall               about 14.4 cycles/warp
Uncoalesced global accesses          about 19% excessive sectors
Branch efficiency                    about 75.96%
Avg active threads/warp              about 19.84
Avg not-predicated-off threads/warp  about 18.69
```

## Top Source Lines

The largest source-page sample counts are no longer in z-cache fill.  They
are concentrated in CPML memory updates and final pressure writeback:

| line | code | profile read |
| ---: | --- | --- |
| 1951 | `mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);` | z-upper CPML memory load/store has high samples and L1TEX scoreboard stalls. |
| 1984 | `p0[outIndex]=2*__ldg(p1+outIndex)-p0[outIndex]` | final pressure update has high global load/store samples. |
| 1944 | `mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);` | z-lower CPML memory update remains significant. |
| 1979 | `mem_dyy[pind]=mem_dyy[pind]*coef+c3*(coef-1);` | y-upper CPML memory update is significant. |
| 1972 | `mem_dyy[pind]=mem_dyy[pind]*coef+c3*(coef-1);` | y-lower CPML memory update is significant. |
| 1958/1965 | `mem_dxx[pind]=mem_dxx[pind]*coef+c2*(coef-1);` | x CPML memory updates are present but lower than z/y and p0 writeback. |
| 1823-1830 | `vz_line_cache[...]` shared loads | z-cache shared loads are visible but no longer dominant. |

## Follow-Up Tested

`pml_local_mem_accum` rewrote the CPML memory updates to use an explicit local
`new_mem` variable before writing back and accumulating into `c1/c2/c3`.

Result:

```text
correctness rel L2                   0 for 6 outputs
perf6 output compares                pass
mean WP speedup vs direct-fill       1.000647x
mean Gradient speedup vs direct-fill 0.998957x
```

Decision:

```text
Reject local new_mem accumulation as a performance optimization.  The compiler
already handles the apparent write-then-read expression well enough; this does
not meet the >=2% small-candidate gate.
```

## Next Direction

The next meaningful pressure-PML work should not target the z-cache fill or
the syntactic memory-update expression.  It should target larger structure:

```text
1. Reduce PML pressure divergence / low active threads per warp.
2. Reduce CPML memory-update traffic or split ownership in a way that lowers
   global load/store scoreboard stalls.
3. Avoid repeating old TILE_MASK_FASTPATH and z-face fusion attempts unless
   new profiler evidence changes the design.
```
