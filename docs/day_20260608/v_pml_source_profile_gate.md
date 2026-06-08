# V-PML Source Profile Gate

Date: 2026-06-08

Context:

```text
Current best is the direct-fill pressure z-cache combo:
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_CPML_VMEM_DISABLE_MPI
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

## NCU SourceCounters Profile

Target kernel:

```text
cuda_fd3d_v_pml_tile_ns
```

Profile command shape:

```text
sections      SourceCounters, SchedulerStats, WarpStateStats
launch-skip   10
launch-count  10
kernel        regex:.*cuda_fd3d_v_pml_tile.*
case          perf_1gpu_6shots
flags         current best + -lineinfo
```

Committed reports:

```text
reports/day_20260608/directfill_v_pml_source_ncu.csv
reports/day_20260608/directfill_v_pml_source_summary.md
reports/day_20260608/directfill_v_pml_source_summary.json
```

Large remote-only artifacts:

```text
/work/wenzhe/cuda3D_codex_day_20260608_68de1a7/reports/day_20260608/v_pml_source_profile_20260608_140439/source_page.txt
/work/wenzhe/cuda3D_codex_day_20260608_68de1a7/reports/day_20260608/v_pml_source_profile_20260608_140439/source_page_cuda_sass.txt
```

Key metrics:

| metric | value |
| --- | ---: |
| No eligible | 44.891% |
| Eligible warps/scheduler | 1.629 |
| Active warps/scheduler | 10.170 |
| Warp cycles/issued inst | 18.456 |
| Avg active threads/warp | 23.700 |
| Avg not-predicated threads/warp | 21.670 |
| Branch efficiency | 86.970% |
| Branch instructions | 2,079,334 |
| Avg divergent branches | 143.480 |

NCU rule signals:

```text
IssueSlotUtilization       local estimated speedup about 44%
CPIStall                   about 11.8 cycles/warp waiting on L1TEX scoreboard
ThreadDivergence           active threads 23.7, predication lowers to 21.7
UncoalescedGlobalAccess    about 2,431,758 excessive sectors, about 22%
```

Interpretation:

```text
v_pml_tile is still memory-latency sensitive, but it is less branch-divergent
than direct-fill p_pml_tile.  The strongest signal is global-memory access
quality / L1TEX scoreboard, not a simple expression-level rewrite.
```

## Component Split Static Budget

Candidate considered:

```text
Split current vx/vy velocity-PML work into separate component-owner kernels:
one kernel for vx/mem_dx, one kernel for vy/mem_dy.
```

Static budget for `384 x 384 x 95`, `PmlTileBlockSize=32x4x2`, `npml=12`:

| shape | tiles | launched threads | active component points | active efficiency |
| --- | ---: | ---: | ---: | ---: |
| current combined vx/vy | 41,100 | 10,521,600 | 7,457,823 | 70.881% |
| vx-only component | 40,848 | 10,457,088 | 7,322,562 | 70.025% |
| vy-only component | 40,762 | 10,435,072 | 7,322,562 | 70.173% |

Derived budget:

```text
split tile sum / combined tiles          1.985645x
split active work sum / combined active  1.963726x
overlap tiles                            40,510
```

Decision:

```text
Reject component-owner split before CUDA implementation.  It would nearly
double tile launches and active component work because vx/vy ownership
strongly overlaps in the current tile geometry.
```

Follow-up boundary:

```text
Do not implement vx/vy split kernels with the current 32x4x2 tile geometry.
Future v_pml work needs a layout or memory-coalescing design that reduces the
22% excessive sector / L1TEX scoreboard signal without doubling component work.
```
