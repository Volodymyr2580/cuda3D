# NCU CSV Summary

## Profiles

- `directfill_v_pml`: `benchmarks/profiles/day_20260608/directfill_v_pml_source_ncu.csv`

## `cuda_fd3d_v_pml_tile_ns`

| metric | directfill_v_pml |
| --- | ---: |
| Duration | - |
| SOL compute | - |
| SOL memory | - |
| SOL DRAM | - |
| Mem pipes busy | - |
| L1/TEX hit | - |
| L2 hit | - |
| No eligible | 44.891% |
| Issued warp/scheduler | 0.549 |
| Active warps/scheduler | 10.170 |
| Eligible warps/scheduler | 1.629 |
| Warp cycles/issued inst | 18.456 |
| Avg active threads/warp | 23.700 |
| Avg not-predicated threads/warp | 21.670 |
| Branch efficiency | 86.970% |
| Branch instructions | 2,079,334 |
| Avg divergent branches | 143.480 |
| Achieved occupancy | - |

## Rules
- `directfill_v_pml` `cuda_fd3d_v_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 1.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `directfill_v_pml` `cuda_fd3d_v_pml_tile_ns` `CPIStall`: On average, each warp of this workload spends 11.8 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles...
