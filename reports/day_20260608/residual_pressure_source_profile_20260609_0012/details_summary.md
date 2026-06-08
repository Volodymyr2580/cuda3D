# NCU CSV Summary

## Profiles

- `residual_p_pml_tile_source`: `reports/day_20260608/residual_pressure_source_profile_20260609_0012/residual_p_pml_tile_source.csv`

## `cuda_fd3d_p_pml_tile_ns`

| metric | residual_p_pml_tile_source |
| --- | ---: |
| Duration | - |
| SOL compute | - |
| SOL memory | - |
| SOL DRAM | - |
| Mem pipes busy | 35.193% |
| L1/TEX hit | 64.758% |
| L2 hit | 59.332% |
| No eligible | 63.162% |
| Issued warp/scheduler | 0.369 |
| Active warps/scheduler | 8.725 |
| Eligible warps/scheduler | 0.766 |
| Warp cycles/issued inst | 23.682 |
| Avg active threads/warp | 23.050 |
| Avg not-predicated threads/warp | 21.730 |
| Branch efficiency | 83.750% |
| Branch instructions | 2,293,760 |
| Avg divergent branches | 274.070 |
| Achieved occupancy | 73.389% |

## Rules
- `residual_p_pml_tile_source` `cuda_fd3d_p_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.7 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `residual_p_pml_tile_source` `cuda_fd3d_p_pml_tile_ns` `CPIStall`: On average, each warp of this workload spends 15.9 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles...
