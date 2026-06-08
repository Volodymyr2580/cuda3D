# NCU CSV Summary

## Profiles

- `len16_source`: `reports/day_20260608/len16_source_profile_20260608_1646/details.csv`

## `cuda_fd3d_p_pml_len16_halfwarp_ns`

| metric | len16_source |
| --- | ---: |
| Duration | - |
| SOL compute | - |
| SOL memory | - |
| SOL DRAM | - |
| Mem pipes busy | 28.451% |
| L1/TEX hit | 61.537% |
| L2 hit | 54.157% |
| No eligible | 73.545% |
| Issued warp/scheduler | 0.264 |
| Active warps/scheduler | 8.986 |
| Eligible warps/scheduler | 0.427 |
| Warp cycles/issued inst | 33.970 |
| Avg active threads/warp | 26.380 |
| Avg not-predicated threads/warp | 24.910 |
| Branch efficiency | 65.220% |
| Branch instructions | 1,264,864 |
| Avg divergent branches | 316.610 |
| Achieved occupancy | 74.912% |

## Rules
- `len16_source` `cuda_fd3d_p_pml_len16_halfwarp_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 3.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `len16_source` `cuda_fd3d_p_pml_len16_halfwarp_ns` `CPIStall`: On average, each warp of this workload spends 24.6 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles...
