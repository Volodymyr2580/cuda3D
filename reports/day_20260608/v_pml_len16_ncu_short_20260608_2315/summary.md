# NCU CSV Summary

## Profiles

- `v_pml_len16_short`: `reports/day_20260608/v_pml_len16_ncu_short_20260608_2315/v_pml_len16_short.csv`

## `cuda_fd3d_p_core_ns`

| metric | v_pml_len16_short |
| --- | ---: |
| Duration | 93.730 ns |
| SOL compute | 58.590% |
| SOL memory | 96.790% |
| SOL DRAM | 42.360% |
| Mem pipes busy | 58.590% |
| L1/TEX hit | 35.810% |
| L2 hit | 86.300% |
| No eligible | 56.540% |
| Issued warp/scheduler | 0.430 |
| Active warps/scheduler | 7.930 |
| Eligible warps/scheduler | 1.160 |
| Warp cycles/issued inst | 18.240 |
| Avg active threads/warp | 29.220 |
| Avg not-predicated threads/warp | 28.950 |
| Branch efficiency | 75.000% |
| Branch instructions | 871,653 |
| Avg divergent branches | 80.180 |
| Achieved occupancy | 67.730% |

## `cuda_fd3d_p_pml_len16_halfwarp_ns`

| metric | v_pml_len16_short |
| --- | ---: |
| Duration | 66.180 ns |
| SOL compute | 28.090% |
| SOL memory | 59.580% |
| SOL DRAM | 59.580% |
| Mem pipes busy | 28.090% |
| L1/TEX hit | 61.550% |
| L2 hit | 54.080% |
| No eligible | 73.850% |
| Issued warp/scheduler | 0.260 |
| Active warps/scheduler | 8.980 |
| Eligible warps/scheduler | 0.420 |
| Warp cycles/issued inst | 34.330 |
| Avg active threads/warp | 26.380 |
| Avg not-predicated threads/warp | 24.910 |
| Branch efficiency | 65.220% |
| Branch instructions | 1,264,864 |
| Avg divergent branches | 316.610 |
| Achieved occupancy | 75.550% |

## `cuda_fd3d_p_pml_tile_ns`

| metric | v_pml_len16_short |
| --- | ---: |
| Duration | 71.940 ns |
| SOL compute | 35.000% |
| SOL memory | 42.130% |
| SOL DRAM | 42.130% |
| Mem pipes busy | 35.000% |
| L1/TEX hit | 64.380% |
| L2 hit | 59.580% |
| No eligible | 63.180% |
| Issued warp/scheduler | 0.370 |
| Active warps/scheduler | 8.630 |
| Eligible warps/scheduler | 0.740 |
| Warp cycles/issued inst | 23.420 |
| Avg active threads/warp | 22.950 |
| Avg not-predicated threads/warp | 21.650 |
| Branch efficiency | 83.320% |
| Branch instructions | 2,678,841 |
| Avg divergent branches | 326.740 |
| Achieved occupancy | 71.040% |

## `cuda_fd3d_v_pml_len16_halfwarp_ns`

| metric | v_pml_len16_short |
| --- | ---: |
| Duration | 20.030 ns |
| SOL compute | 35.250% |
| SOL memory | 73.790% |
| SOL DRAM | 49.690% |
| Mem pipes busy | 35.250% |
| L1/TEX hit | 55.360% |
| L2 hit | 65.350% |
| No eligible | 61.070% |
| Issued warp/scheduler | 0.390 |
| Active warps/scheduler | 10.170 |
| Eligible warps/scheduler | 0.910 |
| Warp cycles/issued inst | 26.130 |
| Avg active threads/warp | 32.000 |
| Avg not-predicated threads/warp | 31.550 |
| Branch efficiency | 0.000% |
| Branch instructions | 150,528 |
| Avg divergent branches | 0.000 |
| Achieved occupancy | 85.640% |

## `cuda_fd3d_v_pml_tile_ns`

| metric | v_pml_len16_short |
| --- | ---: |
| Duration | 32.130 ns |
| SOL compute | 45.220% |
| SOL memory | 72.450% |
| SOL DRAM | 38.150% |
| Mem pipes busy | 44.590% |
| L1/TEX hit | 53.620% |
| L2 hit | 68.930% |
| No eligible | 48.860% |
| Issued warp/scheduler | 0.510 |
| Active warps/scheduler | 9.650 |
| Eligible warps/scheduler | 1.460 |
| Warp cycles/issued inst | 18.880 |
| Avg active threads/warp | 29.360 |
| Avg not-predicated threads/warp | 26.940 |
| Branch efficiency | 94.770% |
| Branch instructions | 1,093,152 |
| Avg divergent branches | 30.310 |
| Achieved occupancy | 80.540% |

## Rules
- `v_pml_len16_short` `cuda_fd3d_p_core_ns` `SOLBottleneck`: This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L2 in the Memory Workload Analysis section.
- `v_pml_len16_short` `cuda_fd3d_p_core_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.3 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `v_pml_len16_short` `cuda_fd3d_p_pml_len16_halfwarp_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `v_pml_len16_short` `cuda_fd3d_p_pml_len16_halfwarp_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 3.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `v_pml_len16_short` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `v_pml_len16_short` `cuda_fd3d_p_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.7 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `v_pml_len16_short` `cuda_fd3d_v_pml_len16_halfwarp_ns` `SOLBottleneck`: Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the L2 bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more w...
- `v_pml_len16_short` `cuda_fd3d_v_pml_len16_halfwarp_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.6 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `v_pml_len16_short` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the L2 bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more w...
- `v_pml_len16_short` `cuda_fd3d_v_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.0 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
