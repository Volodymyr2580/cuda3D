# NCU CSV Summary

## Profiles

- `directfill`: `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/directfill_profile_ncu_details.csv`
- `len16`: `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/len16_profile_ncu_details.csv`

## `cuda_fd3d_p_core_ns`

| metric | directfill | len16 |
| --- | ---: | ---: |
| Duration | 93,752 ns | 93,547 ns |
| SOL compute | 58.297% | 58.720% |
| SOL memory | 96.640% | 96.890% |
| SOL DRAM | 42.345% | 42.167% |
| Mem pipes busy | 58.297% | 58.720% |
| L1/TEX hit | 35.847% | 35.853% |
| L2 hit | 86.362% | 86.340% |
| No eligible | 55.695% | 55.707% |
| Issued warp/scheduler | 0.443 | 0.440 |
| Active warps/scheduler | 7.933 | 7.930 |
| Eligible warps/scheduler | 1.188 | 1.203 |
| Warp cycles/issued inst | 17.907 | 17.910 |
| Avg active threads/warp | 29.220 | 29.220 |
| Avg not-predicated threads/warp | 28.950 | 28.950 |
| Branch efficiency | 75.000% | 75.000% |
| Branch instructions | 871,653 | 871,653 |
| Avg divergent branches | 80.180 | 80.180 |
| Achieved occupancy | 66.197% | 66.047% |

## `cuda_fd3d_p_pml_len16_halfwarp_ns`

| metric | directfill | len16 |
| --- | ---: | ---: |
| Duration | - | 65,771 ns |
| SOL compute | - | 28.347% |
| SOL memory | - | 58.440% |
| SOL DRAM | - | 58.440% |
| Mem pipes busy | - | 28.347% |
| L1/TEX hit | - | 61.563% |
| L2 hit | - | 54.110% |
| No eligible | - | 73.827% |
| Issued warp/scheduler | - | 0.260 |
| Active warps/scheduler | - | 8.953 |
| Eligible warps/scheduler | - | 0.433 |
| Warp cycles/issued inst | - | 34.210 |
| Avg active threads/warp | - | 26.380 |
| Avg not-predicated threads/warp | - | 24.910 |
| Branch efficiency | - | 65.220% |
| Branch instructions | - | 1,264,864 |
| Avg divergent branches | - | 316.610 |
| Achieved occupancy | - | 74.070% |

## `cuda_fd3d_p_pml_tile_ns`

| metric | directfill | len16 |
| --- | ---: | ---: |
| Duration | 164,328 ns | 72,683 ns |
| SOL compute | 37.282% | 34.577% |
| SOL memory | 46.303% | 41.787% |
| SOL DRAM | 46.303% | 41.787% |
| Mem pipes busy | 35.203% | 34.577% |
| L1/TEX hit | 63.348% | 64.403% |
| L2 hit | 56.547% | 59.640% |
| No eligible | 61.170% | 63.497% |
| Issued warp/scheduler | 0.388 | 0.363 |
| Active warps/scheduler | 8.883 | 8.637 |
| Eligible warps/scheduler | 0.775 | 0.733 |
| Warp cycles/issued inst | 22.870 | 23.660 |
| Avg active threads/warp | 19.680 | 22.950 |
| Avg not-predicated threads/warp | 18.550 | 21.650 |
| Branch efficiency | 75.530% | 83.320% |
| Branch instructions | 6,715,641 | 2,678,841 |
| Avg divergent branches | 1,118 | 326.740 |
| Achieved occupancy | 73.575% | 71.503% |

## `cuda_fd3d_v_pml_tile_ns`

| metric | directfill | len16 |
| --- | ---: | ---: |
| Duration | 65,528 ns | 65,248 ns |
| SOL compute | 49.373% | 49.920% |
| SOL memory | 60.410% | 60.633% |
| SOL DRAM | 49.558% | 49.347% |
| Mem pipes busy | 47.895% | 48.427% |
| L1/TEX hit | 54.888% | 54.903% |
| L2 hit | 66.052% | 66.023% |
| No eligible | 46.328% | 45.687% |
| Issued warp/scheduler | 0.538 | 0.543 |
| Active warps/scheduler | 9.950 | 9.957 |
| Eligible warps/scheduler | 1.545 | 1.557 |
| Warp cycles/issued inst | 18.538 | 18.330 |
| Avg active threads/warp | 23.370 | 23.370 |
| Avg not-predicated threads/warp | 21.370 | 21.370 |
| Branch efficiency | 86.490% | 86.490% |
| Branch instructions | 2,498,080 | 2,498,080 |
| Avg divergent branches | 177.890 | 177.890 |
| Achieved occupancy | 83.243% | 83.307% |

## Rules
- `directfill` `cuda_fd3d_p_core_ns` `SOLBottleneck`: This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L2 in the Memory Workload Analysis section.
- `directfill` `cuda_fd3d_p_core_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.2 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `directfill` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `directfill` `cuda_fd3d_p_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.6 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `directfill` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `directfill` `cuda_fd3d_v_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 1.9 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `len16` `cuda_fd3d_p_core_ns` `SOLBottleneck`: This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L2 in the Memory Workload Analysis section.
- `len16` `cuda_fd3d_p_core_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.3 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `len16` `cuda_fd3d_p_pml_len16_halfwarp_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `len16` `cuda_fd3d_p_pml_len16_halfwarp_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 3.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `len16` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `len16` `cuda_fd3d_p_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.7 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `len16` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the L2 bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more w...
- `len16` `cuda_fd3d_v_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 1.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
