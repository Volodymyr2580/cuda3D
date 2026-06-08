# NCU CSV Summary

## Profiles

- `zmem`: `benchmarks/profiles/day_20260608/zmem_vs_combo_zmem_ncu.csv`
- `combo`: `benchmarks/profiles/day_20260608/zmem_vs_combo_combo_ncu.csv`

## `cuda_fd3d_p_core_ns`

| metric | zmem | combo |
| --- | ---: | ---: |
| Duration | 76,061 ns | 75,306 ns |
| SOL compute | 56.827% | 57.454% |
| SOL memory | 95.068% | 95.875% |
| SOL DRAM | 41.040% | 41.299% |
| Mem pipes busy | 56.827% | 57.454% |
| L1/TEX hit | 35.889% | 35.916% |
| L2 hit | 86.200% | 86.189% |
| No eligible | 56.581% | 56.328% |
| Eligible warps/scheduler | 1.157 | 1.165 |
| Achieved occupancy | 66.361% | 66.359% |

## `cuda_fd3d_p_pml_tile_ns`

| metric | zmem | combo |
| --- | ---: | ---: |
| Duration | 158,291 ns | 142,902 ns |
| SOL compute | 55.738% | 37.291% |
| SOL memory | 55.738% | 43.994% |
| SOL DRAM | 39.645% | 43.994% |
| Mem pipes busy | 55.738% | 33.169% |
| L1/TEX hit | 86.084% | 64.186% |
| L2 hit | 56.699% | 56.651% |
| No eligible | 47.114% | 60.879% |
| Eligible warps/scheduler | 1.166 | 0.798 |
| Achieved occupancy | 74.034% | 75.965% |

## `cuda_fd3d_v_pml_tile_ns`

| metric | zmem | combo |
| --- | ---: | ---: |
| Duration | 58,320 ns | 53,101 ns |
| SOL compute | 45.505% | 51.182% |
| SOL memory | 57.667% | 63.279% |
| SOL DRAM | 42.830% | 47.693% |
| Mem pipes busy | 44.556% | 49.558% |
| L1/TEX hit | 56.785% | 55.066% |
| L2 hit | 68.278% | 66.284% |
| No eligible | 50.560% | 44.692% |
| Eligible warps/scheduler | 1.322 | 1.649 |
| Achieved occupancy | 83.846% | 84.840% |

## Rules
- `zmem` `cuda_fd3d_p_core_ns` `SOLBottleneck`: This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L2 in the Memory Workload Analysis section.
- `zmem` `cuda_fd3d_p_core_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.3 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `zmem` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `zmem` `cuda_fd3d_p_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 1.9 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `zmem` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `zmem` `cuda_fd3d_v_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.0 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `combo` `cuda_fd3d_p_core_ns` `SOLBottleneck`: This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L2 in the Memory Workload Analysis section.
- `combo` `cuda_fd3d_p_core_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.3 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `combo` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `combo` `cuda_fd3d_p_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.6 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `combo` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the L2 bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more w...
- `combo` `cuda_fd3d_v_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 1.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
