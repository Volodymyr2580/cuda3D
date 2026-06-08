# NCU CSV Summary

## Profiles

- `zmem`: `/work/wenzhe/cuda3D_codex_day_20260608_68de1a7/benchmarks/profiles/day_20260608/zmem_vs_directfill68_zmem_ncu.csv`
- `directfill`: `/work/wenzhe/cuda3D_codex_day_20260608_68de1a7/benchmarks/profiles/day_20260608/zmem_vs_directfill68_directfill_ncu.csv`

## `cuda_fd3d_p_core_ns`

| metric | zmem | directfill |
| --- | ---: | ---: |
| Duration | 75,942 ns | 75,270 ns |
| SOL compute | 56.864% | 57.384% |
| SOL memory | 95.163% | 95.892% |
| SOL DRAM | 40.868% | 41.092% |
| Mem pipes busy | 56.864% | 57.384% |
| L1/TEX hit | 35.955% | 35.942% |
| L2 hit | 86.156% | 86.231% |
| No eligible | 56.331% | 56.315% |
| Eligible warps/scheduler | 1.163 | 1.164 |
| Achieved occupancy | 66.356% | 66.361% |

## `cuda_fd3d_p_pml_tile_ns`

| metric | zmem | directfill |
| --- | ---: | ---: |
| Duration | 158,438 ns | 134,099 ns |
| SOL compute | 55.555% | 38.083% |
| SOL memory | 55.555% | 46.485% |
| SOL DRAM | 39.406% | 46.485% |
| Mem pipes busy | 55.555% | 36.155% |
| L1/TEX hit | 85.961% | 63.588% |
| L2 hit | 56.924% | 56.504% |
| No eligible | 47.116% | 59.885% |
| Eligible warps/scheduler | 1.159 | 0.820 |
| Achieved occupancy | 74.031% | 74.662% |

## `cuda_fd3d_v_pml_tile_ns`

| metric | zmem | directfill |
| --- | ---: | ---: |
| Duration | 58,794 ns | 53,590 ns |
| SOL compute | 45.379% | 50.688% |
| SOL memory | 57.285% | 62.744% |
| SOL DRAM | 42.425% | 47.024% |
| Mem pipes busy | 44.430% | 49.077% |
| L1/TEX hit | 56.677% | 55.005% |
| L2 hit | 68.336% | 66.174% |
| No eligible | 50.956% | 45.020% |
| Eligible warps/scheduler | 1.296 | 1.630 |
| Achieved occupancy | 83.919% | 84.602% |

## Rules
- `zmem` `cuda_fd3d_p_core_ns` `SOLBottleneck`: This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L2 in the Memory Workload Analysis section.
- `zmem` `cuda_fd3d_p_core_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.3 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `zmem` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `zmem` `cuda_fd3d_p_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 1.9 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `zmem` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `zmem` `cuda_fd3d_v_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.0 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `directfill` `cuda_fd3d_p_core_ns` `SOLBottleneck`: This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L2 in the Memory Workload Analysis section.
- `directfill` `cuda_fd3d_p_core_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.3 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `directfill` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `directfill` `cuda_fd3d_p_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.4 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `directfill` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the L2 bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more w...
- `directfill` `cuda_fd3d_v_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 1.8 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
