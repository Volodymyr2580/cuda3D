# NCU CSV Summary

## Profiles

- `zmem_core_pml`: `benchmarks/profiles/day_20260608/zmem_core_pml_sol_ncu.csv`

## `cuda_fd3d_p_core_ns`

| metric | zmem_core_pml |
| --- | ---: |
| Duration | 93,670 ns |
| SOL compute | 58.483% |
| SOL memory | 96.810% |
| SOL DRAM | 42.257% |
| Mem pipes busy | - |
| L1/TEX hit | - |
| L2 hit | - |
| No eligible | - |
| Eligible warps/scheduler | - |
| Achieved occupancy | - |

## `cuda_fd3d_p_pml_tile_ns`

| metric | zmem_core_pml |
| --- | ---: |
| Duration | 189,562 ns |
| SOL compute | 56.021% |
| SOL memory | 56.021% |
| SOL DRAM | 40.457% |
| Mem pipes busy | - |
| L1/TEX hit | - |
| L2 hit | - |
| No eligible | - |
| Eligible warps/scheduler | - |
| Achieved occupancy | - |

## `cuda_fd3d_v_pml_tile_ns`

| metric | zmem_core_pml |
| --- | ---: |
| Duration | 71,610 ns |
| SOL compute | 44.572% |
| SOL memory | 55.251% |
| SOL DRAM | 45.121% |
| Mem pipes busy | - |
| L1/TEX hit | - |
| L2 hit | - |
| No eligible | - |
| Eligible warps/scheduler | - |
| Achieved occupancy | - |

## Rules
- `zmem_core_pml` `cuda_fd3d_p_core_ns` `SOLBottleneck`: This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L2 in the Memory Workload Analysis section.
- `zmem_core_pml` `cuda_fd3d_p_core_ns` `SOLBottleneck`: This workload is utilizing greater than 80.0% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing L2 in the Memory Workload Analysis section.
- `zmem_core_pml` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `zmem_core_pml` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `zmem_core_pml` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `zmem_core_pml` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
