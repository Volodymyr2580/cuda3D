# NCU CSV Summary

## Profiles

- `zmem`: `benchmarks/profiles/day_20260608/zmem_pml_state_ncu.csv`
- `cpml_dbuf`: `benchmarks/profiles/day_20260608/cpml_dbuf_pml_state_ncu.csv`

## `cuda_fd3d_p_pml_tile_ns`

| metric | zmem | cpml_dbuf |
| --- | ---: | ---: |
| Duration | 189,840 ns | 190,293 ns |
| SOL compute | 55.965% | 56.005% |
| SOL memory | 55.965% | 56.005% |
| SOL DRAM | 40.603% | 39.958% |
| Mem pipes busy | 55.965% | 56.005% |
| L1/TEX hit | 85.978% | 85.957% |
| L2 hit | 56.932% | 56.988% |
| No eligible | - | - |
| Eligible warps/scheduler | - | - |
| Achieved occupancy | - | - |

## `cuda_fd3d_v_pml_tile_ns`

| metric | zmem | cpml_dbuf |
| --- | ---: | ---: |
| Duration | 71,493 ns | 66,000 ns |
| SOL compute | 44.872% | 49.392% |
| SOL memory | 55.435% | 59.955% |
| SOL DRAM | 45.010% | 49.463% |
| Mem pipes busy | 44.003% | 47.913% |
| L1/TEX hit | 56.527% | 54.893% |
| L2 hit | 67.908% | 66.095% |
| No eligible | - | - |
| Eligible warps/scheduler | - | - |
| Achieved occupancy | - | - |

## Rules
- `zmem` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `zmem` `cuda_fd3d_p_pml_tile_ns` `CPIStall`: On average, each warp of this workload spends 8.9 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles ...
- `zmem` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `zmem` `cuda_fd3d_v_pml_tile_ns` `CPIStall`: On average, each warp of this workload spends 14.4 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles...
- `cpml_dbuf` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `cpml_dbuf` `cuda_fd3d_p_pml_tile_ns` `CPIStall`: On average, each warp of this workload spends 8.6 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles ...
- `cpml_dbuf` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the L2 bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more w...
- `cpml_dbuf` `cuda_fd3d_v_pml_tile_ns` `CPIStall`: On average, each warp of this workload spends 11.7 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. Find the instruction producing the data being waited upon to identify the culprit. To reduce the number of cycles...
