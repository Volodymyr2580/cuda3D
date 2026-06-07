# NCU CSV Summary

## Profiles

- `zmem`: `benchmarks\profiles\wavestep_v2_night_20260608\zmem_pml_short.csv`
- `cpml_vmem`: `benchmarks\profiles\wavestep_v2_night_20260608\cpml_vmem_double_buffer_all_pml_short.csv`
- `direct_inline`: `benchmarks\profiles\wavestep_v2_night_20260608\direct_inline_fused_zface_pml_short.csv`

## `cuda_fd3d_p_pml_tile_ns`

| metric | zmem | cpml_vmem | direct_inline |
| --- | ---: | ---: | ---: |
| Duration | 188,856 ns | 189,896 ns | 248,200 ns |
| SOL compute | 55.745% | 55.788% | 69.227% |
| SOL memory | 55.745% | 55.788% | 69.227% |
| SOL DRAM | 40.195% | 40.125% | 28.700% |
| Mem pipes busy | 55.745% | 55.788% | 69.227% |
| L1/TEX hit | 86.005% | 86.005% | 85.890% |
| L2 hit | 56.782% | 56.693% | 67.088% |
| No eligible | 47.483% | 46.968% | 39.163% |
| Eligible warps/scheduler | 1.140 | 1.150 | 1.620 |
| Achieved occupancy | 72.797% | 72.757% | 74.903% |

## `cuda_fd3d_v_pml_tile_ns`

| metric | zmem | cpml_vmem | direct_inline |
| --- | ---: | ---: | ---: |
| Duration | 71,872 ns | 65,528 ns | 63,744 ns |
| SOL compute | 44.237% | 49.498% | 55.780% |
| SOL memory | 55.055% | 60.320% | 53.477% |
| SOL DRAM | 44.915% | 49.385% | 40.175% |
| Mem pipes busy | 43.380% | 48.017% | 48.828% |
| L1/TEX hit | 56.520% | 54.898% | 55.020% |
| L2 hit | 67.948% | 66.132% | 64.257% |
| No eligible | 52.320% | 46.418% | 40.455% |
| Eligible warps/scheduler | 1.230 | 1.538 | 1.853 |
| Achieved occupancy | 82.373% | 82.875% | 83.700% |

## Rules
- `zmem` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `zmem` `cuda_fd3d_p_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 1.9 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `zmem` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `zmem` `cuda_fd3d_v_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 2.1 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `cpml_vmem` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `cpml_vmem` `cuda_fd3d_p_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 1.9 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `cpml_vmem` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the L2 bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more w...
- `cpml_vmem` `cuda_fd3d_v_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 1.9 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
- `direct_inline` `cuda_fd3d_p_pml_tile_ns` `SOLBottleneck`: Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced. Check both the Compute Workload Analysis and Memory Workload Analysis sections.
- `direct_inline` `cuda_fd3d_p_pml_tile_ns` `CPIStall`: The optional metric smsp__pcsamp_sample_count could not be found. Collecting it as an additional metric could enable the rule to provide more guidance.
- `direct_inline` `cuda_fd3d_v_pml_tile_ns` `SOLBottleneck`: This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp Stat...
- `direct_inline` `cuda_fd3d_v_pml_tile_ns` `IssueSlotUtilization`: Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 1.7 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per sch...
