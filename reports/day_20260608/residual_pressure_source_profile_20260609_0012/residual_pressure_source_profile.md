# Residual Pressure-PML Source Profile

## Context

- candidate: `current_best_v_pml_len16` with `-lineinfo`
- kernel: `cuda_fd3d_p_pml_tile_ns` residual pressure-PML path
- NCU: `--launch-skip 10 --launch-count 10`
- sections: SourceCounters, SchedulerStats, WarpStateStats, MemoryWorkloadAnalysis, Occupancy

## Key Metrics

| metric | value |
| --- | ---: |
| No Eligible | `63.162%` |
| Eligible warps/scheduler | `0.766` |
| Warp cycles/issued inst | `23.682` |
| Avg active threads/warp | `23.050` |
| Avg not-predicated threads/warp | `21.730` |
| Branch efficiency | `83.750%` |
| L1/TEX hit | `64.758%` |
| L2 hit | `59.332%` |
| Achieved occupancy | `73.389%` |

## Gate Notes

- This profile is evidence-gathering only; no CUDA source was modified.
- The full `.ncu-rep` is kept on the remote worktree and is not committed.
