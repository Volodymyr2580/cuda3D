# V-PML Len16 Candidate NCU Profile

## Context

- Candidate macro: `CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK`.
- Remote worktree: `/work/wenzhe/cuda3D/.codex_worktrees/v_pml_len16_20260608_2238`.
- Profile case: `profile_1gpu`.
- NCU mode: short profile with `--launch-count 5`.
- Kernel sequence captured:
  - `cuda_fd3d_v_pml_len16_halfwarp_ns`
  - `cuda_fd3d_v_pml_tile_ns`
  - `cuda_fd3d_p_core_ns`
  - `cuda_fd3d_p_pml_len16_halfwarp_ns`
  - `cuda_fd3d_p_pml_tile_ns`

The raw NCU CSV reports duration in `us`.  The existing `tools/ncu_csv_summary.py` labels the generic duration row as `ns`, so this document treats those values as microseconds.

## Kernel Durations

| kernel | duration |
| --- | ---: |
| `cuda_fd3d_p_core_ns` | `93.730us` |
| `cuda_fd3d_p_pml_len16_halfwarp_ns` | `66.180us` |
| `cuda_fd3d_p_pml_tile_ns` | `71.940us` |
| `cuda_fd3d_v_pml_len16_halfwarp_ns` | `20.030us` |
| `cuda_fd3d_v_pml_tile_ns` | `32.130us` |

Sampled main total: `284.010us`.

| group | duration | share |
| --- | ---: | ---: |
| p-core | `93.730us` | `33.00%` |
| pressure-PML total | `138.120us` | `48.63%` |
| velocity-PML total | `52.160us` | `18.37%` |

## Signals

- `cuda_fd3d_p_core_ns`
  - SOL memory: `96.790%`.
  - L2 hit: `86.300%`.
  - Interpretation: still L2/memory limited; previously rejected shared-plane shape family remains rejected.
- `cuda_fd3d_p_pml_len16_halfwarp_ns`
  - duration: `66.180us`.
  - No Eligible: `73.850%`.
  - eligible warps/scheduler: `0.420`.
  - Interpretation: still latency/issue limited, matching the earlier source-level profile.
- `cuda_fd3d_p_pml_tile_ns`
  - duration: `71.940us`.
  - branch efficiency: `83.320%`.
  - Interpretation: residual pressure-PML remains meaningful, but branch/control-only len32 specialization was already modeled below the 5% gate.
- `cuda_fd3d_v_pml_len16_halfwarp_ns`
  - duration: `20.030us`.
  - avg active threads/warp: `32.000`.
  - avg not-predicated threads/warp: `31.550`.
  - Interpretation: packed v-PML is doing what it was designed to do; remaining gains from this kernel alone cannot move total runtime much.
- `cuda_fd3d_v_pml_tile_ns`
  - duration: `32.130us`.
  - branch efficiency: `94.770%`.
  - Interpretation: residual v-PML is smaller and cleaner after packing.

## Gate

- v-PML total share after packing is only `18.37%` of sampled main.
- An additional standalone v-PML route would need about `1.276x` total v-PML speedup to produce a `>=5%` sampled-main improvement.
- The already tested whole-len16 packing delivered `1.032058x` WP and `1.028730x` Gradient on `perf_1gpu_6shots`.
- Incremental line/point descriptor packing after whole-len16 has too little remaining sampled-main headroom unless descriptor/control overhead is proven extremely low.

Decision:

- Keep `CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK` as a minor current-best candidate.
- Do not continue v-PML descriptor packing, exact active-point lists, or tile-shape sweeps without a new overhead model and source-level evidence.
- Next CUDA-core work should prioritize pressure-PML or a materially new p-core design, but only after a model proves `>=5%` repeat-speedup ceiling.

## Artifacts

- `reports/day_20260608/v_pml_len16_ncu_short_20260608_2315/v_pml_len16_short.csv`
- `reports/day_20260608/v_pml_len16_ncu_short_20260608_2315/summary.json`
- `reports/day_20260608/v_pml_len16_ncu_short_20260608_2315/summary.md`
