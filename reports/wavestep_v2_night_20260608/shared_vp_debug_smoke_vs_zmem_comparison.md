# Output Comparison

- Pass: `True`
- Baseline: `benchmarks/runs/smoke_1gpu_zmem_reference_night_smoke_20260608_013204/outputs`
- Candidate: `/work/wenzhe/cuda3D/benchmarks/runs/smoke_1gpu_shared_vp_debug_smoke_20260608_020046/outputs`
- Missing files: `0`
- Extra files: `0`

| File | Pass | Criterion | Rel L2 | Max Abs | Max Rel | RMS |
|---|---:|---|---:|---:|---:|---:|
| d_obs_salt_gpu_cpu_checked_ricker1_8hz_3d_ny_384_nx_384_nz95_nbell_1_bscl_0.9_moffy_9.5625_moffx_9.5625_h_obs_nt_1501_dt_2ms_shot_0.dir | True | rel_l2 | 8.053794e-08 | 7.152557e-07 | 4.264451e-07 | 5.440062e-08 |
| d_obs_salt_gpu_cpu_checked_ricker1_8hz_3d_ny_384_nx_384_nz95_nbell_1_bscl_0.9_moffy_9.5625_moffx_9.5625_h_obs_nt_1501_dt_2ms_shot_1.dir | True | rel_l2 | 3.120506e-08 | 4.768372e-07 | 5.780939e-06 | 2.139079e-08 |
| d_obs_salt_gpu_cpu_checked_ricker1_8hz_3d_ny_384_nx_384_nz95_nbell_1_bscl_0.9_moffy_9.5625_moffx_9.5625_h_obs_nt_1501_dt_2ms_shot_2.dir | True | rel_l2 | 8.671668e-08 | 4.768372e-07 | 8.322430e-07 | 5.695365e-08 |
