# Phase2 Fused Z-Face Forensics

## Verdict

`CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY` in the direct global `p1` x/y second-derivative form remains stopped.

The failure mode is now visible in profiler data:

- direct inline fusion makes `cuda_fd3d_p_pml_tile_ns` slower, from about `188.856us` to `248.200us` per sampled launch;
- that is a `1.314x` per-launch slowdown for the pressure PML tile kernel;
- the direct path increases compute/memory-pipe pressure instead of removing enough memory latency;
- the earlier perf6 slowdown is therefore structural, not noise.

Continue only with the shared-tile VP design described in:

```text
docs/wavestep_v2/pml_zface_shared_vp_design.md
docs/wavestep_v2/pml_zface_shared_tile_budget.md
```

## Profiles

Short Nsight Compute profiles were collected on the RTX 5090 server with `profile_1gpu`, `--launch-skip 20`, `--launch-count 8`, and PML kernel filter:

```text
regex:.*cuda_fd3d_.*pml.*
```

Raw CSV files:

```text
benchmarks/profiles/wavestep_v2_night_20260608/zmem_pml_short.csv
benchmarks/profiles/wavestep_v2_night_20260608/cpml_vmem_double_buffer_all_pml_short.csv
benchmarks/profiles/wavestep_v2_night_20260608/direct_inline_fused_zface_pml_short.csv
```

Summary files:

```text
docs/wavestep_v2/ncu_forensics_summary.md
reports/wavestep_v2_night_20260608/ncu_forensics_summary.json
```

Build hashes:

```text
zmem ncu binary:
22c7b3d8837429f28b97f72ff8554aef302d930255f6e45565ff9139aac6a4ef  bin/cuda_3D_FM

cpml vmem ncu binary:
fb2d58de9f42dbe5768e00134214faf6d42d6140e36aa9ecf90fcae3c25394fb  bin/cuda_3D_FM

direct inline fused zface ncu binary:
1cbee97062277f8b5ee1691571d698e72f76649a2039c9ae6a8852202dbb7cb2  bin/cuda_3D_FM
```

The separate direct fused kernel was not re-profiled in this pass because the current `ee3a6b0` launch path no longer invokes that separate kernel. Its historical perf6 result remains:

- correctness pass;
- mean WP `2.660077s`;
- speed ratio vs same-session zmem `0.915184x`.

## Key Metrics

`cuda_fd3d_p_pml_tile_ns`:

| metric | zmem | cpml_vmem | direct_inline |
| --- | ---: | ---: | ---: |
| duration | `188.856us` | `189.896us` | `248.200us` |
| duration ratio vs zmem | `1.000x` | `1.006x` | `1.314x` |
| SOL compute | `55.745%` | `55.788%` | `69.227%` |
| SOL memory | `55.745%` | `55.788%` | `69.227%` |
| SOL DRAM | `40.195%` | `40.125%` | `28.700%` |
| mem pipes busy | `55.745%` | `55.788%` | `69.227%` |
| L1/TEX hit rate | `86.005%` | `86.005%` | `85.890%` |
| L2 hit rate | `56.782%` | `56.693%` | `67.088%` |
| no eligible | `47.483%` | `46.968%` | `39.163%` |
| eligible warps/scheduler | `1.140` | `1.150` | `1.620` |
| achieved occupancy | `72.797%` | `72.757%` | `74.903%` |

`cuda_fd3d_v_pml_tile_ns`:

| metric | zmem | cpml_vmem | direct_inline |
| --- | ---: | ---: | ---: |
| duration | `71.872us` | `65.528us` | `63.744us` |
| duration ratio vs zmem | `1.000x` | `0.912x` | `0.887x` |
| SOL compute | `44.237%` | `49.498%` | `55.780%` |
| SOL memory | `55.055%` | `60.320%` | `53.477%` |
| SOL DRAM | `44.915%` | `49.385%` | `40.175%` |
| no eligible | `52.320%` | `46.418%` | `40.455%` |
| eligible warps/scheduler | `1.230` | `1.538` | `1.853` |
| achieved occupancy | `82.373%` | `82.875%` | `83.700%` |

## Interpretation

CPML double-buffer mostly improves the velocity PML kernel and leaves the pressure PML kernel essentially unchanged:

- `v_pml_tile` duration improves from `71.872us` to `65.528us`;
- `p_pml_tile` stays near `189us`;
- this matches the perf6 result where CPML double-buffer gives about `1.0335x` WP speedup.

Direct inline z-face fusion improves or preserves the velocity-side profile but makes the pressure-side kernel much heavier:

- `p_pml_tile` duration rises by about `31.4%`;
- SOL compute and memory-pipe utilization rise to about `69%`;
- NCU labels the direct inline pressure kernel as compute/memory balanced, meaning a simple memory-only fix is no longer enough;
- the direct p1 second-derivative path is therefore moving work into the wrong kernel and increasing the critical pressure update cost.

The zmem pressure kernel still shows L1TEX scoreboard and issue-slot underutilization in NCU rules, so shared-memory locality remains a plausible target. The important difference is that a valid retry must reduce global velocity round-trip traffic without converting the pressure update into a large direct second-derivative recompute kernel.

## Decision

Do not continue:

```text
CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY
```

in any direct global `p1` x/y second-derivative form.

Allowed next step:

```text
CUDA3D_PML_ZFACE_SHARED_VP_DEBUG
```

as a debug-only, default-off prototype based on S2 from the tile budget:

```text
output tile = 12 x 16 x 12
shared p tile = 26 x 30 x 26
shared bytes = 81,120
threads = 256
estimated shared p loads/output = 8.802
```

Stop that prototype immediately if `perf_1gpu_6shots repeat` is not at least `>=5%` faster than the same-session zmem baseline.
