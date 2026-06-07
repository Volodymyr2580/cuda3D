# CPML VMEM Double Buffer All Result

## Verdict

Gate: `continue_as_scaffold`

`CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` is still correctness-clean and mildly faster than the same-session `zmem_reference` baseline on the RTX 5090 server. It does not meet the `>=5%` final candidate gate by itself, but it remains the required ownership scaffold for any later shared-tile VP fusion.

## Build

```bash
NVFLAGS="-O3 -arch=sm_120 --use_fast_math \
-DCUDA3D_PML_RECOMPUTE_Z \
-DCUDA3D_PML_TILE_LIST \
-DCUDA3D_PML_ZMEM_IN_P \
-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL \
-DCUDA3D_CPML_VMEM_DISABLE_MPI \
-DPmlTileBlockSize1=32 \
-DPmlTileBlockSize2=4 \
-DPmlTileBlockSize3=2"
```

Binary SHA256:

```text
11b6eb87700d698f3c5786b7bd57b3a7cddb34b1ccbe2bb85acd25db4324acfa  bin/cuda_3D_FM
```

Build log:

```text
benchmarks/build_logs/cpml_vmem_double_buffer_all_night_20260608_014216.log
```

## Run Paths

Baseline:

```text
benchmarks/runs/smoke_1gpu_zmem_reference_night_smoke_20260608_013204
benchmarks/runs/correctness_zmem_reference_night_correctness_20260608_013207
benchmarks/runs/perf_1gpu_6shots_zmem_reference_night_perf6_a_20260608_013209
benchmarks/runs/perf_1gpu_6shots_zmem_reference_night_perf6_b_20260608_013215
```

Candidate:

```text
/work/wenzhe/cuda3D/benchmarks/runs/smoke_1gpu_cpml_vmem_double_buffer_all_night_smoke_20260608_014223
/work/wenzhe/cuda3D/benchmarks/runs/correctness_cpml_vmem_double_buffer_all_night_correctness_20260608_014226
/work/wenzhe/cuda3D/benchmarks/runs/perf_1gpu_6shots_cpml_vmem_double_buffer_all_night_perf6_a_20260608_014229
/work/wenzhe/cuda3D/benchmarks/runs/perf_1gpu_6shots_cpml_vmem_double_buffer_all_night_perf6_b_20260608_014234
```

## Performance

Same-session zmem baseline:

| run | WP computing time | Gradient TIME all | wall |
| --- | ---: | ---: | ---: |
| zmem perf6 a | `2.447898s` | `2.559258s` | `0:04.73` |
| zmem perf6 b | `2.449256s` | `2.562289s` | `0:04.84` |
| zmem mean | `2.448577s` | `2.560774s` | - |

CPML double-buffer candidate:

| run | WP computing time | Gradient TIME all | wall |
| --- | ---: | ---: | ---: |
| cpml perf6 a | `2.367973s` | `2.483764s` | `0:04.76` |
| cpml perf6 b | `2.370387s` | `2.488680s` | `0:04.73` |
| cpml mean | `2.369180s` | `2.486222s` | - |

Speedup vs zmem mean:

- WP: `1.033512x`
- Gradient: `1.029986x`

## Correctness

Output comparisons all pass with `rel_l2 <= 1e-5`:

```text
reports/wavestep_v2_night_20260608/cpml_smoke_vs_zmem/comparison.md
reports/wavestep_v2_night_20260608/cpml_correctness_vs_zmem/comparison.md
reports/wavestep_v2_night_20260608/cpml_perf6_repeat_vs_zmem/comparison.md
```

The local summary copies are:

```text
reports/wavestep_v2_night_20260608/cpml_smoke_vs_zmem_comparison.md
reports/wavestep_v2_night_20260608/cpml_correctness_vs_zmem_comparison.md
reports/wavestep_v2_night_20260608/cpml_perf6_repeat_vs_zmem_comparison.md
```

All compared files have `rel_l2 = 0`.

## Decision

Do not promote CPML double-buffer alone as a final speedup candidate. Keep it macro-gated and default-off.

Use it as the mandatory base for any z-face shared-tile VP prototype, because that prototype needs explicit current/next CPML velocity-memory ownership.
