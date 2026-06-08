# CPML VMEM Double-Buffer Revalidation

Date: 2026-06-08

Platform: `/work/wenzhe/cuda3D` on RTX 5090, CUDA 13.0, Intel MPI 2021.18.

## Goal

Revalidate `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` against the stable `zmem_reference` flags before starting any larger compact-state or temporal-pipeline work.

The key test discipline today was to rebuild the correct binary before each A/B run. This avoids the common failure mode where a run tag says `zmem` but the executable is still the previous CPML build.

## Flags

Baseline zmem:

```bash
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DCUDA3D_PML_ZMEM_IN_P
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
```

CPML double-buffer:

```bash
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DCUDA3D_PML_ZMEM_IN_P
-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
-DCUDA3D_CPML_VMEM_DISABLE_MPI
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
```

## Correctness

- zmem phase0 smoke: `benchmarks/runs/smoke_1gpu_day_zmem_phase0_smoke_20260608_094936`
- zmem phase0 correctness: `benchmarks/runs/correctness_day_zmem_phase0_correctness_20260608_094940`
- CPML smoke: `benchmarks/runs/smoke_1gpu_day_cpml_dbuf_phase1_smoke_20260608_095127`
- CPML correctness: `benchmarks/runs/correctness_day_cpml_dbuf_phase1_correctness_20260608_095130`
- CPML correctness vs zmem: pass, max rel L2 `0.000000000e+00`, max abs `0.000000000e+00`

## Perf A/B

Case: `perf_1gpu_6shots`, 6 shots, 441 receivers/shot, `nt = 1501`.

| round | zmem WP (s) | CPML WP (s) | WP speedup | zmem Gradient (s) | CPML Gradient (s) | Gradient speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a | 2.456481 | 2.379958 | 1.032153x | 2.564714 | 2.494277 | 1.028239x |
| b | 2.422000 | 2.353140 | 1.029263x | 2.544060 | 2.475148 | 1.027842x |
| c | 2.426137 | 2.342810 | 1.035567x | 2.542697 | 2.470961 | 1.029032x |

All-mean:

- zmem WP: `2.434873s`
- CPML WP: `2.358636s`
- WP speedup: `1.032329x`
- zmem Gradient: `2.550490s`
- CPML Gradient: `2.480129s`
- Gradient speedup: `1.028370x`

All three perf output comparisons passed with max rel L2 `0`.

## Gate Result

`CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` passes the Phase 1 gate:

- Correctness pass.
- Average WP speedup is above `1.025x`.
- Average Gradient speedup is above `1.025x`.
- Every single A/B round is above `1.015x` for both WP and Gradient.

Decision: keep CPML double-buffer as the active ownership scaffold for later wave-step restructuring, but do not treat it as a 2x breakthrough by itself.
