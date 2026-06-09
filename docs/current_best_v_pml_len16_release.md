# current_best_v_pml_len16 Release

## Status

This document freezes the current best RTX 5090 single-GPU candidate for
handoff and future comparison.

It is **not** a `1.5x` speed-threshold archive.  It is a current-best release
record for the exact-FP32 single-GPU line.

```text
tag                           current-best-v-pml-len16-rtx5090-20260609
tag target commit             f637ba115d52852b493867ab4a957113a01142a5
branch                        exp/day-20260608-cpml-compact-temporal
candidate                     current_best_v_pml_len16
platform                      RTX 5090, CUDA 13, sm_120
case                          perf_1gpu_6shots
rounds                        3
```

## Flags

```text
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DCUDA3D_PML_ZMEM_IN_P
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
-DCUDA3D_CPML_VMEM_DISABLE_MPI
-DCUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
-DCUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
-DCUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
```

## Formal 3-Round Table

Source:

```text
reports/day_20260608/formal_vpmlen16_table_20260608_2359/summary.json
```

| round | elapsed s | Gradient s | WP s | binary sha256 |
| ---: | ---: | ---: | ---: | --- |
| 1 | `3.160000` | `2.110839` | `1.990542` | `aa58035a8a084bfd34fc2336bfbbb10fb3586ba9352c75109e49fd2be7909278` |
| 2 | `2.970000` | `2.114358` | `1.988189` | `dd085284245186517599db50cb98f19affef2855ef8ed17cc13a54273c64162b` |
| 3 | `2.920000` | `2.110594` | `1.987985` | `881e9e35f0291bad9b63da90bf04f18d0ad07550325a8b808475b2a1264940b9` |

Mean result:

```text
mean elapsed                  3.016667 s
mean Gradient                 2.111930 s
mean WP                       1.988905 s
elapsed speedup vs zmem        1.118261x
Gradient speedup vs zmem       1.206588x
WP speedup vs zmem             1.222023x
```

Correctness:

```text
all compare pass              true
max rel L2                    6.384336e-07
max abs                       4.768372e-06
NaN/Inf                       none reported
```

## Accepted Stack

- `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`
- `CUDA3D_CPML_VMEM_DISABLE_MPI`
- `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
- `CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK`
- `CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK`

## Estimated Total Speedup Context

The project history contains an earlier approximate `1.8x` anchor before the
RTX 5090 `zmem_reference` phase, and `zmem_reference` was measured at about
`1.049300x` WP over `current_best_reference`.

If that earlier anchor is accepted as comparable, the estimated total WP
speedup vs the very first implementation is:

```text
1.8 * 1.049300 * 1.222023 = 2.308x
```

This is only an estimate until the true original source/commit is rebuilt on
the same RTX 5090 platform and compared directly.

## Milestone Status

```text
1.5x archive                  false
additional WP speedup to 1.5x 1.227472x
```

Do not copy this package into `archives/speedups/` as a speed-threshold archive.
