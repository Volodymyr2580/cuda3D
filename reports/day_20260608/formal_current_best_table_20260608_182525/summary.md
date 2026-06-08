# Formal Same-Session Speed Table

## Context

- report dir: `reports/day_20260608/formal_current_best_table_20260608_182525`
- case: `perf_1gpu_6shots`
- rounds: `3`
- baseline in each round: `zmem`
- current best alias: `len16_current_best`

## Summary Vs Zmem

| candidate | mean WP speedup | mean Gradient speedup | mean elapsed speedup | mean WP | all compare pass | max rel L2 | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `directfill` | `1.099957x` | `1.097977x` | `1.105408x` | `2.203315s` | `True` | `0.000000e+00` | `0.000000e+00` |
| `len16_current_best` | `1.192835x` | `1.179213x` | `1.156108x` | `2.031753s` | `True` | `6.384336e-07` | `4.768372e-06` |

## Rounds

| round | candidate | zmem WP | candidate WP | WP speedup | zmem Gradient | candidate Gradient | Gradient speedup | compare | max rel L2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `directfill` | `2.438352` | `2.216956` | `1.099865x` | `2.551901` | `2.322793` | `1.098635x` | `True` | `0.000000e+00` |
| 1 | `len16_current_best` | `2.438352` | `2.032625` | `1.199607x` | `2.551901` | `2.154372` | `1.184522x` | `True` | `6.384336e-07` |
| 2 | `directfill` | `2.412607` | `2.196933` | `1.098170x` | `2.536826` | `2.312208` | `1.097144x` | `True` | `0.000000e+00` |
| 2 | `len16_current_best` | `2.412607` | `2.030254` | `1.188328x` | `2.536826` | `2.155571` | `1.176870x` | `True` | `6.384336e-07` |
| 3 | `directfill` | `2.419689` | `2.196055` | `1.101834x` | `2.538063` | `2.311211` | `1.098153x` | `True` | `0.000000e+00` |
| 3 | `len16_current_best` | `2.419689` | `2.032380` | `1.190569x` | `2.538063` | `2.157764` | `1.176247x` | `True` | `6.384336e-07` |

## Flags

### zmem

```text
-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2
```

### directfill

```text
-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2 -DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL -DCUDA3D_CPML_VMEM_DISABLE_MPI -DCUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

### len16_current_best

```text
-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2 -DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL -DCUDA3D_CPML_VMEM_DISABLE_MPI -DCUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE -DCUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
```
