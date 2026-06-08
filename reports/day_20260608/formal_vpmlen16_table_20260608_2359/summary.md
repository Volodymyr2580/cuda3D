# Formal Same-Session Speed Table After V-PML Len16

## Context

- report dir: `reports/day_20260608/formal_vpmlen16_table_20260608_2359`
- case: `perf_1gpu_6shots`
- rounds: `3`
- baseline in each round: `zmem`
- current best alias: `current_best_v_pml_len16`

## Summary Vs Zmem

| candidate | mean WP speedup | mean Gradient speedup | mean elapsed speedup | mean WP | all compare pass | max rel L2 | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `directfill` | `1.101172x` | `1.100029x` | `1.081287x` | `2.207205s` | `True` | `0.000000e+00` | `0.000000e+00` |
| `pressure_len16` | `1.194495x` | `1.179869x` | `1.098568x` | `2.034743s` | `True` | `6.384336e-07` | `4.768372e-06` |
| `current_best_v_pml_len16` | `1.222023x` | `1.206588x` | `1.118261x` | `1.988905s` | `True` | `6.384336e-07` | `4.768372e-06` |

## Rounds

| round | candidate | zmem WP | candidate WP | WP speedup | zmem Gradient | candidate Gradient | Gradient speedup | compare | max rel L2 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `directfill` | `2.443817` | `2.223808` | `1.098933x` | `2.559809` | `2.325493` | `1.100760x` | `True` | `0.000000e+00` |
| 1 | `pressure_len16` | `2.443817` | `2.036063` | `1.200266x` | `2.559809` | `2.158922` | `1.185689x` | `True` | `6.384336e-07` |
| 1 | `current_best_v_pml_len16` | `2.443817` | `1.990542` | `1.227714x` | `2.559809` | `2.110839` | `1.212697x` | `True` | `6.384336e-07` |
| 2 | `directfill` | `2.424403` | `2.199291` | `1.102357x` | `2.541517` | `2.311426` | `1.099545x` | `True` | `0.000000e+00` |
| 2 | `pressure_len16` | `2.424403` | `2.033737` | `1.192093x` | `2.541517` | `2.160031` | `1.176611x` | `True` | `6.384336e-07` |
| 2 | `current_best_v_pml_len16` | `2.424403` | `1.988189` | `1.219403x` | `2.541517` | `2.114358` | `1.202028x` | `True` | `6.384336e-07` |
| 3 | `directfill` | `2.423261` | `2.198516` | `1.102226x` | `2.543350` | `2.312593` | `1.099783x` | `True` | `0.000000e+00` |
| 3 | `pressure_len16` | `2.423261` | `2.034429` | `1.191126x` | `2.543350` | `2.160310` | `1.177308x` | `True` | `6.384336e-07` |
| 3 | `current_best_v_pml_len16` | `2.423261` | `1.987985` | `1.218953x` | `2.543350` | `2.110594` | `1.205040x` | `True` | `6.384336e-07` |

## Flags

### zmem

```text
-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2
```

### directfill

```text
-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2 -DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL -DCUDA3D_CPML_VMEM_DISABLE_MPI -DCUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

### pressure_len16

```text
-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2 -DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL -DCUDA3D_CPML_VMEM_DISABLE_MPI -DCUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE -DCUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
```

### current_best_v_pml_len16

```text
-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2 -DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL -DCUDA3D_CPML_VMEM_DISABLE_MPI -DCUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE -DCUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK -DCUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
```
