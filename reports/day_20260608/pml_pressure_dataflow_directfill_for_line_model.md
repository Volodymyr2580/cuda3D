# Pressure PML Dataflow Audit

## Case

- case_dir: `E:\cuda3D\benchmarks\cases\perf_1gpu_6shots`
- logical ny/nx/nz: `384/384/95`
- shots/receivers_per_shot: `6/441`
- npml/xpad: `12/0.5`
- PML tile block z/x/y: `32/4/2`

## Tile And Thread Shape

- kept pressure-PML tiles: `113840` / full grid `181232`
- active thread efficiency after core return: `65.60%`
- valid-domain thread efficiency before core return: `87.32%`
- boundary padding threads: `3695098`
- returned-core threads inside kept tiles: `6328998`

### Tile Masks

| mask | tiles | active points | shell points |
| --- | ---: | ---: | ---: |
| `shell` | 5184 | 1232256 | 1232256 |
| `x` | 8640 | 2045952 | 55296 |
| `xy` | 1000 | 232064 | 128 |
| `y` | 8400 | 2107392 | 43008 |
| `z` | 72576 | 9732570 | 2765274 |
| `zx` | 8640 | 1758240 | 26784 |
| `zxy` | 1000 | 199430 | 62 |
| `zy` | 8400 | 1811040 | 20832 |

## Point Categories

| category | points | share of active |
| --- | ---: | ---: |
| `shell` | 4143640 | 21.67% |
| `z_only` | 7004208 | 36.63% |
| `x_only` | 2961720 | 15.49% |
| `y_only` | 3073440 | 16.08% |
| `xy_edge` | 328320 | 1.72% |
| `xz_edge` | 748224 | 3.91% |
| `yz_edge` | 776448 | 4.06% |
| `xyz_corner` | 82944 | 0.43% |
| `true_pml_total` | 14975304 | 78.33% |

## Z-Recompute Reuse Budget

- current `recompute_vz_after_update` calls: `152951552`
- shared z-line cache calls estimate: `29093740`
- call reduction estimate: `80.98%`
- current p1 load estimate inside z recompute: `4667.711 MiB/step aggregate-shots`
- shared-cache p1 load estimate: `887.870 MiB/step aggregate-shots`

## NCU Link And Model

- p_pml duration: `158438.4` ns
- p_core duration: `75942.4` ns
- v_pml duration: `58793.6` ns
- p_pml sampled-main share: `54.04%`
- modeled p_pml speedup if shared z recompute succeeds: `1.573x`
- modeled sampled-main speedup: `1.245x`

## Gate

- verdict: `open_p_pml_z_recompute_line_cache_prototype`
- allowed prototype: `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
- reason: Shared z-line recompute has a >=5% sampled-main model ceiling and is not the forbidden z-face/shared-VP route.
- discipline: this gate does not reopen tile-mask fastpath, z-face specialize/fusion, or RECOMPUTE_X/Y/XYZ.

## Shot Table

| shot | domain y/x | active points | shell share | z-recompute reduction | active thread efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 216x216 | 3090432 | 21.63% | 81.03% | 67.13% |
| 1 | 216x241 | 3352032 | 21.75% | 80.89% | 65.16% |
| 2 | 216x217 | 3100896 | 21.63% | 81.03% | 65.60% |
| 3 | 217x216 | 3100896 | 21.63% | 81.03% | 66.47% |
| 4 | 217x241 | 3363296 | 21.75% | 80.89% | 64.52% |
| 5 | 217x217 | 3111392 | 21.64% | 81.02% | 64.97% |
