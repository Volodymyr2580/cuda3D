# PML Compact-State Audit

## Case

- case_dir: `/work/wenzhe/cuda3D/benchmarks/cases/perf_1gpu_6shots`
- logical ny/nx/nz: `384/384/95`
- domain n3/n2/n1 without radius: `408/408/119`
- padded wavefield nypad/nxpad/nzpad: `416/416/127`
- npml/radius/CorePmlMargin: `12/4/4`
- nt/shots/receivers_per_shot: `1501/6/441`
- PML kernel work points: `7509504`

## State Allocation

| array | axis | elements | MiB | formula | owner |
| --- | --- | ---: | ---: | --- | --- |
| `memory_dy` | y | 1165248 | 4.445 | `2*npml*n2*n1` | v_pml |
| `memory_dx` | x | 1165248 | 4.445 | `n3*2*npml*n1` | v_pml |
| `memory_dz` | z | 3995136 | 15.240 | `n3*n2*2*npml` | p_pml zmem old |
| `memory_dyy` | y | 1165248 | 4.445 | `2*npml*n2*n1` | p_pml |
| `memory_dxx` | x | 1165248 | 4.445 | `n3*2*npml*n1` | p_pml |
| `memory_dzz` | z | 3995136 | 15.240 | `n3*n2*2*npml` | p_pml |
| `memory_dz_next` | z | 3995136 | 15.240 | `n3*n2*2*npml` | p_pml zmem next |
| `memory_dy_next` | y | 1165248 | 4.445 | `2*npml*n2*n1` | v_pml next |
| `memory_dx_next` | x | 1165248 | 4.445 | `n3*2*npml*n1` | v_pml next |

- total CPML state footprint for `cpml_dbuf`: `72.391 MiB`
- wavefield/cw2 floor footprint, six padded arrays: `503.039 MiB`
- state as share of six wavefield arrays: `14.39%`

Important read: the current code already stores CPML memory as axis slabs, not as full padded-domain arrays.

## PML Face/Edge/Corner Distribution

| region | points |
| --- | ---: |
| z_only | 3538944 |
| x_only | 875520 |
| y_only | 875520 |
| xy_edge | 54720 |
| xz_edge | 221184 |
| yz_edge | 221184 |
| xyz_corner | 13824 |

## Z-Face Compact Coverage

- `memory_dz` elements: `3995136`
- true z-face elements: `3538944` (88.58% of `memory_dz`)
- safe z-face elements with CorePmlMargin: `3393024` (84.93% of `memory_dz`)
- residual z edge/corner elements that still need state: `602112`

A safe compact z-face layout can be affine-indexed without division/mod, but it does not remove the residual z edge/corner state.

## Static Traffic Floor

- mandatory CPML state update traffic floor: `96.521 MiB/step`
- zmem `memory_dz` old reads from recompute path: `111.762 MiB/step`
- zmem `memory_dz_next` writes: `15.240 MiB/step`
- pressure PML vx/vy load estimate: `458.344 MiB/step`
- pressure PML p0/p1/cw2/store floor: `114.586 MiB/step`

## Gate

- verdict: `stop_compact_state`
- estimated compact-state WP speedup ceiling: `1.005x`
- reason: Current CPML memory is already stored as y/x/z slabs.  Safe z-face compacting does not remove edge/corner state and has no >=5% static WP ceiling without new profiler evidence that CPML state layout dominates stalls.
