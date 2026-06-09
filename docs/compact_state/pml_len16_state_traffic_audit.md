# PML Len16 Compact-State Traffic Audit

## Case

- case_dir: `benchmarks/cases/perf_1gpu_6shots`
- logical grid: `384 x 384 x 95`
- nt: `1501`
- shots in manifest/log: `6` / `6`
- npml: `12`

## Full CPML State Arrays

| array | formula | elements | MiB |
| --- | --- | ---: | ---: |
| `memory_dy` | `2*npml*n2*n1` | `1165248` | `4.445` |
| `memory_dx` | `n3*2*npml*n1` | `1165248` | `4.445` |
| `memory_dz` | `n3*n2*2*npml` | `3995136` | `15.240` |
| `memory_dy_next` | `2*npml*n2*n1` | `1165248` | `4.445` |
| `memory_dx_next` | `n3*2*npml*n1` | `1165248` | `4.445` |
| `memory_dz_next` | `n3*n2*2*npml` | `3995136` | `15.240` |
| `memory_dyy` | `2*npml*n2*n1` | `1165248` | `4.445` |
| `memory_dxx` | `n3*2*npml*n1` | `1165248` | `4.445` |
| `memory_dzz` | `n3*n2*2*npml` | `3995136` | `15.240` |

## Len16 Ownership From Runtime Log

- pressure len16 tiles: `67392`
- pressure residual tiles: `46448`
- velocity len16 tiles: `62400`
- velocity residual tiles: `56432`
- pressure len16 active points: `8626176`
- velocity len16 active points: `7987200`
- pressure len16 compact lines: `539136`

## Compact Pressure-State Budget

The accepted velocity len16 kernel currently writes `vx/vy` derivative
fields and does not update `memory_dx/memory_dy/memory_dz` state.
The compact-state opportunity is therefore anchored on pressure
len16 state: `memory_dzz` plus the z-recompute `memory_dz` old/next
window.

- compact pressure-state bytes: `127.512 MiB`
- full pressure-related state bytes x shots: `274.324 MiB`
- compact/full ratio: `0.464821`

## NCU-Anchored Ceiling

- sampled main kernel us: `266.710`
- p_len16 us: `67.04`
- p_len16 share of sampled main: `0.2514`
- assumed removable p_len16 state fraction: `0.35`
- p_len16 speedup ceiling: `1.5385x`
- whole-job sampled-main speedup ceiling: `1.0965x`

## Gate Decision

- decision: `allow_commit_prototype_after_design`

Estimated ceiling is at least `5%`; design and mirror must still pass before commit path.
