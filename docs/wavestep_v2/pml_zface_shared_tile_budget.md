# PML Z-Face Shared-Tile VP Budget

## Case Geometry

- case_dir: `/work/wenzhe/cuda3D/benchmarks/cases/perf_1gpu_6shots`
- logical model: ny/nx/nz = `384/384/95`
- padded domain estimate: n3/n2/n1 = `408/408/119`
- npml/core_margin: `12/4`
- nt/shots/receivers_per_shot: `1501/6/441`
- estimated total points: `19809216`
- estimated core points: `12299712`
- estimated PML points: `7509504`
- estimated pure z-face PML points: `3393024`
- z-face share of estimated PML pressure work: `45.18%`

## Tile Candidates

| candidate | output z/x/y | threads | shared z/x/y | shared bytes | outputs | shared p loads/output | p-load reduction vs direct xy est | blocks/SM by shared | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| S1 | 8x16x16 | 256 | 22x30x30 | 79200 | 2048 | 9.668 | 67.77% | 1 | pass: debug-prototype budget candidate |
| S2 | 12x16x12 | 256 | 26x30x26 | 81120 | 2304 | 8.802 | 70.66% | 1 | pass: debug-prototype budget candidate |
| S3 | 8x24x12 | 256 | 22x38x26 | 86944 | 2304 | 9.434 | 68.55% | 1 | pass: debug-prototype budget candidate |
| S4 | 12x12x12 | 256 | 26x26x26 | 70304 | 1728 | 10.171 | 66.10% | 1 | pass: debug-prototype budget candidate |

## Assumptions

```json
{
  "composed_pressure_radius": 7,
  "direct_xy_second_derivative_loads_per_output_est": 30,
  "halo": 7,
  "halo_reason": "composed p-current footprint for local velocity-gradient plus pressure-divergence path",
  "max_block_smem_bytes": 99000,
  "saved_velocity_global_bytes_per_output_est": "2 velocity stores + 16 velocity stencil reads = 72 bytes/output",
  "smem_per_sm_bytes": 131072,
  "source_receiver_exclusion": "unknown_binary_nav",
  "velocity_radius": 4
}
```

## Gate Read

- All candidates are design-only until Nsight Compute evidence shows the z-face path is still memory-traffic limited.
- Passing this budget only means the shared pressure tile fits under the configured opt-in shared-memory limit.
- The estimate intentionally treats source/receiver exclusion as unknown because `.nav` is a binary float file; a real kernel must either reject those tiles at runtime or prove they cannot appear in the fused z-face region.