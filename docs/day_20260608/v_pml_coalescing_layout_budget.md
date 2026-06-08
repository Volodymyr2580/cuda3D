# V-PML Coalescing/Layout Budget

## Context

- case: `benchmarks/cases/perf_1gpu_6shots`
- npml/core margin: `12/4`
- NCU anchor profile: `len16`
- sampled main: `297.248us`
- v-PML duration/share: `65.248us` / `21.95%`
- v-kernel speedup required for 5% sampled-main gain: `1.2770x`

The current CUDA mapping uses `threadIdx.x` as the z index.  With the accepted `32x4x2` tile, each warp is one contiguous z-line at fixed x/y.  That is already the favorable coalescing shape for the `p1`, `mem_dx`, and `mem_dy` paths.

## Candidate Shapes

| shape | block z/x/y | tiles | launched lanes ratio | active lane eff | component density | warp split factor | p1 sector-unit excess | p1 sector-unit ratio | optimistic v speedup | sampled-main ceiling |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_32x4x2` | `32/4/2` | 118832 | 1.0000 | 73.70% | 130.80% | 1.00 avg / 1 max | 0.00% | 1.0000 | 1.0000x | 1.0000x |
| `z64_x2_y2` | `64/2/2` | 180268 | 1.5170 | 48.59% | 86.22% | 1.00 avg / 1 max | 0.00% | 1.0000 | 1.0000x | 1.0000x |
| `z32_x8_y1` | `32/8/1` | 122330 | 1.0294 | 71.60% | 127.06% | 1.00 avg / 1 max | 0.00% | 1.0000 | 1.0000x | 1.0000x |
| `z32_x2_y4` | `32/2/4` | 118616 | 0.9982 | 73.84% | 131.04% | 1.00 avg / 1 max | 0.00% | 1.0000 | 1.0018x | 1.0004x |
| `z16_x8_y2` | `16/8/2` | 123160 | 1.0364 | 71.11% | 126.21% | 2.00 avg / 2 max | 100.00% | 2.0000 | 1.0000x | 1.0000x |
| `z16_x4_y4` | `16/4/4` | 119584 | 1.0063 | 73.24% | 129.98% | 2.00 avg / 2 max | 100.00% | 2.0000 | 1.0000x | 1.0000x |
| `z16_x16_y1` | `16/16/1` | 128284 | 1.0795 | 68.27% | 121.17% | 2.00 avg / 2 max | 100.00% | 2.0000 | 1.0000x | 1.0000x |
| `z8_x8_y4` | `8/8/4` | 104925 | 0.8830 | 83.47% | 148.14% | 4.00 avg / 4 max | 300.00% | 4.0000 | 1.1325x | 1.0264x |
| `z8_x16_y2` | `8/16/2` | 110520 | 0.9301 | 79.25% | 140.64% | 4.00 avg / 4 max | 300.00% | 4.0000 | 1.0752x | 1.0156x |

## Interpretation

- `32x4x2` is not an arbitrary local optimum: it gives one contiguous z segment per warp.
- `component density` counts `vx + vy` work per launched lane, so it can be above 100%.
- Shapes with `z < 32` split a warp across multiple x/y lines.  Some remove launched lanes, but the most optimistic sampled-main ceiling is still only about 2.6% before implementation overhead.
- Larger-z or x/y-rebalanced shapes do not help this perf case; they pack more inactive interior/margin lanes into each velocity tile or keep the same component work.
- The earlier rejected vx/vy split remains rejected because it nearly doubles component work under this geometry.

## Gate

- decision: `reject_tile_layout_cuda_prototype`
- best shape: `z8_x8_y4`
- best optimistic v-kernel speedup ceiling: `1.1325x`
- best optimistic sampled-main ceiling: `1.0264x`
- reason: The current 32x4x2 layout already maps each warp to one contiguous z-line segment. The best reasoned v-only shape does not reach the >=5% sampled-main ceiling, and the real implementation would also need separate velocity tile-list plumbing plus pressure-path compatibility.

Allowed next directions:

- profile current-best v_pml source hotlines if source-level evidence is stale
- design a memory-ownership change that reduces global vx/vy round trip without doubling component work
- revisit v-PML only if a new model shows >=5% perf_1gpu_6shots repeat ceiling after tile-list overhead

Do not continue:

- random PmlTileBlockSize sweep
- current-geometry vx/vy component split
- v-only tile-layout CUDA prototype below the 5% gate
