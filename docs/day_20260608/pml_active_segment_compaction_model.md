# PML Active Segment Compaction Model

## Context

- case_dir: `E:\cuda3D\benchmarks\cases\perf_1gpu_6shots`
- tile block z/x/y: `32/4/2`
- npml/core_pml_margin: `12/4`
- p_pml sampled-main share: `54.04%`

## Current Lane Shape

- kept tiles: `113840`
- current launched lanes: `29143040`
- active lanes after core return: `19118944`
- current lane efficiency: `65.60%`
- active line slots: `893204`
- average active lanes per active line: `21.405`

## Active Z-Line Length Histogram

| active z length | line slots | active lanes |
| ---: | ---: | ---: |
| 16 | 542100 | 8673600 |
| 23 | 87776 | 2018848 |
| 32 | 263328 | 8426496 |

## Candidate Lane Ceilings

| candidate | lanes | lane efficiency | lane reduction | p_pml lane speedup ceiling | sampled-main ceiling | gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `active_line_list_8warp` | 28582656 | 66.89% | 1.92% | 1.020x | 1.011x | `reject` |
| `exact_active_point_list` | 19119104 | 100.00% | 34.40% | 1.524x | 1.228x | `design_only` |
| `pack_len16_halfwarp_plus_fullwarps` | 19908928 | 96.03% | 31.69% | 1.464x | 1.207x | `design_only` |

## Descriptor Traffic

- point list with one uint32 per active point: `72.933 MiB/step aggregate-shots`
- line list with one uint64 per active line: `6.815 MiB/step aggregate-shots`

## Gate

- active-line list: `reject`
- length-16 half-warp packing: `design_only`
- exact active-point list: `design_only`
- reason: Simple active-line compaction only removes empty lines and has about a 1.0196x p_pml lane ceiling. The meaningful ceiling comes from packing length-16 z-face/margin lines or exact active points, but those designs must preserve the accepted z-cache dataflow before CUDA implementation.

## Design Boundary

A CUDA prototype is not opened by the simple active-line list, because it only removes about `1.92%` of launched lanes.

A future prototype may be opened only if it preserves the accepted pressure z-cache dataflow while packing the length-16 z-face/margin lines or exact active points. This is a lane-utilization design, not a repeat of the rejected z-face direct-derivative/fusion route.
