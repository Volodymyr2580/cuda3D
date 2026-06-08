# V-PML Active Segment Packing Model

## Context

- case: `E:\cuda3D\benchmarks\cases\perf_1gpu_6shots`
- tile block z/x/y: `32/4/2`
- NCU anchor profile: `len16`
- sampled main: `297.248us`
- v-PML duration/share: `65.248us` / `21.95%`
- v-kernel speedup required for 5% sampled-main: `1.2770x`

## Current Lane Shape

- tiles: `118832`
- current launched lanes: `30420992`
- in-domain line lanes: `26725894`
- true vx/vy active-any lanes: `20646925`
- vx lanes: `19880803`
- vy lanes: `19910644`
- component lanes vx+vy: `39791447`
- active-any lane efficiency vs launched: `67.87%`
- component density vs launched: `130.80%`
- z-line slots: `936104`
- empty z-line slots: `7774`
- length-16 z-line slots: `506974`
- whole length-16 tiles: `62400`

## Active-Any Z-Line Histogram

| active z length | line slots |
| ---: | ---: |
| 0 | 7774 |
| 16 | 506974 |
| 23 | 105339 |
| 32 | 316017 |

## Component-Lane Histogram Per Z-Line

| vx+vy component lanes | line slots |
| ---: | ---: |
| 0 | 7774 |
| 32 | 541512 |
| 39 | 17269 |
| 46 | 88070 |
| 48 | 17269 |
| 64 | 264210 |

## Candidate Ceilings

| candidate | implementation class | lanes | lane reduction | v lane speedup ceiling | sampled-main ceiling |
| --- | --- | ---: | ---: | ---: | ---: |
| `whole_len16_tile_halfwarp_pack` | `tile_list_only` | 22433792 | 26.26% | 1.3560x | 1.0612x |
| `line_descriptor_len16_pack_remove_empty` | `line_descriptor` | 21594976 | 29.01% | 1.4087x | 1.0680x |
| `exact_active_point_list` | `point_descriptor` | 20647168 | 32.13% | 1.4734x | 1.0759x |

## Descriptor Traffic

- line descriptor uint64: `7.142 MiB/step aggregate-shots`
- point descriptor uint32: `78.762 MiB/step aggregate-shots`

## Gate

- decision: `allow_whole_len16_v_pml_cuda_prototype`
- best candidate: `exact_active_point_list`
- best v lane speedup ceiling: `1.4734x`
- best sampled-main speedup ceiling: `1.0759x`
- reason: Whole-tile length-16 velocity lines clear the >=5% sampled-main gate without a line descriptor. Open a macro-default-off CUDA prototype that mirrors the pressure len16 split.

Do not continue if rejected:

- Do not write a v-PML whole-len16 half-warp CUDA prototype below the >=5% gate.
- Do not re-open random v-PML tile-shape sweep or current-geometry vx/vy split.
- Do not use line/point descriptors without a descriptor/control overhead model.
