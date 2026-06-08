# PML Vx/Vy Round-Trip Ownership Gate

## Context

- sampled main: `297.248us`
- p_core: `93.547us`
- v_pml: `65.248us`
- pressure-PML total: `138.453us`
- formal current-best WP speedup vs zmem: `1.1928x`
- baseline component lanes / active pressure lane: `1.7747`

Generous savable-time model:

- len16 unknown/unparsed source time assigned to vx/vy: `4.056us`
- residual pressure-PML generous savable fraction: `20.00%`
- residual generous savable time: `14.537us`
- total generous vx/vy-roundtrip savable time: `18.593us`
- duplicate velocity work factor allowed for 5% sampled-main speedup: `1.068`

## Candidate Macro Tiles

| candidate | macro x/y | shared bytes | duplicate v work | sampled-main speedup | shared OK | decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `current_pressure_tile_4x2` | `4/2` | `7424` | `4.085` | `0.6193x` | `True` | `reject_cuda_prototype` |
| `macro_8x4` | `8/4` | `18944` | `2.606` | `0.7752x` | `True` | `reject_cuda_prototype` |
| `macro_16x8` | `16/8` | `54272` | `1.866` | `0.8868x` | `True` | `reject_cuda_prototype` |
| `macro_16x16` | `16/16` | `94208` | `1.620` | `0.9315x` | `True` | `reject_cuda_prototype` |
| `macro_32x8` | `32/8` | `101376` | `1.743` | `0.9086x` | `False` | `reject_cuda_prototype` |
| `macro_32x16` | `32/16` | `174080` | `1.497` | `0.9556x` | `False` | `reject_cuda_prototype` |
| `ideal_no_duplicate_cross_cta_owner` | `n/a/n/a` | `n/a` | `1.000` | `1.0667x` | `None` | `reject_not_ordinary_cuda` |

## Gate

- decision: `reject_vxvy_roundtrip_ownership_cuda_prototype`
- best candidate: `ideal_no_duplicate_cross_cta_owner`
- best sampled-main speedup ceiling: `1.0667x`
- reason: Under a generous vx/vy-roundtrip savings budget, an implementable CTA-local cache would need duplicate velocity/CPML work factor <= 1.068.  Feasible macro tiles exceed that or exceed a conservative shared-memory limit, while the only passing ceiling requires impossible cross-CTA exchange without global memory.

Allowed next directions:

- source-aware multi-step/wavefront design only after synchronization and halo ownership proof
- precision-relaxation study only with explicit tolerance policy
- application-level multi-shot batching if CUDA-core exact routes remain gated off

Do not continue:

- CTA-local vx/vy shared-cache fusion under current tile/macro-tile ownership
- RECOMPUTE_X/Y/XYZ or direct p1 x/y derivative replacement
- current-geometry vx/vy component-owner split
- ordinary CUDA producer-consumer vx/vy fusion that relies on cross-CTA shared values
