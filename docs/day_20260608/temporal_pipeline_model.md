# Temporal Pipeline Byte/Synchronization Model

## Case Geometry

- case_dir: `/work/wenzhe/cuda3D/benchmarks/cases/perf_1gpu_6shots`
- logical ny/nx/nz: `384/384/95`
- domain n3/n2/n1: `408/408/119`
- npml/radius/CorePmlMargin: `12/7/4`
- pressure core z/x/y: `87/376/376`
- pressure core points: `12299712`
- K=2 deep core z/x/y: `73/362/362`
- K=2 deep core share: `77.78%`

## Current P-Core Byte Model

- p_core block z/x/y: `128/2/1`
- estimated p1 global floats/output: `29.109375`
- p0/cw2/store floats/output: `3.000000`
- estimated current bytes/output: `128.438`
- estimated current bytes/core step: `1506.562 MiB`
- estimated current bytes/two core steps: `3013.123 MiB`

## Ideal K=2 Upper Bound

This is an impossible upper bound unless the second-step `p(t+1)` stencil is reused locally without unsafe CTA-boundary reads.

- saved bytes/deep output: `116.438`
- saved bytes/pair: `1062.265 MiB`
- p_core pair reduction upper bound: `35.25%`
- sampled-main speedup upper bound: `1.103x`

## Cooperative Grid Gate

- p_core grid blocks: `70688` with grid z/x/y `[1, 188, 376]`
- assumed resident block capacity: `1360`
- over capacity factor: `51.98x`
- fits cooperative grid: `False`

## CTA-Local P-Mid Candidates

| name | output z/x/y | p_mid z/x/y | shared KiB | p_mid/output | local pair bytes / baseline | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| T1 | 32x4x4 | 46x18x18 | 58.2 | 29.11 | 14.15x | fail_halo_duplication |
| T2 | 16x8x4 | 30x22x18 | 46.4 | 23.20 | 11.29x | fail_halo_duplication |
| T3 | 16x4x8 | 30x18x22 | 46.4 | 23.20 | 11.29x | fail_halo_duplication |
| T4 | 32x8x2 | 46x22x16 | 63.2 | 31.62 | 15.37x | fail_halo_duplication |
| T5 | 64x4x2 | 78x18x16 | 87.8 | 43.88 | 21.30x | fail_halo_duplication |
| T6 | 24x6x4 | 38x20x18 | 53.4 | 23.75 | 11.55x | fail_halo_duplication |

## Gate

- verdict: `stop_cuda_prototype`
- meaningful speedup required: `1.050x`

Reasons:
- A global-middle K=2 design is safe but does not remove the p(t+1) global stencil traffic, so it has no meaningful byte saving.
- A cooperative grid-wide sync design cannot launch the full p_core grid resident at once under conservative RTX 5090 assumptions.
- The no-duplication ideal has >5% sampled-main upside, but concrete CTA-local p_mid candidates become slower after halo duplication.
- CTA-local p_mid reuse is also the previously rejected/forbidden two-step family unless redesigned as a source-aware swept/wavefront ownership algorithm.

Next allowed work: source-aware swept/wavefront temporal design or explicit Pro-approved CTA-local temporal research, not direct CUDA prototype.
