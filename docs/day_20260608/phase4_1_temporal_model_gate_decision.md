# Phase 4.1 Temporal Pipeline Model Gate

Date: 2026-06-08

Decision: do not write a direct K=2 temporal CUDA prototype yet.

## What Was Tested

Added and ran `tools/temporal_pipeline_model.py` locally and on `/work/wenzhe/cuda3D`.

Model inputs:

- case: `benchmarks/cases/perf_1gpu_6shots`
- pressure stencil radius: `7`
- core PML margin: `4`
- current p_core block: `128 x 2 x 1`
- Phase 4 timing input: `reports/day_20260608/phase4_global_temporal_pipeline_design_summary.json`

Primary output:

- `docs/day_20260608/temporal_pipeline_model.md`
- `reports/day_20260608/temporal_pipeline_model.json`

## Geometry

For the 384 x 384 x 95 perf case:

- pressure core: `87 x 376 x 376`
- pressure core points: `12299712`
- K=2 deep core: `73 x 362 x 362`
- K=2 deep-core share: `77.78%`

The geometry is large enough that K=2 is worth analyzing. The problem is not coverage.

## Byte Model

Current `cuda_fd3d_p_core_ns` roughly uses:

- p1 global floats/output: `29.109375`
- p0/cw2/store floats/output: `3.0`
- current bytes/output: `128.438`
- current bytes/core step: `1506.562 MiB`
- current bytes/two core steps: `3013.123 MiB`

An ideal K=2 implementation that reuses the second-step `p(t+1)` stencil locally without halo duplication could save:

- saved bytes/deep output: `116.438`
- saved bytes/pair: `1062.265 MiB`
- p_core pair reduction upper bound: `35.25%`
- sampled-main speedup upper bound: `1.103x`

This confirms Pro's intuition that real temporal reuse could matter. But this is a no-duplication upper bound, not an implementable plan.

Once CTA-local `p_mid` halo duplication is included, the candidate tile shapes become clearly non-viable:

| candidate | p_mid/output | local pair bytes / baseline | verdict |
| --- | ---: | ---: | --- |
| T1 | 29.11 | 14.15x | fail halo duplication |
| T2 | 23.20 | 11.29x | fail halo duplication |
| T3 | 23.20 | 11.29x | fail halo duplication |
| T4 | 31.62 | 15.37x | fail halo duplication |
| T5 | 43.88 | 21.30x | fail halo duplication |
| T6 | 23.75 | 11.55x | fail halo duplication |

## Synchronization Gate

A safe global-middle design:

- materializes `p(t+1)` in global memory;
- synchronizes through a kernel boundary or grid sync;
- computes `p(t+2)` from global `p(t+1)`.

This preserves correctness, but it also preserves almost all second-step global stencil traffic. It fails the meaningful speedup gate.

A cooperative-grid design:

- current p_core grid blocks: `70688`
- assumed RTX 5090 resident block capacity: `1360`
- over capacity factor: `51.98x`

So a simple cooperative grid-wide barrier cannot cover the current p_core grid.

A naive CTA-local p_mid design:

- has candidate shared-memory shapes that fit under 99 KiB;
- is the only no-duplication ideal route with `>5%` sampled-main upside;
- but loses badly after local halo duplication is counted;
- and is exactly the previously rejected/forbidden CTA-local two-step family unless redesigned as a real swept/wavefront algorithm.

It also has unresolved correctness hazards:

- ownership of overlapping p_mid halos;
- source injection between `p(t+1)` and `p(t+2)`;
- receiver extraction at the intermediate step;
- duplicate or racing writes for `p(t+1)` halo values;
- interaction with PML/shell update order.

## Gate Result

`stop_cuda_prototype`.

Do not implement `CUDA3D_WAVESTEP_ENGINE_V2_TEMPORAL_PIPELINE` as a direct K=2 fused kernel yet.

The only acceptable next temporal direction is a source-aware swept/wavefront design that explicitly solves:

1. p_mid ownership without duplicate races;
2. source injection between substeps;
3. intermediate-step receiver extraction;
4. shell/PML reconciliation;
5. a proof that dependency cones do not read half-updated values.

Until that design exists, CUDA code would be fast-looking but numerically dangerous.

## Next Work

Recommended next phase:

`Phase 4.2: source-aware swept/wavefront temporal design`

Minimum deliverables before code:

- block/wavefront ownership diagram;
- per-pair schedule for V/P core/PML/source/extract/swap;
- explicit exclusion or embedding rule for source bell;
- byte model including halo duplication;
- predicted WP speedup `>=1.05x`;
- debug plan comparing step 0/1/2/3 dumps before perf testing.
