# Phase 4.2 Source-Aware Temporal Gate Decision

Date: 2026-06-08

Decision: stop swept/wavefront CUDA prototype for the current K=2 temporal route.

## What Was Added

Tool:

- `tools/source_aware_temporal_model.py`

Outputs:

- `docs/day_20260608/source_aware_temporal_model.md`
- `reports/day_20260608/source_aware_temporal_model.json`

The tool reads the real benchmark `.nav` file, reproduces the shot-local y/x subdomain cropping, and checks source/receiver footprints against the K=2 deep-core temporal region.

## Findings

For `perf_1gpu_6shots`:

- aggregate K=2 deep-core share across shot-local subdomains: `73.22%`
- source influence overlaps K=2 deep core in `0` shots
- receiver footprint overlaps K=2 deep core in `0` shots

Shot-local domains:

| shot | y/x domain | K=2 deep share |
| ---: | ---: | ---: |
| 0 | 216 x 216 | 72.99% |
| 1 | 216 x 241 | 73.56% |
| 2 | 216 x 217 | 73.02% |
| 3 | 217 x 216 | 73.02% |
| 4 | 217 x 241 | 73.58% |
| 5 | 217 x 217 | 73.04% |

This is an important negative/positive split:

- Positive: source and receiver placement do not kill temporal blocking in this benchmark.
- Negative: this does not solve the core `p(t+1)` ownership problem.

## Why The Prototype Still Stops

Phase 4.1 already showed:

- ideal no-duplication K=2 sampled-main upper bound: `1.103x`
- safe global-middle design: no meaningful byte saving
- cooperative grid-sync: over-capacity by about `51.98x`
- CTA-local candidates after `p_mid` halo duplication: `11.29x` to `21.30x` baseline pair bytes

Phase 4.2 removes source/receiver as the blocker, but leaves the ownership blocker intact.

Direct CUDA code would need one of these, none of which currently passes:

1. A non-duplicating `p_mid` ownership mechanism without grid-wide sync.
2. A swept/wavefront schedule with bounded halo duplication and no half-updated reads.
3. A safe global intermediate design that somehow avoids reloading the `p(t+1)` stencil.

No such schedule is currently specified.

## Gate

`stop_swept_wavefront_cuda_prototype`.

Do not write temporal CUDA code until there is a new ownership mechanism with:

- predicted WP speedup `>=1.05x` after halo duplication;
- explicit source/extract schedule;
- proof that no CTA reads half-updated values;
- debug dump plan for steps 0/1/2/3.

## Next Direction

Temporal blocking is not permanently dead, but the current K=2 path has no implementable prototype gate.

Recommended next autonomous direction:

- keep `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` as stable scaffold;
- stop temporal work for now;
- switch to pressure PML dataflow or wave-step scheduling around the dominant `cuda_fd3d_p_pml_tile_ns` path, because NCU shows it remains the largest sampled kernel.
