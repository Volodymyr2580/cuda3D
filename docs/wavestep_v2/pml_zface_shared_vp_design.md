# PML Z-Face Shared-Tile VP Design

## Decision

The previous `CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY` path is stopped because it recomputed direct global-memory p1 x/y second derivatives and was slower than `zmem_reference`.

The only allowed z-face VP retry is a CTA-local shared-pressure-tile design:

- load `p1` for a z-face tile plus halo into shared memory;
- compute local x/y velocity intermediates from shared `p1`;
- immediately consume those intermediates in the pressure update;
- avoid writing the x/y velocity intermediates to global memory for this fused z-face subregion.

This is not a host-side pointer shuffle and not another direct p1 second-derivative replacement. The value proposition must be reduced global traffic for the velocity round trip.

## Baseline

The comparison baseline remains `zmem_reference`:

```bash
NVFLAGS="-O3 -arch=sm_120 --use_fast_math \
-DCUDA3D_PML_RECOMPUTE_Z \
-DCUDA3D_PML_TILE_LIST \
-DCUDA3D_PML_ZMEM_IN_P \
-DPmlTileBlockSize1=32 \
-DPmlTileBlockSize2=4 \
-DPmlTileBlockSize3=2"
```

The same-session night baseline measured by Phase 0:

- `perf_1gpu_6shots_zmem_reference_night_perf6_a`: WP `2.447898s`, Gradient `2.559258s`
- `perf_1gpu_6shots_zmem_reference_night_perf6_b`: WP `2.449256s`, Gradient `2.562289s`
- mean WP `2.448577s`
- mean Gradient `2.560774s`

## Budget Summary

Budget tool:

```text
tools/pml_zface_shared_tile_budget.py
```

Generated evidence:

```text
docs/wavestep_v2/pml_zface_shared_tile_budget.md
reports/wavestep_v2_night_20260608/pml_zface_shared_tile_budget.json
```

For `perf_1gpu_6shots`, the estimated padded domain is `408 x 408 x 119`, with `npml=12` and `CorePmlMargin=4`.

Estimated pure z-face PML pressure work:

- total points: `19,809,216`
- core points: `12,299,712`
- estimated PML points: `7,509,504`
- estimated pure z-face PML points: `3,393,024`
- z-face share of estimated PML pressure work: `45.18%`

Candidate tile budget with halo `7`:

| candidate | output z/x/y | shared z/x/y | shared bytes | outputs | shared p loads/output | reduction vs direct xy estimate | blocks/SM by shared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | `8x16x16` | `22x30x30` | `79,200` | `2,048` | `9.668` | `67.77%` | `1` |
| S2 | `12x16x12` | `26x30x26` | `81,120` | `2,304` | `8.802` | `70.66%` | `1` |
| S3 | `8x24x12` | `22x38x26` | `86,944` | `2,304` | `9.434` | `68.55%` | `1` |
| S4 | `12x12x12` | `26x26x26` | `70,304` | `1,728` | `10.171` | `66.10%` | `1` |

## Prototype Choice

Use S2 first:

- it has the lowest estimated shared `p1` loads per output among the four candidates;
- it keeps shared memory under the 99KB opt-in block limit;
- it produces 2,304 output points per CTA, or 9 outputs/thread with 256 threads;
- its x/y footprint is less extreme than S3, reducing boundary and indexing risk.

S4 is the fallback if occupancy or register pressure is worse than expected, because it uses the least shared memory.

## Correctness Hazards

The debug prototype must explicitly handle these hazards:

- source injection and receiver extraction cannot be fused blindly;
- `.nav` is a binary coordinate file, so the first prototype should reject or skip tiles that may contain source/receiver points;
- CPML memory ownership must use the double-buffer discipline from `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`;
- z memory already lives in pressure via `CUDA3D_PML_ZMEM_IN_P`, so the prototype must not reintroduce old `mem_dz` write/read ownership;
- the macro must remain default-off and single-rank-only until correctness is proven.

## Gate

Proceed to debug-only implementation only if:

- the current `zmem_reference` baseline is rebuilt and recorded;
- `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` still passes correctness and does not regress;
- Nsight Compute or an equivalent profile shows the z-face PML path remains global-memory-traffic limited.

Commit candidate only if:

- debug dump step 0/1/2 aligns;
- smoke/correctness/perf outputs are finite and `rel_l2 <= 1e-5`;
- `perf_1gpu_6shots repeat` is at least `>=5%` faster than the same-session zmem baseline.

## Result

This design was implemented and tested as a default-off debug prototype after the budget and NCU gates.

Result:

```text
stop
```

Reason:

- S2 p-only, S4 p-only, and S4 staged-V variants all passed correctness.
- All variants were significantly slower than `zmem_reference`.
- The detailed result is recorded in `docs/wavestep_v2/shared_vp_debug_result.md`.

Do not continue this shared-tile z-face VP shape without a materially different profiler-backed dataflow.
