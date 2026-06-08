# Length-16 Half-Warp Pressure-PML Prototype

## Decision

Accept `CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK` as a macro-gated,
default-off prototype for the current RTX 5090 single-GPU optimization line.

This prototype passes the required debug/correctness/performance gates and
becomes the current best candidate on top of the direct-fill pressure z-cache
path.  It is not a `perf_3gpu` threshold archive, because the evidence here is
from the RTX 5090 `perf_1gpu_6shots` gate.

## Build Flags

The accepted candidate uses the current direct-fill best flags plus the new
half-warp packing macro:

```bash
NVFLAGS="-O3 -arch=sm_120 --use_fast_math \
-DCUDA3D_PML_RECOMPUTE_Z \
-DCUDA3D_PML_TILE_LIST \
-DCUDA3D_PML_ZMEM_IN_P \
-DPmlTileBlockSize1=32 \
-DPmlTileBlockSize2=4 \
-DPmlTileBlockSize3=2 \
-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL \
-DCUDA3D_CPML_VMEM_DISABLE_MPI \
-DCUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE \
-DCUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK"
```

The macro intentionally requires:

- `CUDA3D_PML_RECOMPUTE_Z`
- `CUDA3D_PML_ZMEM_IN_P`
- `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
- `PmlTileBlockSize=32x4x2`

## Implementation Summary

Changed files:

- `include/inc3D/single_solver.h`
- `src/rem_fd.cu`
- `src/single_solver.cu`

The host side splits pressure-PML tiles into:

- residual tiles handled by the existing `cuda_fd3d_p_pml_tile_ns`
- whole length-16 active-z tiles handled by the new
  `cuda_fd3d_p_pml_len16_halfwarp_ns`

The host launch path guards both residual allocation and residual launch when
the residual tile count is zero.  The current `perf_1gpu_6shots` case still has
nonzero residual tiles, but the guard prevents future all-packed cases from
issuing a 0-size `cudaMalloc` or 0-grid kernel launch.

The new kernel maps one warp to two length-16 z-lines:

- lanes `0..15` update the first line
- lanes `16..31` update the second line
- one CTA has `32x4x1` threads, so it covers eight packed lines
- shared z-line cache length is `16 + 7`, preserving the accepted pressure
  z-recompute cache dataflow

The prototype does not reopen the rejected z-face direct derivative, z-face
fusion, or shared-VP routes.  It still consumes the existing `vx/vy` divergence
path and only changes active pressure-PML lane ownership for length-16 z-lines.

## Validation Evidence

Remote isolated worktree:

```text
/work/wenzhe/cuda3D/.codex_worktrees/sprint_0648
```

Smoke:

```text
benchmarks/runs/smoke_1gpu_len16_halfwarp_smoke_datafixed_20260608_153645
returncode: 0
outputs: 3
```

Post-review guard rebuild smoke:

```text
benchmarks/runs/smoke_1gpu_len16_halfwarp_guard_rebuild_20260608_155133
returncode: 0
outputs: 3
tile split: len16 0, residual pressure-PML 240
```

Correctness:

```text
baseline:  benchmarks/runs/correctness_directfill_base_for_len16_ab_20260608_152404
candidate: benchmarks/runs/correctness_len16_halfwarp_candidate_20260608_152436
compare:   reports/day_20260608/len16_halfwarp_correctness_compare_20260608_152526
pass:      True
rel L2:    0 for all 6 outputs
```

The correctness case has no length-16 pressure-PML tiles, so it validates the
macro wiring and residual path.  The `perf_1gpu_6shots` and debug profile cases
exercise the new packed kernel.

Debug dump gate on `profile_1gpu`:

```text
reports/day_20260608/len16_halfwarp_debug_profile_20260608_153419
step 0: pass, all arrays rel L2 0
step 1: pass, all arrays rel L2 0
step 2: pass, p0 rel L2 7.852061e-09, all other arrays rel L2 0
```

`perf_1gpu_6shots` repeat:

```text
reports/day_20260608/len16_halfwarp_perf6_repeat_20260608_152944
rounds: 3
pass: True
mean base WP: 2.207751s
mean candidate WP: 2.039080s
mean WP speedup: 1.082719x
mean base Gradient: 2.316433s
mean candidate Gradient: 2.159948s
mean Gradient speedup: 1.072448x
max rel L2: 6.384336e-07
```

Per-round timing:

| round | base WP | candidate WP | WP speedup | base Gradient | candidate Gradient | Gradient speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.212397 | 2.041717 | 1.083596 | 2.322951 | 2.161666 | 1.074611 |
| 2 | 2.203528 | 2.037935 | 1.081255 | 2.312805 | 2.159917 | 1.070784 |
| 3 | 2.207329 | 2.037589 | 1.083304 | 2.313544 | 2.158260 | 1.071949 |

Candidate tile split on the six perf shots:

| shot/run order | len16 tiles | residual pressure-PML tiles |
| ---: | ---: | ---: |
| 1 | 10816 | 7168 |
| 2 | 12064 | 8032 |
| 3 | 10816 | 7648 |
| 4 | 10816 | 7408 |
| 5 | 12064 | 8300 |
| 6 | 10816 | 7892 |

## Interpretation

This is the first post-direct-fill CUDA prototype in this sprint that clears
the `>=5%` meaningful repeat gate.  It validates the active-segment model:
the remaining pressure-PML issue is not just instruction syntax, but warp lane
ownership on partially active z-lines.

The measured gain is lower than the model ceiling because the new kernel still
pays shared z-cache, `vx/vy` divergence loads, pressure update, and launch costs.
Even so, the repeat gain is stable enough to keep it as the next current best
candidate.

The estimated cumulative WP speedup versus the old `zmem_reference` line is
approximately:

```text
direct-fill vs zmem:      1.100929x
len16 vs direct-fill:     1.082719x
estimated product:        1.191983x
```

This product is useful orientation only.  A formal cumulative table still needs
same-session zmem/direct-fill/len16 reruns if the number is used externally.

## Next Work

Allowed next directions:

- Profile the accepted len16 candidate with Nsight Compute to see whether the
  remaining pressure-PML bottleneck moved from lane utilization to memory
  coalescing, shared-memory pressure, or final `p0/mem_dzz` update latency.
- Explore length-23 handling only after a static model shows a meaningful
  sampled-main ceiling and the descriptor/control overhead is bounded.
- Explore exact active-point or compact descriptor forms only if descriptor
  traffic and indexing overhead are modeled first.

Stop boundaries:

- Do not repeat simple active-line list compaction.
- Do not reopen z-face direct derivative/fusion/shared-VP variants.
- Do not return to `p0` read-only load, local `new_mem`, ptxas cache-policy,
  inject/extract block-size, or current-tile `vx/vy` component split routes
  without new profiler evidence.
