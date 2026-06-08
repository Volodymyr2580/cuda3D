# V-PML Len16 Half-Warp Prototype

## Context

- Current-best baseline: CPML vmem double-buffer + pressure z-cache + pressure len16 half-warp packing.
- New gate tool: `tools/v_pml_active_segment_packing_model.py`.
- Prototype macro: `CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK`.
- Remote worktree: `/work/wenzhe/cuda3D/.codex_worktrees/v_pml_len16_20260608_2238`.

## Model Gate

- v-PML sampled-main share: `21.95%`.
- Required v-kernel speedup for `>=5%` sampled-main: `1.2770x`.
- Whole length-16 tile half-warp packing lane ceiling:
  - v lane speedup ceiling: `1.3560x`.
  - sampled-main ceiling: `1.0612x`.
- Decision: open a macro-default-off CUDA prototype for whole-tile len16 packing only.

The line-descriptor and point-descriptor variants have higher lane ceilings, but they require descriptor/control overhead modeling before any CUDA implementation.

## Implementation

- Host split:
  - `split_v_pml_len16_tiles` separates whole z-face length-16 velocity tiles from residual velocity PML tiles.
  - Only tiles with full `4x2` x/y coverage and x/y inside the vx/vy safe interior are accepted.
- New kernel:
  - `cuda_fd3d_v_pml_len16_halfwarp_ns`.
  - One warp processes two active length-16 z-lines.
  - It computes the same `vx/vy` x/y pressure derivatives as the original `cuda_fd3d_v_pml_tile_ns`.
  - It does not update `mem_dx/mem_dy`, because accepted tiles are z-face interior tiles.
- Original residual kernel remains responsible for all non-packed velocity PML work.

## Validation

- Build: pass.
- Smoke:
  - pass, `outputs=3`.
  - smoke has `len16_tiles=0`, so it validates wiring but not packed-kernel coverage.
- Correctness:
  - baseline and candidate run pass.
  - correctness has `len16_tiles=0`, so perf case is the packed-kernel coverage test.
- `perf_1gpu_6shots` repeat:
  - 3 baseline runs and 3 candidate runs in the same remote worktree.
  - Candidate output compared against baseline round 1.
  - All compares pass with max rel L2 `0`.

## Performance

- mean base WP: `2.052228s`.
- mean candidate WP: `1.988482s`.
- WP speedup: `1.032058x`.
- mean base Gradient: `2.169915s`.
- mean candidate Gradient: `2.109314s`.
- Gradient speedup: `1.028730x`.
- Candidate perf tile coverage:
  - final shot log line shows `len16_tiles=10000`, `residual_v_tiles=9524`.
  - Aggregate per-shot logs match the model order of magnitude.

## Decision

- Strict `>=5%` breakthrough gate: not reached.
- Minor `>=2%` candidate gate: passed.
- Keep `CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK` as a macro-default-off minor current-best candidate.
- Stop expanding this route unless a new model proves descriptor/control overhead and still predicts `>=5%` repeat speedup.

Do not continue:

- v-PML line-descriptor len16 packing without an overhead model.
- v-PML exact active-point descriptor prototype without an overhead model.
- random v-PML tile-shape sweeps.
- current-geometry vx/vy component-owner split.

## Artifacts

- `docs/day_20260608/v_pml_active_segment_packing_model.md`
- `reports/day_20260608/v_pml_active_segment_packing_model.json`
- `reports/day_20260608/v_pml_len16_prototype_20260608_2238/summary.md`
- `reports/day_20260608/v_pml_len16_prototype_20260608_2238/perf6_retry_summary.json`
