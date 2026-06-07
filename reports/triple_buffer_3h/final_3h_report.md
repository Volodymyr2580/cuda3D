# CUDA3D Pressure Triple-Buffer 3H Sprint Report

## Summary

- Branch: `exp/pressure-triple-buffer-pipeline`
- Base commit at sprint start: `c491351`
- Implementation status: `correctness_pass_perf_gate_fail`
- Final server binary status: rebuilt as `zmem_reference` after perf gate, not triple-buffer.
- Main conclusion: `CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE` is numerically correct and useful as an architecture scaffold, but it does not meet the `>=2%` repeat speedup gate on `perf_1gpu_6shots`.

## Implemented Changes

- `include/inc3D/single_solver.h`
  - Added macro-gated triple-buffer signatures for pressure kernels.
  - `p_core`, `p_pml`, `p_pml_tile`, and `p_pml_zface` accept explicit `p_next`, `p_curr`, `p_prev` when `CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE` is enabled.
- `src/single_solver.cu`
  - Split pressure update ownership:
    - read current pressure from `p_curr`
    - read previous pressure from `p_prev`
    - write next pressure to `p_next`
  - Preserved existing math and CPML memory update order.
- `src/rem_fd.cu`
  - Allocated the third pressure buffer only under `CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE`.
  - Kept `v_pml` reading `d_p1` as current pressure.
  - Kept source injection and receiver extraction after pressure/PML update.
  - Routed source/receiver through `p_next` in triple-buffer mode.
  - Rotated buffers as `p_prev <- p_curr`, `p_curr <- p_next`, `p_next <- old p_prev`.
  - Added debug-only active-domain NaN fill and finite coverage check.
  - Added optional pressure state dumps for `p_prev`, `p_curr`, `p_next`.

## Macros

- `CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE`
  - Enables triple-buffer pressure ownership.
- `CUDA3D_PRESSURE_TRIPLE_BUFFER_DEBUG`
  - Enables optional dump helper for `p_prev`, `p_curr`, `p_next`.
- `CUDA3D_PRESSURE_TRIPLE_BUFFER_DEBUG_FILL`
  - Fills only the active wavefield domain of `p_next` with NaN before pressure update, then verifies the active domain is finite after update.
  - The radius halo is intentionally not poisoned because it is a retained boundary/halo region, not a pressure update target.
- `CUDA3D_PRESSURE_TRIPLE_BUFFER_DISABLE_MPI`
  - Debug guard for single-rank testing.

## Validation

### Baseline

- `smoke_1gpu_zmem_reference_before_triple_buffer_3h_20260607_182919`
  - return code `0`
  - outputs `3`
  - WP `0.002368s`
  - Gradient `0.003193s`
- `correctness_zmem_reference_before_triple_buffer_3h_20260607_182922`
  - return code `0`
  - outputs `6`
  - WP `0.012920s`
  - Gradient `0.014544s`

### Default Build Safety

- Default zmem build after source refactor passed.
- `smoke_1gpu_default_after_triple_buffer_patch_20260607_184226`
  - comparison vs zmem baseline: pass
  - all files rel L2 `0.0`
- `correctness_default_after_triple_buffer_patch_20260607_184229`
  - comparison vs zmem baseline: pass
  - all files rel L2 `0.0`

### Triple-Buffer Debug

- First debug fill attempted to poison the entire padded pressure buffer.
  - Program completed, but output contained non-finite values from time step 2 onward.
  - Diagnosis: radius halo is intentionally retained and not part of active pressure update coverage; poisoning the whole padded buffer was too broad.
- Revised debug fill poisons only active wavefield cells and checks finite coverage after pressure/PML update.
- `smoke_1gpu_triple_buffer_debug_fill_active_20260607_185044`
  - comparison vs zmem baseline: pass
  - all files rel L2 `0.0`

### Triple-Buffer Release Correctness

- `smoke_1gpu_triple_buffer_release_20260607_185212`
  - comparison vs zmem baseline: pass
  - all files rel L2 `0.0`
- `correctness_triple_buffer_release_20260607_185215`
  - comparison vs zmem baseline: pass
  - all files rel L2 `0.0`
- `perf_1gpu_6shots_triple_buffer_release_perf6_ab_20260607_185624`
  - comparison vs zmem perf6 output: pass
  - all files rel L2 `0.0`

## Performance

### perf_1gpu A/B

| Build | Run | WP (s) | Gradient (s) | Wall |
|---|---|---:|---:|---:|
| zmem | `perf_1gpu_zmem_reference_before_triple_ab_20260607_185404` | 0.482847 | 0.511858 | 0:02.86 |
| triple | `perf_1gpu_triple_buffer_release_ab_20260607_185440` | 0.473002 | 0.502685 | 0:02.62 |

- WP speedup: `1.0208x`
- Gradient speedup: `1.0182x`

### perf_1gpu_6shots Repeat A/B

| Build | Run | WP (s) | Gradient (s) | Wall |
|---|---|---:|---:|---:|
| triple | `perf_1gpu_6shots_triple_buffer_release_perf6_ab_20260607_185624` | 2.401650 | 2.515900 | 0:04.53 |
| triple | `perf_1gpu_6shots_triple_buffer_release_perf6_ab_repeat_20260607_185629` | 2.403235 | 2.518785 | 0:04.78 |
| zmem | `perf_1gpu_6shots_zmem_reference_perf6_ab_after_triple_20260607_185736` | 2.412274 | 2.530637 | 0:04.84 |
| zmem | `perf_1gpu_6shots_zmem_reference_perf6_ab_after_triple_repeat_20260607_185742` | 2.414473 | 2.535884 | 0:04.84 |

- WP mean speedup: `1.0045x`
- Gradient mean speedup: `1.0063x`
- Result: below the `>=2%` repeat speedup gate.

## Decision

- Do not promote triple-buffer release into `zmem_reference`.
- Keep the implementation macro-gated and default-off.
- Treat it as a correct structural foundation for later temporal pipeline experiments, not as a standalone accepted speedup.

## Risks

- The current triple-buffer path adds one pressure buffer, so memory footprint increases by one active padded pressure volume.
- The implementation is validated only on single GPU / single MPI rank.
- `perf_1gpu` showed a small positive result, but `perf_1gpu_6shots repeat` did not meet the acceptance threshold.
- No NCU run was performed after release correctness because the repeat performance gate was not strong enough to justify profiler expansion in this sprint.

## Recommendation

- Next work should use this explicit `p_prev/p_curr/p_next` ownership to prototype a true temporal pipeline or multi-step pressure schedule.
- Do not spend more time on standalone triple-buffer as a performance feature.
- Keep final deployed binary as stable `zmem_reference` until a later candidate clears correctness plus repeat speedup gates.
