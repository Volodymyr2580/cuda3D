# Fused Wave-Step Refactor Gate

## Summary

This report evaluates the proposed fused-kernel / z-tiling / CPML-unification refactor against the current exact-FP32 CUDA3D frontier.

- current best: `current_best_v_pml_len16`
- formal WP speedup vs zmem: `1.2220x`
- formal Gradient speedup vs zmem: `1.2066x`
- max rel L2: `6.384336e-07`
- target gate for a new CUDA prototype: `1.0500x` sampled-main / repeat speedup ceiling
- decision: `reject_immediate_fused_wave_step_cuda_prototype`

The proposal is not ignored.  It is rejected as an immediate CUDA prototype because the current data show no safe `>=5%` exact-FP32 single-GPU ceiling after synchronization and halo costs.

## Current Profile Anchor

| region | duration | share |
| --- | ---: | ---: |
| `p_core` | `93.730us` | `33.00%` |
| `pressure-PML total` | `138.120us` | `48.63%` |
| `v-PML total` | `52.160us` | `18.37%` |
| sampled main total | `284.010us` | `100.00%` |

To pass a `>=5%` gate, a candidate needs to save about `13.524us` from sampled-main work.

## Why Full Fusion Is Not a Free Win

The tempting goal is to compute updated velocity and immediately consume it in pressure-PML.  The catch is that pressure uses neighboring `vx/vy` values.  With the accepted `32x4x2` PML tile, an exact fused tile must either read globally written neighbor velocities or recompute halo velocities locally.

| quantity | value |
| --- | ---: |
| pressure outputs per tile | `256` |
| `vx` points needed by pressure tile | `704` |
| `vy` points needed by pressure tile | `1152` |
| x/y component duplication factor | `3.6250x` |

That duplication is before accounting for extra registers, shared memory, branch control, and CPML state halos.

## Route Matrix

| route from proposal | modeled speedup | optimistic speedup | decision | reason |
| --- | ---: | ---: | --- | --- |
| `single_kernel_fuse_v_p_pml` | `0.7568x` | `0.7935x` | `reject_cuda_prototype` | Exact pressure update needs updated vx/vy from neighboring x/y positions.  Without a grid-wide barrier, a single kernel cannot safely consume neighbor-block velocity values; computing halo velocities locally duplicates about 3.6250x x/y component work before register/shared-memory overhead. |
| `reduce_kernel_launch_count` | `1.0029x` | `1.0052x` | `reject_as_primary_route` | Nsight Systems shows WP time is already almost fully GPU-kernel time; the visible WP-minus-kernel gap is only 0.28%, and an async scheduling prototype measured only 1.0052x WP speedup. |
| `z_direction_loop_tiling_prefetch` | `1.1248x` | `1.1248x` | `reject_current_exact_cuda_prototype` | Spatial z reuse is already present in p_core shared z tiles and accepted pressure/v-PML len16 z-line ownership.  True multi-step temporal z-tiling reopens the K=2 p_mid dependency problem, which ordinary CUDA and cluster DSM gates already rejected. |
| `unify_cpml_formula_template` | `1.0000x` | `1.0000x` | `documentation_or_cleanup_only` | A formula template can reduce source duplication, but current hot spots are final pressure writeback and recursive CPML state traffic.  A template alone does not remove bytes or synchronization. |
| `aos_or_soa_layout_rewrite` | `1.0000x` | `1.0000x` | `reject_aos_for_exact_current_best` | The code already uses SoA arrays, which match p_core and PML streaming access.  AoS would make single-field stencil loads stride across unrelated variables unless a new all-field fused ownership model passes first. |
| `mixed_precision_tensor_core` | `n/a` | `n/a` | `out_of_scope_for_exact_fp32_branch` | This may be a separate relaxed-precision branch, but it changes the tolerance policy and cannot be mixed into the current exact-FP32 line. |
| `cluster_or_persistent_fusion` | `0.9498x` | `1.1317x` | `design_only_rejected_by_current_cluster_model` | RTX 5090 supports clusters, but the current best DSM tile is slower than baseline (0.9498x); full cooperative K=2 needs 70688 blocks versus a 2040 block resident ceiling. |

## Gate

- decision: `reject_immediate_fused_wave_step_cuda_prototype`
- CUDA prototype allowed: `false`
- reason: The proposal is directionally ambitious, but the exact-FP32 current-best path already removed the cheap z/global round trips.  The remaining one-kernel fusion needs cross-CTA velocity availability or heavy halo recomputation, while launch reduction, simple z-tiling, CPML templating, and AoS layout do not pass the >=5% modeled repeat-speedup gate.

Next allowed work:

- If staying exact-FP32 single-GPU, only study a new persistent/cluster ownership representation that first beats the halo/DSM byte gate.
- If the goal is total throughput, move to multi-GPU batching using the current-best kernel stack.
- If the scientific tolerance can change, open a separate relaxed-precision branch with its own correctness policy.
