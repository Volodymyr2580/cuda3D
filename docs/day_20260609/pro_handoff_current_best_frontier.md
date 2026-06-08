# CUDA3D Pro Handoff: Current Best And Closed Exact-CUDA Frontier

## Executive Summary

Current formal best on the RTX 5090 single-GPU platform is:

```text
current_best_v_pml_len16
WP speedup vs zmem        1.222023x
Gradient speedup vs zmem  1.206588x
Elapsed speedup vs zmem   1.118261x
max rel L2                6.384336e-07
```

This is a valid current-best candidate, but it is not a `1.5x` milestone archive.
Reaching `1.5x` from here still needs another `1.2275x` WP speedup.

The exact ordinary-CUDA micro-prototype frontier is currently closed.  Do not
start another small CUDA prototype unless a new primitive, tolerance policy, or
application-level scope change is first approved by evidence.

## Current Best Flags

```text
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DCUDA3D_PML_ZMEM_IN_P
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
-DCUDA3D_CPML_VMEM_DISABLE_MPI
-DCUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
-DCUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
-DCUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
```

## Formal Benchmark Evidence

Report:

```text
reports/day_20260608/formal_vpmlen16_table_20260608_2359/summary.md
reports/day_20260608/formal_vpmlen16_table_20260608_2359/summary.json
```

Same-session `perf_1gpu_6shots`, 3 rounds, each config rebuilt before each run:

| candidate | mean WP speedup | mean Gradient speedup | max rel L2 |
| --- | ---: | ---: | ---: |
| `directfill` | `1.101172x` | `1.100029x` | `0` |
| `pressure_len16` | `1.194495x` | `1.179869x` | `6.384336e-07` |
| `current_best_v_pml_len16` | `1.222023x` | `1.206588x` | `6.384336e-07` |

## What Worked

Accepted current-best stack:

1. `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`
   - Works as ownership scaffold and improves v-PML.
2. `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
   - Reuses pressure-PML z-line recompute data.
3. `CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK`
   - Packs whole length-16 pressure-PML z-face lines.
4. `CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK`
   - Adds a smaller but repeat-stable v-PML improvement.

## Closed Routes

Do not reopen these without new profiler evidence and a new model:

- residual pressure branch-only split.
- pressure length-32 branch/control specialization.
- pressure length-23 or exact active-point descriptors.
- residual `p0 __ldg`, local `new_mem`, ptxas cache-policy, z-cache fill, or shared-z-cache micro-tuning.
- v-PML descriptor / point-list expansion after v-len16.
- direct z-face VP fusion or shared-VP retry.
- current p-core shared-plane/block/register family.
- K=2 ordinary CUDA temporal/wavefront prototype.
- CUDA Graph / host launch aggregation for this single-GPU case.

## Hard Gates

Latest ownership frontier:

```text
docs/day_20260608/ownership_frontier_gate.md
reports/day_20260608/ownership_frontier_gate.json
```

Decision:

```text
ordinary_exact_cuda_frontier_exhausted_for_micro_routes
ordinary CUDA allowed prototype count = 0
```

Current sampled-main profile anchor:

```text
sampled main total       284.010us
p-core                    93.730us  33.00%
pressure-PML total       138.120us  48.63%
pressure len16            66.180us  23.30%
pressure residual         71.940us  25.33%
v-PML total               52.160us  18.37%
```

Residual pressure route gate:

```text
docs/day_20260608/residual_pressure_route_gate.md
```

Key result:

```text
residual local speedup required for 5% sampled-main  1.2315x
perfect branch-efficiency ceiling                    1.0429x sampled-main
exact length-23 descriptor calibrated speedup         1.0153x sampled-main
```

## Allowed Next Directions

1. Handoff/reporting
   - Use this document as the starting point for Pro or another agent.
   - The next agent should read `AGENTS.md`, `AGENT_LOG.md`, and the three reports above before making changes.

2. Concrete cross-CTA or cluster-level primitive study
   - Only a concrete synchronization/ownership primitive can reopen temporal or producer-consumer fusion.
   - The study must first prove feasibility on RTX 5090 / `sm_120`, then produce a byte/synchronization model before CUDA code.

3. Precision-relaxation plan
   - Only allowed if the user explicitly changes the current tolerance policy.
   - Must define allowed relative L2, max-abs, NaN/Inf, and scientific acceptability before FP16/TF32 work.

4. Application-level batching / multi-shot scheduling
   - This is outside the exact CUDA-core route.
   - Must compare wall-clock and Gradient time, not only rank-local WP time.

## Operational Notes

- Remote root `/work/wenzhe/cuda3D` is dirty from older experiments.  Use isolated worktrees under `/work/wenzhe/cuda3D/.codex_worktrees/`.
- For isolated `perf_1gpu_6shots` worktrees, create a case-local `d_obs/` directory.  Do not symlink `d_obs` from the root checkout.
- Do not commit `.ncu-rep` files or large source-page dumps.
- Current stable benchmark platform has one RTX 5090; do not compare old 3GPU threshold archives directly to this single-GPU table.

## Recommended Next Action

Before writing more CUDA, ask Pro or the next agent to decide which scope change is acceptable:

```text
A. investigate cluster/cooperative persistent-kernel primitive
B. relax precision policy and define new correctness tolerances
C. switch to application-level multi-shot scheduling
D. stop CUDA-core sprint at current-best and package results
```

Without one of those scope changes, continuing ordinary exact-CUDA micro
prototyping is now explicitly prohibited by the current gates.
