# PML Fused Z-Slab Prototype Result

## Decision

`CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE` is correct under the tested cases, but it does not meet the continuation threshold.

Decision: stop this PML fusion line for now. Do not enable `CUDA3D_PML_FUSED_ZSLAB_SKIP_V_OWNED`.

Reason: the best validated version is slower than `zmem_reference` on repeat:

```text
perf_1gpu_6shots_repeat WP speedup       = 0.956846x
perf_1gpu_6shots_repeat Gradient speedup = 0.961207x
perf_1gpu_6shots_repeat wall speedup     = 0.985772x
```

The required continuation threshold was `>= 1.05x`.

## Implemented Scope

Branch:

```text
exp/pml-fused-zslab-prototype
```

Macro-gated implementation:

```text
CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE
CUDA3D_PML_FUSED_ZSLAB_DEBUG
CUDA3D_PML_FUSED_ZSLAB_SKIP_V_OWNED  // guarded as not implemented in phase 1
FusedZSlabBlockSize1/2/3             // default 32x4x2
FusedSafetyRadius                    // default 8
```

Main code changes:

- Added `cuda_fd3d_p_pml_fused_zslab_ns`.
- Added fused z-slab tile-list construction for lower/upper z slabs with x/y safety radius.
- Generic `p_pml_tile` skips fused-owned pressure points.
- `v_pml_tile` behavior is unchanged; no velocity pruning was enabled.
- Fused kernel updates only z pressure memory (`memory_dzz`) and `ZMEM_IN_P` old-to-next z velocity memory.
- Source injection and extraction order is unchanged.
- Default build path is unchanged unless `CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE` is defined.

## Tested Variants

### Strict Z-PML Face

Tag:

```text
fused_zslab_phase1_32x4x2
```

Scope:

- Fused-owned points only where `z < npml || z >= n1 - npml`.
- Generic PML tile list unchanged, with per-point skip inside generic kernel.

Result:

```text
correctness: pass
perf_1gpu: pass
perf_1gpu_6shots: pass
perf_1gpu_6shots_repeat: pass
```

Performance versus `zmem_reference`:

| Case | WP Speedup | Gradient Speedup | Wall Speedup |
|---|---:|---:|---:|
| `perf_1gpu` | 0.857697x | 0.865553x | 0.965157x |
| `perf_1gpu_6shots` | 0.858732x | 0.864723x | 0.928433x |
| `perf_1gpu_6shots_repeat` | 0.855143x | 0.863322x | 0.932692x |

Conclusion: correct but much slower because it adds a new kernel while generic PML still launches almost the full tile list.

### Residual-Skipping Z-Slab

Tag:

```text
fused_zslab_phase1_residual_fix_32x4x2
```

Scope:

- Fused-owned z range expanded to the z-side non-core slab: `z < npml + CorePmlMargin || z >= n1 - npml - CorePmlMargin`.
- Host-side pressure tile-list builder skips generic p tiles fully inside x/y safe columns.
- Generic p tile count is reduced, residual x/y edges/corners stay on generic path.
- A bug in the intermediate `fused_zslab_phase1_residual_32x4x2` run treated lower z-margin points as upper PML memory. It was fixed by changing the upper memory update to `else if (gtid1 >= n1 - npml)`.

Correctness:

| Test | Result | Error Summary |
|---|---|---:|
| debug dump step 0/1/2 | pass | bitwise match, but smoke case had `fused_tiles=0` |
| `correctness` | pass | rel L2 `0.0` for all 6 outputs |
| `perf_1gpu` | pass | rel L2 `7.147390e-07` |
| `perf_1gpu_6shots` | pass | max rel L2 `6.290364e-07` |
| `perf_1gpu_6shots_repeat` | pass | max rel L2 `6.290364e-07` |

Performance versus `zmem_reference`:

| Case | Baseline WP | Candidate WP | WP Speedup | Baseline Gradient | Candidate Gradient | Gradient Speedup | Wall Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `perf_1gpu` | 0.480130 | 0.491165 | 0.977533x | 0.509650 | 0.520283 | 0.979563x | 0.955172x |
| `perf_1gpu_6shots` | 2.393577 | 2.499972 | 0.957442x | 2.514862 | 2.613513 | 0.962253x | 0.979592x |
| `perf_1gpu_6shots_repeat` | 2.390644 | 2.498462 | 0.956846x | 2.514458 | 2.615939 | 0.961207x | 0.985772x |

Tile counts on `perf_1gpu`:

| Build | `v_tiles` | `p_tiles` | `fused_tiles` |
|---|---:|---:|---:|
| `zmem_reference` | 23100 | 22188 | 0 |
| residual fused | 23100 | 10524 | 11990 |

## NCU Summary

Reports:

```text
benchmarks/profiles/fused_zslab_residual_fix_ncu_main_20260607.ncu-rep
benchmarks/profiles/fused_zslab_residual_fix_ncu_warpstates_20260607.ncu-rep
```

Main kernel metrics:

| Kernel | Samples | Grid | Time Avg (us) | Compute/Mem % | DRAM % | Warps Active % | Reg/Thread |
|---|---:|---|---:|---:|---:|---:|---:|
| `p_pml_fused_zslab` | 10 | `(11990, 1, 1)` | 109.104 | 29.58 | 30.25 | 73.48 | 44 |
| `p_pml_tile` | 10 | `(10524, 1, 1)` | 87.091 | 35.62 | 40.39 | 69.90 | 44 |
| `v_pml_tile` | 10 | `(23100, 1, 1)` | 71.386 | 50.98 | 45.23 | 82.30 | 38 |
| `p_core` | 10 | `(1, 117, 233)` | 93.555 | 96.95 | 42.29 | 66.25 | 48 |

Compared with `zmem_reference`, baseline `p_pml_tile` was about `189.366 us`. The residual version reduces generic `p_pml_tile` to `87.091 us`, but adds `p_pml_fused_zslab` at `109.104 us`. The combined p-side sampled cost is therefore about `196.195 us`, slightly worse than baseline.

Top stalls:

| Kernel | Top Stalls |
|---|---|
| `p_pml_fused_zslab` | long_scoreboard `8.31`, short_scoreboard `2.69`, wait `2.10` |
| `p_pml_tile` | long_scoreboard `8.78`, wait `1.98`, short_scoreboard `1.53` |
| `v_pml_tile` | long_scoreboard `15.43`, wait `1.56`, not_selected `1.49` |
| `p_core` | long_scoreboard `8.60`, barrier `3.47`, not_selected `1.67` |

## Interpretation

This prototype reduced the generic p-PML tile count, but the new fused kernel did not remove enough global-memory dependency work. It mostly split the same p-side work into two kernels. The long-scoreboard pattern remains in both p-side kernels, and the extra launch plus extra tile traversal outweighs the generic tile reduction.

The next productive direction is not phase-2 velocity pruning unless a stronger ownership proof and a larger fused region are designed. Based on the current data, shift attention to `p_core` z-pencil/shared-memory or a more global algorithmic rewrite rather than this limited z-slab PML split.

