# WAVESTEP Engine V2 Phase 2 Fused Z-Face Report

Date: 2026-06-08

## Decision

Stop `CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY` in its current direct-second-derivative form.

Both tested variants are numerically correct, but neither reaches the performance gate. The direct replacement of the `vx/vy` global round trip with p1-based x/y second derivatives is slower than the stable `zmem_reference`.

## Baseline

Stable zmem flags:

```text
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DCUDA3D_PML_ZMEM_IN_P
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
```

Same-session zmem binary:

```text
c768270c431b3922f803fc787b1eaaffdc8967b072dfc5f74f30c0a94bf459e5
```

Same-session zmem perf6:

| Run | WP Time | Gradient Time | Wall Time |
|---|---:|---:|---:|
| `perf_1gpu_6shots_wavestep_v2_zmem_same_session_after_fused_20260608_010619` | 2.432802s | 2.545055s | 0:04.88 |
| `perf_1gpu_6shots_wavestep_v2_zmem_same_session_after_fused_repeat_20260608_010625` | 2.436119s | 2.550727s | 0:04.58 |

Mean zmem WP: `2.434461s`

## Candidate A: Separate Fused Z-Face Kernel

Flags:

```text
zmem flags
-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
-DCUDA3D_CPML_VMEM_DISABLE_MPI
-DCUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY
```

Binary:

```text
63531df023ebab0bf8104eafe4b420658f6eb1cfd0801b6cbe3d5d4dd4adad4a
```

Implementation:

```text
v_pml skips fused-owned vx/vy global writes.
p_pml skips fused-owned z-face points.
Separate cuda_fd3d_pml_fused_vp_zface_ns kernel updates those points.
```

Correctness:

```text
smoke: pass
correctness: pass
perf6 repeat output comparison: pass
max rel L2 in perf6 repeat: 6.358816e-07
```

Performance:

| Run | WP Time | Gradient Time | Wall Time |
|---|---:|---:|---:|
| `perf_1gpu_6shots_wavestep_v2_fused_zface_release_20260608_005956` | 2.656186s | 2.771636s | 0:05.09 |
| `perf_1gpu_6shots_wavestep_v2_fused_zface_release_repeat_20260608_010002` | 2.663968s | 2.776977s | 0:05.17 |

Mean WP: `2.660077s`

Speed ratio vs same-session zmem:

```text
2.434461 / 2.660077 = 0.915184x
```

This is a slowdown, not a speedup.

## Candidate B: Inline Fused Path Inside p_pml

Binary:

```text
c88b2acf88025f7796288603250d3f63749a2af8b548449af9b1373507e1cff9
```

Implementation:

```text
v_pml skips fused-owned vx/vy global writes.
p_pml_tile directly handles fused z-face points with p1-based x/y second derivatives.
No extra fused kernel launch.
```

Correctness:

```text
smoke: pass
correctness: pass
perf6 repeat output comparison: pass
max rel L2 in perf6 repeat: 6.358816e-07
```

Performance:

| Run | WP Time | Gradient Time | Wall Time |
|---|---:|---:|---:|
| `perf_1gpu_6shots_wavestep_v2_fused_zface_inline_release_20260608_010449` | 2.691287s | 2.812600s | 0:05.10 |
| `perf_1gpu_6shots_wavestep_v2_fused_zface_inline_release_repeat_20260608_010455` | 2.693871s | 2.817426s | 0:05.21 |

Mean WP: `2.692579s`

Speed ratio vs same-session zmem:

```text
2.434461 / 2.692579 = 0.904137x
```

This is also a slowdown.

## Interpretation

The direct z-face fusion is mathematically valid, but it replaces `vx/vy` global reads with more p1 loads and extra arithmetic. On RTX 5090, that trade is losing:

```text
saved: fused-owned vx/vy writes in v_pml, vx/vy reads in p_pml
added: p1-based x/y second derivative work, extra branch pressure, CPML double-buffer overhead
```

The separate-kernel version also pays an extra launch and z-face tile traversal. The inline version removes that launch, but still remains slower, so the core problem is the direct p1 recompute strategy itself.

## Stop Rule

Do not continue the following route without new profiler evidence:

```text
CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY using direct p1 x/y second derivatives as a replacement for vx/vy global round trip
```

Only reopen PML z-face fusion if the new design keeps velocity intermediates CTA-local with shared-memory reuse or otherwise proves with Nsight Compute that it reduces total memory stalls. Do not repeat pressure-only split or direct global p1 recompute.

## Artifacts

Comparison reports on server:

```text
reports/wavestep_engine_v2_phase2_fused_zface_20260608_010000/smoke_vs_zmem
reports/wavestep_engine_v2_phase2_fused_zface_20260608_010000/correctness_vs_zmem
reports/wavestep_engine_v2_phase2_fused_zface_20260608_010000/perf6_repeat_vs_zmem
reports/wavestep_engine_v2_phase2_fused_zface_20260608_010000/inline_smoke_vs_zmem
reports/wavestep_engine_v2_phase2_fused_zface_20260608_010000/inline_correctness_vs_zmem
reports/wavestep_engine_v2_phase2_fused_zface_20260608_010000/inline_perf6_repeat_vs_zmem
```

Final server binary restored to zmem:

```text
c768270c431b3922f803fc787b1eaaffdc8967b072dfc5f74f30c0a94bf459e5
```
