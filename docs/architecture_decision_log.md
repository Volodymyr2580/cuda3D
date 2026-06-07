# Architecture Decision Log

This file records major CUDA optimization route decisions so future agents do not reopen stopped paths without new evidence.

## 2026-06-07

### Accepted Baseline

`CUDA3D_PML_ZMEM_IN_P` is the stable RTX 5090 baseline.

Stable flags:

```text
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DCUDA3D_PML_ZMEM_IN_P
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
```

Reason:

```text
perf_1gpu_6shots repeat speedup vs current_best_reference was about 1.049x.
Later micro-routes did not clear the 2% repeat gate.
```

### Stopped: PML Fused Z-Slab

Decision:

```text
Stop CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE.
```

Reason:

```text
Correctness passed, but repeat performance was slower than zmem_reference.
The fused kernel reduced some generic PML tile work but added enough cost to lose overall.
```

### Stopped: p_core Z-Pencil Duplicate

Decision:

```text
Do not implement CUDA3D_CORE_ZPENCIL_SHARED.
```

Reason:

```text
Source-level NCU/profiler gate showed baseline cuda_fd3d_p_core_ns already uses a z-direction shared-memory tile.
Adding a duplicate z-pencil route would repeat existing behavior.
```

### Completed: Core Two-Step Correctness Proof

Decision:

```text
Keep CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE evidence, but do not treat standalone predict/copy as a performance route.
```

Evidence:

```text
debug p(t+2) predictor: rel_l2=0, max_abs=0
commit schedule prototype: receiver output rel_l2=0, strict interior dumps rel_l2=0
```

Reason:

```text
The work proved math and scheduling feasibility for strict safe interior, but standalone prediction and copy add kernels and do not remove enough baseline work.
```

### Stopped: CTA-Local Fused Core Two-Step Commit

Decision:

```text
Stop CUDA3D_CORE_2STEP_FUSED_COMMIT_V2 before kernel implementation.
```

Evidence:

```text
Meaningful case eligible_ratio = 0.453758.
Debug-only fused p2-shift: rel_l2=0, max_abs=0.
Receiver output vs zmem_reference: rel_l2=0, max_abs=0.
Stage-4 tile budget:
  A commit/core = 0.0319
  D commit/core = 0.0320
Gate: stop if first-implementation commit ratio < 0.10.
```

Reason:

```text
R=7 erosion plus O-inside-core and non-overlapping M tile constraints leave only about 3.2% of core pressure points commit-able.
This is too little skipped work to justify retiled residual p_core and fused commit complexity.
```

### New Direction: Pressure Triple-Buffer Pipeline

Decision:

```text
Start CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE design/audit on branch exp/pressure-triple-buffer-pipeline.
```

Reason:

```text
The in-place pressure update makes p0 both old p(t-1) input and new p(t+1) output.
This creates an old/new race for naive fused commit and makes future temporal blocking fragile.
Explicit p_prev / p_curr / p_next buffers should make pressure dataflow clean before further temporal optimization.
```

Initial acceptance target:

```text
Correctness first.
perf_1gpu_6shots repeat slowdown <= 2% is acceptable for a disabled dataflow-clean baseline candidate.
Slowdown > 5% should stop implementation and trigger memory/dataflow overhead analysis.
```
