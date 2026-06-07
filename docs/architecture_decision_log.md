# Architecture Decision Log

This file records major CUDA optimization route decisions so future agents do not reopen stopped paths without new evidence.

## 2026-06-07

### Accepted Baseline

`CUDA3D_PML_ZMEM_IN_P` is the stable RTX 5090 baseline.

Stable tag:

```text
stable-zmem-rtx5090-20260607 -> ea091c5e97b9d00e9f4d7847e5ada1f884de0cab
```

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

Same-machine formal rerun:

```text
reports/formal_speed_table_20260607_215057/formal_speed_table.md
```

Formal RTX 5090 results, using identical cases and source with only compile macros changed:

```text
default_no_macro -> current_best_reference:
  WP speedup       1.082255x
  Gradient speedup 1.080468x

default_no_macro -> zmem_reference:
  WP speedup       1.123747x
  Gradient speedup 1.123155x

current_best_reference -> zmem_reference:
  WP speedup       1.038339x
  Gradient speedup 1.039507x
```

All formal output comparisons passed with max rel L2 `0`.

Note: `default_no_macro` is the available original-like/no-macro path in the current source tree. It is not proof of a pristine upstream original tarball.

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

### Completed: Pressure Triple-Buffer Standalone Prototype

Decision:

```text
Keep CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE macro-gated and default-off.
Do not promote it to zmem_reference.
Do not continue standalone triple-buffer micro-optimization.
```

Evidence:

```text
smoke_1gpu: pass, rel_l2=0
correctness: pass, rel_l2=0
perf_1gpu_6shots output compare: pass, rel_l2=0

perf_1gpu A/B:
  WP speedup       about 1.0208x
  Gradient speedup about 1.0182x

perf_1gpu_6shots repeat A/B:
  WP speedup       about 1.0045x
  Gradient speedup about 1.0063x
```

Reason:

```text
Triple buffer fixed the pressure dataflow alias and gives a clean p_prev/p_curr/p_next foundation.
It does not remove enough work by itself to clear the 2% repeat gate on the meaningful 6-shot case.
Its value is architectural, not standalone speed.
```

Report:

```text
reports/triple_buffer_3h/final_3h_report.md
```

### Consolidated Stop List

Decision:

```text
Do not reopen these directions unless new profiler evidence directly contradicts the recorded reason.
```

Stopped or forbidden:

```text
CUDA3D_PML_ZMEM_V_TILE_PRUNE
CUDA3D_PML_TILE_MASK_FASTPATH
CUDA3D_PML_ZFACE_P_SPECIALIZE
CUDA3D_PML_ZFACE_V_SPECIALIZE
CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE
CUDA3D_PML_FUSED_ZSLAB_SKIP_V_OWNED
RECOMPUTE_X / RECOMPUTE_Y / RECOMPUTE_XYZ
PML tile block shape sweep
p_core simple block shape sweep
-maxrregcount / register cap sweep
standalone predict+copy micro tuning
simple CUDA Graph replay
simple static memory pool
simple v_pml + p_pml fusion without grid-wide synchronization
full-domain temporal blocking
MPI temporal blocking
```

Reason:

```text
These routes either already failed correctness, failed the repeat performance gate, duplicated existing baseline behavior, or produced too little eligible work to justify implementation complexity.
```

### Accepted Next Gate: Phase2 Triple-Buffer Temporal Pipeline

Decision:

```text
Start a smaller phase2 only for triple-buffer-based temporal pipeline work.
No other optimization family is allowed in phase2.
```

Hard gate:

```text
single GPU / single MPI rank first
zmem_reference remains the baseline
correctness rel_l2 <= 1e-5 and finite outputs required
meaningful perf_1gpu_6shots repeat WP speedup >= 5% required
if the meaningful case is <5%, stop immediately and write a report
```

Details:

```text
docs/phase2_temporal_pipeline_gate.md
```
