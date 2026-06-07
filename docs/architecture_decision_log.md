# Architecture Decision Log

This file records CUDA3D optimization route decisions so future agents do not reopen failed paths without new evidence.

## 2026-06-08 - Start CUDA3D_WAVESTEP_ENGINE_V2

Decision:

```text
Start CUDA3D_WAVESTEP_ENGINE_V2 on branch exp/wavestep-engine-v2-pml-vp-fusion.
```

Reason:

```text
The zmem_reference baseline is stable. Further block-size, register-cap, face-split, and pressure-only split routes have not produced meaningful repeat speedup.
The remaining high-leverage path is ownership/dataflow rewrite, especially PML velocity -> pressure fusion.
```

Current stable baseline flags:

```text
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DCUDA3D_PML_ZMEM_IN_P
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
```

## 2026-06-08 - Accepted Phase 1 CPML VMEM Double Buffer

Decision:

```text
Keep CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL macro-gated and default-off.
Use it as the ownership-clean scaffold for PML fused VP.
Proceed to PML_REGION_FUSED_VP_ZFACE_ONLY design/prototype.
```

Implemented macros:

```text
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_CPML_VMEM_DEBUG_FILL
CUDA3D_CPML_VMEM_DISABLE_MPI
```

Evidence:

```text
debug fill smoke/correctness: pass
debug dump step 0/1/2 vs zmem_reference: pass
release smoke/correctness/perf6/perf6_repeat comparisons: pass
all output rel_l2: <= 1e-5
```

Performance A/B:

```text
Phase1 perf6 mean WP:        2.365721s
ZMEM pre mean WP:            2.450038s, speedup 1.035641x
ZMEM post mean WP:           2.435677s, speedup 1.029570x
ZMEM all mean WP:            2.442857s, speedup 1.032605x
ZMEM all mean Gradient:      2.555540s
Phase1 all mean Gradient:    2.484369s, speedup 1.028648x
```

Report:

```text
reports/wavestep_engine_v2_phase1_cpml_vmem_20260608_003000/phase1_report.md
```

## Stop List Still Applies

Do not invest in:

```text
PML zface pressure-only split
PML fused z-slab pressure-only split
p_core z-pencil duplicate
CTA-local core two-step fused commit
standalone predict/copy tuning
block-size sweep
register cap sweep
RECOMPUTE_X/Y/XYZ global recompute
naive in-place pressure fusion
full MPI temporal blocking
```

Next allowed route:

```text
CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY
```

Hard gate:

```text
meaningful case repeat speedup >= 10%
perf_1gpu_6shots repeat speedup >= 5%
```

## 2026-06-08 - Stop Direct PML Fused VP Z-Face

Decision:

```text
Stop CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY in its direct p1 second-derivative form.
Do not repeat this route without new profiler evidence.
```

Implemented but rejected variants:

```text
1. Separate fused z-face kernel:
   v_pml skips fused-owned vx/vy writes; p_pml skips fused points; a separate zface kernel updates p0.

2. Inline p_pml fused branch:
   v_pml skips fused-owned vx/vy writes; p_pml_tile handles zface points directly without an extra kernel launch.
```

Correctness evidence:

```text
smoke: pass
correctness: pass
perf_1gpu_6shots repeat output compare: pass
max perf6 repeat rel L2: 6.358816e-07
```

Performance evidence on RTX 5090 same-session A/B:

```text
zmem mean WP:              2.434461s
separate zface mean WP:    2.660077s, speed ratio 0.915184x
inline zface mean WP:      2.692579s, speed ratio 0.904137x
```

Reason:

```text
The direct z-face fusion is mathematically valid but replaces vx/vy global reads with additional p1 loads, extra arithmetic, and branch pressure.
The saved vx/vy round trip is not enough to overcome this cost on the perf_1gpu_6shots gate.
```

Stop rule:

```text
Do not continue direct p1 x/y second-derivative z-face fusion.
Only reopen PML z-face fusion if the design keeps velocity intermediates CTA-local with shared-memory reuse or has Nsight Compute evidence showing lower total memory stalls.
```

Report:

```text
reports/wavestep_engine_v2_phase2_fused_zface_20260608_010000/phase2_fused_zface_report.md
```
