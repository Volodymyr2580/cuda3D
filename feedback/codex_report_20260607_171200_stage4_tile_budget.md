# Codex Feedback To Pro: Stage-4 Tile Budget Gate

Date: 2026-06-07 17:12 +0800

## Executive Summary

I implemented the requested Stage-4 Architecture Gate before writing any commit kernel.

Result:

```text
tools/core2step_tile_budget.py: done
docs/core_2step_stage4_tile_budget.md: done
A/D first-implementation gate: failed
decision: stop before CUDA3D_CORE_2STEP_FUSED_COMMIT_V2
```

The gate failed because the actual non-overlapping CTA-local commit ratio is only about `3.2%` of core pressure points for both A and D. This is far below the requested `10%` stop threshold.

## Case

Case:

```text
benchmarks/cases/core_2step_meaningful_1gpu
```

Geometry:

```text
ny=160 nx=160 nz=96 nt=501 npml=12
core_region=z:[16,104),x:[16,168),y:[16,168)
core_points=2033152
source_zyx=(13,24,24)
receivers=49
```

The source/receivers remain outside all proposed O/C regions for the generated M tile lists.

## Budget Results

Definition:

```text
R = 7
M = first-step owned / staged tile
O = M + R halo on each side
C = M - R margin on each side
shared = p_next_local over M only
threads = 256
sm_120 budget = 128 KiB/SM, 99 KiB/block, 48 warps/SM
```

Results:

```text
A: M=[32,24,20], C=[18,10,6], kept_tiles=60, commit_points=64800, commit/core=0.0319, CTA/SM est=2
B: M=[32,24,24], C=[18,10,10], kept_tiles=50, commit_points=90000, commit/core=0.0443, CTA/SM est=1
C: M=[40,24,20], C=[26,10,6], kept_tiles=30, commit_points=46800, commit/core=0.0230, CTA/SM est=1
D: M=[40,24,24], C=[26,10,10], kept_tiles=25, commit_points=65000, commit/core=0.0320, CTA/SM est=1
E: M=[48,20,24], C=[34,6,10], kept_tiles=30, commit_points=61200, commit/core=0.0301, CTA/SM est=1
```

First-implementation candidates:

```text
A commit ratio = 3.19%
D commit ratio = 3.20%
```

Gate:

```text
if A and D both commit_ratio < 10%, stop fused commit
```

Decision:

```text
STOP
```

## Interpretation

The debug predictor proved that `p(t+2)` is mathematically predictable in a strict safe interior. But the CTA-local shared-memory version loses too much volume to the `R=7` erosion.

Even D has a decent per-tile `C/M = 11.28%`, but the meaningful case can only fit `25` non-overlapping M tiles under the O-inside-core constraint. The total committed C points are therefore only `65000 / 2033152 = 3.20%`.

This is not enough skipped work to pay for:

```text
fused commit kernel complexity
d_p2_fused storage
commit copy or in-place commit cost
retiled residual p_core machinery
additional launch/synchronization overhead
NCU/tuning cost
```

## Decision For Codex

I did not implement:

```text
CUDA3D_CORE_RETILED_RESIDUAL
CUDA3D_CORE_2STEP_FUSED_COMMIT_V2
```

Reason:

```text
Stage 4.0 gate failed before Stage 4.1.
```

This follows the instruction:

```text
If A and D both commit ratio <10%, write report and stop.
```

## Updated Project Route

Updated:

```text
AGENTS.md
```

New rule:

```text
Do not implement CUDA3D_CORE_2STEP_FUSED_COMMIT_V2 unless a new tile plan or gate is provided.
```

Recommended next route:

```text
CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE
```

This route would explicitly split:

```text
p_prev / p_curr / p_next
```

and remove the in-place `p0` old/new race at the system dataflow level. It is a larger rewrite and should begin with a design doc and pointer-swap/PML/source/receiver audit, not direct kernel hacking.

## Artifacts

Local/Git artifacts:

```text
tools/core2step_tile_budget.py
docs/core_2step_stage4_tile_budget.md
benchmarks/reports/core2step_stage4_tile_budget_local.json
feedback/codex_report_20260607_171200_stage4_tile_budget.md
```

The JSON report under `benchmarks/reports/` is generated output and may stay untracked unless we decide to archive this gate formally.
