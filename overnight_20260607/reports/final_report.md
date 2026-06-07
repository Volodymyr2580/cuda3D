# Overnight CUDA Optimization Report 2026-06-07

## Decision

Final binary strategy: **zmem_reference**.

Reason: no candidate after zmem reached the required >=2% repeat speedup. The zmem strategy itself reproduced a stable gain over `current_best_reference`, so `/work/wenzhe/cuda3D/bin/cuda_3D_FM` was rebuilt with the zmem flags.

Binary SHA256:

```text
db55f6505d3bf2460a07028056a3b00da8bf0b884ffb69a250b2d1bd023ab488  bin/cuda_3D_FM
```

## Baseline vs ZMEM

| Variant | perf6 WP | perf6 Gradient | perf6_repeat WP | perf6_repeat Gradient |
|---|---:|---:|---:|---:|
| current_best_reference | 2.507059 | 2.632964 | 2.508503 | 2.632298 |
| zmem_reference | 2.393577 | 2.514862 | 2.390644 | 2.514458 |

Speedup vs current_best_reference:

- perf6 WP: 1.047411x
- perf6 Gradient: 1.046962x
- perf6_repeat WP: 1.049300x
- perf6_repeat Gradient: 1.046865x

## Tested Routes

| Stage | Candidate | Correctness | Performance Result | Decision |
|---|---|---|---|---|
| Stage1 | `CUDA3D_PML_ZMEM_V_TILE_PRUNE` | pass, rel_l2=0 | no tiles pruned; slower/no benefit | reject for final |
| Stage2 | `CUDA3D_PML_TILE_MASK_FASTPATH` | pass, rel_l2=0 | slower than zmem | reject for final |
| Stage3 | PML tile block sweep | all compared candidates pass | best repeat `32x2x4` WP speedup vs zmem_repeat 0.989123x | reject for final |
| Stage4 | p_core block sweep | all compared candidates pass | best repeat `128x1x2` WP speedup vs zmem_repeat 1.001792x | reject for final |
| Stage4b | combo/register sweep | all compared candidates pass | no candidate beats zmem; `-maxrregcount` slower | reject for final |

## Final Validation

- Final correctness status: `ok`
- Final correctness compare: `True`
- Final smoke ALL DONE: `True`
- Final smoke WP: `0.002515` s
- Final smoke Gradient: `0.003288` s
- Final smoke run: `/work/wenzhe/cuda3D/benchmarks/runs/smoke_1gpu_final_zmem_smoke_20260607_20260607_011703`

## Artifacts

- Machine summary: `overnight_20260607/reports/final_summary.json`
- Best env: `overnight_20260607/reports/best_variant.env`
- Failed variants: `overnight_20260607/reports/failed_variants.md`
- Next steps: `overnight_20260607/reports/next_steps.md`
- Stage summaries: `stage3_block_sweep_summary.json`, `stage4_pcore_sweep_summary.json`, `stage4b_combo_reg_sweep_summary.json`

## Notes

`include/inc3D/cu_common.h` now wraps block-size macros with `#ifndef`, preserving default behavior while allowing future sweeps through `NVFLAGS`. Experimental Stage1/Stage2 code is macro-gated and not enabled in the final binary.
