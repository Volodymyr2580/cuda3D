# Precision Tracks Policy

## Purpose

CUDA3D may now explore relaxed precision, but the exact-FP32 line remains the
gold standard.  Relaxed precision must be isolated so performance gains are not
confused with exact CUDA-code optimization.

## Tracks

### Track A: exact-FP32

- Branch family: `exp/pml-len16-compact-state` and later exact optimization
  branches.
- Storage: FP32 for pressure, velocity, CPML state, source/receiver data, and
  output comparison.
- Baseline: `current_best_v_pml_len16`.
- Default correctness gate: relative L2 `<= 1e-5`, no NaN/Inf.
- Speedups from this track may be called CUDA code optimization.

### Track B: relaxed precision

- Branch family: `exp/relaxed-precision-*`.
- Macro naming must start with `CUDA3D_RELAXED_`.
- The exact-FP32 current-best binary and outputs must remain available for
  every comparison.
- Reports must include:
  - relaxed flags and exact flags,
  - output max relative L2,
  - output max absolute error,
  - NaN/Inf check,
  - per-output error table,
  - WP/Gradient/elapsed speedup,
  - explicit acceptance tier.
- Speedups from this track must be called relaxed-precision speedups, not exact
  CUDA-code speedups.

## Tolerance Tiers

| Tier | Use | Relative L2 | Notes |
| --- | --- | ---: | --- |
| Tier 0 | exact-FP32 default | `<= 1e-5` | Current publication-safe gate. |
| Tier 1 | mildly relaxed | `<= 5e-5` | Candidate only if output waveforms remain visually and statistically stable. |
| Tier 2 | aggressive relaxed | `<= 1e-4` | Requires explicit user acceptance before promotion. |

No relaxed-precision candidate may replace the exact-FP32 current-best tag.

## Current Decision

As of 2026-06-09, the next active single-GPU sprint remains exact-FP32:

```text
PML len16 compact-state ownership prototype
```

Relaxed precision is allowed as a separate design and experiment line after the
compact-state gate, or earlier only if exact-FP32 audit shows no meaningful
implementation path.
