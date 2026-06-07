# Phase2 Triple-Buffer Temporal Pipeline Gate

Date: 2026-06-07

## Decision

Phase2 is allowed to continue only as a triple-buffer-based temporal pipeline effort.

`zmem_reference` remains the stable baseline. The standalone triple-buffer prototype is not a speed baseline; it is a correctness-checked dataflow scaffold.

## Allowed Work

- Single GPU / single MPI rank first.
- Use explicit `p_prev / p_curr / p_next` pressure roles.
- Keep `CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE` macro-gated and default-off.
- Add phase2 macros separately, for example `CUDA3D_PHASE2_TRIPLE_TEMPORAL_PIPELINE`.
- Preserve source injection, receiver extraction, PML memory swap, and final pressure-buffer rotation semantics.
- Compare every candidate against `zmem_reference`.

## Hard Stop

Stop immediately if the meaningful case does not reach:

```text
perf_1gpu_6shots repeat WP speedup >= 5% vs zmem_reference
```

Correctness requirements remain:

```text
output count and byte sizes identical
all values finite
relative L2 <= 1e-5
```

## Forbidden In Phase2

- PML face split.
- PML tile/mask/prune/block-shape sweep.
- `RECOMPUTE_X`, `RECOMPUTE_Y`, or `RECOMPUTE_XYZ`.
- p_core simple block-shape sweep.
- `-maxrregcount` / register cap sweep.
- Simple CUDA Graph replay.
- Simple static memory pool.
- Full-domain temporal blocking.
- MPI temporal blocking.
- Any candidate that changes numerical formulas before passing debug/correctness gates.

## First Prototype Shape

The first phase2 prototype should be intentionally small:

```text
triple-buffer pressure dataflow
one carefully selected strict interior temporal region
single-rank only
debug dumps or pointwise comparison for the first 2-3 steps
correctness before perf
perf_1gpu_6shots repeat before any expansion
```

The goal is to prove that the explicit pressure-buffer pipeline can remove enough repeated pressure work in a meaningful case. If the first meaningful repeat result is below 5%, the phase ends with a report rather than another sweep.

