# Residual Pressure-PML Route Gate

## Context

- sampled main: `284.010us`
- residual pressure-PML: `71.940us` / `25.33%`
- target sampled-main gate: `1.0500x`
- required residual local speedup: `1.2315x`
- required residual local reduction: `18.80%`
- required saved time: `13.524us`

## Residual NCU Anchor

| metric | value |
| --- | ---: |
| No Eligible | `63.162%` |
| eligible warps/scheduler | `0.766` |
| warp cycles/issued inst | `23.682` |
| avg active threads/warp | `23.050` |
| avg not-predicated threads/warp | `21.730` |
| branch efficiency | `83.750%` |
| L1/TEX hit | `64.758%` |
| L2 hit | `59.332%` |
| achieved occupancy | `73.389%` |

## Scenario Ceilings

| scenario | local speedup | sampled-main speedup | interpretation |
| --- | ---: | ---: | --- |
| `perfect_branch_efficiency` | `1.1940x` | `1.0429x` | upper bound for branch/control specialization without removing memory traffic |
| `recover_predicated_off_threads_only` | `1.0607x` | `1.0147x` | upper bound for predicate cleanup while keeping same active-lane shape |
| `recover_all_active_threads_to_full_warp` | `1.3883x` | `1.0762x` | utopian lane-utilization bound; descriptor/control overhead not counted |
| `recover_all_not_predicated_threads_to_full_warp` | `1.4726x` | `1.0885x` | more utopian bound combining active-lane and predication cleanup |
| `twenty_percent_residual_reduction` | `1.2500x` | `1.0534x` | minimum ballpark residual reduction needed to cross the gate |
| `exact_length23_descriptor_calibrated` | `n/a` | `1.0153x` | existing calibrated descriptor budget after accepted pressure len16 |

## Descriptor Prior

- existing compact descriptor decision: `reject_cuda_prototype`
- length-23 calibrated sampled-main speedup: `1.0153x`
- length-23 lane reduction after accepted len16: `3.97%`

## Gate

- decision: `reject_residual_pressure_micro_cuda_prototype`
- reason: Residual pressure-PML would need about 18.80% local time reduction (13.524us) to move sampled-main by >=5%.  The current profile already has branch efficiency 83.75% and avg active threads/warp 23.05; branch/predicate cleanup stays below the gate, while lane/descriptor compaction was already calibrated at only about 1.0153x sampled-main speedup.

Allowed next:

- pressure/wave-step ownership model that removes real pressure writeback or CPML state traffic
- cross-CTA or cluster-level ownership study only if a concrete synchronization primitive is identified
- precision-relaxation study only after an explicit tolerance-policy change
- external handoff/report summarizing current hard gates

Prohibited:

- residual pressure branch-only split
- length-32 branch/control specialization retry
- length-23 or exact active-point descriptor retry
- residual p0 __ldg/local-new_mem/cache-policy/z-cache micro-tuning
