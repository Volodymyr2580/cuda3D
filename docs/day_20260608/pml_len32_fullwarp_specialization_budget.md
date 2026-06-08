# Pressure-PML Len32 Full-Warp Specialization Budget

## Context

- sampled main: `297.248us`
- residual pressure-PML: `72.683us`
- packed len16 pressure-PML: `65.771us`
- target sampled-main speedup gate: `1.0500x`
- target saved time: `14.155us`

## Residual Shape After Len16

| item | value |
| --- | ---: |
| length-23 lines | `87776` |
| length-32 lines | `263328` |
| length-32 line share | `75.00%` |
| length-32 active-lane share | `80.67%` |
| length-32 launched-lane share | `75.00%` |

## Required Speedup

| estimate basis | full32 time | required local speedup | required local reduction |
| --- | ---: | ---: | ---: |
| line share | `54.512us` | `1.3507x` | `25.97%` |
| active-lane share | `58.635us` | `1.3182x` | `24.14%` |

## Scenario Ceilings

| scenario | saved | sampled-main speedup | gate |
| --- | ---: | ---: | --- |
| `control_proxy_on_full32_line_time` | `2.349us` | `1.0080x` | `reject` |
| `control_proxy_on_entire_residual` | `3.133us` | `1.0107x` | `reject` |
| `perfect_branch_efficiency_on_full32_active_time` | `9.780us` | `1.0340x` | `reject` |
| `perfect_branch_efficiency_on_entire_residual` | `12.123us` | `1.0425x` | `reject` |
| `twenty_percent_full32_speedup` | `11.727us` | `1.0411x` | `reject` |

## Decision

- decision: `reject_cuda_prototype`
- candidate: `CUDA3D_PML_PRESSURE_LEN32_FULL_WARP_SPECIALIZE`
- reason: Length-32 residual work has no inactive-lane saving after len16 packing.  A separate full-warp kernel would need about 1.32x-1.35x local speedup to move sampled-main by 5%, while existing branch/control proxies and even a generous perfect-branch-efficiency scenario remain below the gate.
- reopen condition: Only reopen if a source-level profile separates length-32 residual work and proves that removable branch/control or memory-ownership overhead is at least the required local reduction after extra launch and tile-list overhead.

## Boundary

This rejects a branch/control-only `len32` pressure-PML split.  It does not reject a future design that removes real memory traffic or proves a different ownership model with profiler evidence.
