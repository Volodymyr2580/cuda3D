# P-Core ZX Shared-Plane Prototype Result

## Decision

- decision: `reject_cuda_prototype`
- candidate: `CUDA3D_P_CORE_SHARED_ZX_PLANE`
- reason: correctness passed, but perf repeat is a large slowdown; shared-plane fill/control/warp mapping overhead outweighed modeled p1 traffic savings.

## Perf Repeat

| metric | value |
| --- | ---: |
| mean WP | `2.589956s` |
| mean Gradient | `2.731247s` |
| WP speedup vs current-best | `0.784474x` |
| Gradient speedup vs current-best | `0.789347x` |
| all output compares pass | `True` |
| max rel L2 | `0.000000e+00` |

## Rows

| round | WP | Gradient | pass | max rel L2 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `2.589493` | `2.731454` | `True` | `0.000000e+00` |
| 2 | `2.597138` | `2.734513` | `True` | `0.000000e+00` |
| 3 | `2.583236` | `2.727773` | `True` | `0.000000e+00` |

## Boundary

Do not continue the current `16x16x1` z+x shared-plane p_core prototype. Reopen p_core shared-plane work only with a new warp/coalescing design that proves lower shared-fill/control overhead before coding.
