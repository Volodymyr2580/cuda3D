# PML Compact Descriptor Budget

## Context

- accepted len16 lanes: `19908928`
- active lanes: `19118944`
- post-len16 pressure-PML sampled-main share: `46.58%`
- observed direct-fill -> len16 pressure-PML speedup: `1.1869x`
- direct-fill -> len16 lane ceiling: `1.4638x`
- observed lane-to-time efficiency factor: `0.811`

## NCU Anchor

| metric | direct-fill | len16 |
| --- | ---: | ---: |
| pressure-PML total | `164.328us` | `138.453us` |
| sampled main total | `323.608us` | `297.248us` |
| p-core | `93.752us` | `93.547us` |
| v-PML | `65.528us` | `65.248us` |

## Remaining Lane Opportunity

| segment | lines | active lanes | inactive lanes after len16 |
| --- | ---: | ---: | ---: |
| length 16 | 542100 | 8673600 | 0 |
| length 23 | 87776 | 2018848 | 789984 |
| length 32 | 263328 | 8426496 | 0 |

## Candidate Ceilings Vs Accepted Len16

| candidate | lanes | lane reduction | p-PML lane ceiling | sampled-main ceiling | calibrated sampled-main | descriptor MiB/step | bytes/saved lane |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `line_descriptor_len16_current` | 19908928 | 0.00% | 1.0000x | 1.0000x | 1.0000x | 6.815 | n/a |
| `exact_length23_points_only` | 19119168 | 3.97% | 1.0413x | 1.0188x | 1.0153x | 7.701 | 10.23 |
| `exact_all_active_points` | 19119104 | 3.97% | 1.0413x | 1.0188x | 1.0153x | 72.933 | 96.83 |

## Gate

- decision: `reject_cuda_prototype`
- reason: After the accepted len16 packing, exact active-point compaction can only remove the remaining length-23 inactive lanes.  The optimistic sampled-main ceiling is below the >=5% prototype gate, and the required descriptor stream risks adding more memory/control overhead than the saved inactive lanes.
- reopen condition: Only reopen compact descriptors if a new design shows >=5% perf_1gpu_6shots repeat speedup ceiling after descriptor/control overhead.

Allowed next directions:

- source-level drill-down of cuda_fd3d_p_pml_len16_halfwarp_ns
- v-PML memory layout/coalescing design
