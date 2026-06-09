# PML Len16 DZ Halo Ownership Audit

This is a static exact-FP32 ownership audit. It mirrors the pressure PML tile-list split and z-cache state access ranges; it does not run CUDA kernels.

A case with zero len16 pressure tiles is marked `not_applicable_no_len16`; it can still validate residual fallback, but it does not cover the compact len16 path.

| case | len16 tiles | residual tiles | len16 write states | len16 halo reads outside writes | residual reads len16 writes | residual writes len16 writes | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `input_correctness.in` | `0` | `3736` | `0` | `0` | `0` | `0` | `not_applicable_no_len16` |
| `input_perf_1gpu_6shots.in` | `35344` | `12544` | `3393024` | `0` | `0` | `0` | `allow_compact_dz16_old_next_design` |

## Decision

A compact `memory_dz` old/next commit prototype is allowed only for cases with len16 coverage and zero len16 halo reads outside its compact write set plus zero residual overlap with len16-written z-state.
