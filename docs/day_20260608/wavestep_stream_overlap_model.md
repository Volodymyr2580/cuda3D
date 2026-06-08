# Wave-Step Stream Overlap Model

## Context

- profile: `len16`
- sequential sampled main: `297.248us`
- p_core: `93.547us`
- v_pml: `65.248us`
- pressure residual: `72.683us`
- pressure len16: `65.771us`
- pressure total: `138.453us`
- target sampled-main speedup: `1.0500x`

Dependency facts:

- p_core reads p1/cw2 and writes p0 only in the core region.
- v_pml reads p1 and writes vx/vy plus CPML velocity state.
- pressure PML reads vx/vy and writes p0 only in PML regions.
- source injection/extraction must wait for both core and PML streams before pointer swap reuse.

## Candidates

| candidate | critical path | ceiling | overlap window | required overlap for target |
| --- | ---: | ---: | ---: | ---: |
| `overlap_core_with_v_only` | `232.000us` | 1.2812x | `65.248us` | 21.69% |
| `overlap_core_with_serial_pml_path` | `203.701us` | 1.4592x | `93.547us` | 15.13% |
| `overlap_core_with_parallel_pressure_pml` | `137.931us` | 2.1551x | `93.547us` | 15.13% |

## Gate

- decision: `allow_cuda_prototype`
- recommended first prototype: `overlap_core_with_serial_pml_path`
- prototype macro: `CUDA3D_WAVESTEP_ASYNC_STREAMS`
- best sampled-main ceiling: `2.1551x`
- reason: The conservative two-stream schedule only needs about 15.13% realized overlap to reach 1.0500x sampled-main speedup, while preserving kernel math and ownership.

Prototype boundaries:

- macro default off
- do not change CUDA math kernels
- do not overlap next time step before injection/extract completes
- first prototype uses two streams only; pressure residual/len16 parallel fanout requires a later gate
