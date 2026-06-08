# P-Core Shared-Plane Stencil Budget

## Context

- sampled main: `297.248us`
- p_core: `93.547us` / `31.47%`
- p_core L2 SOL: `96.89%`
- p_core L2 hit rate: `86.34%`
- current p_core block z/x/y: `[128, 2, 1]`

## Current Byte Model

- current p1 global floats/output: `29.109375`
- current bytes/output including p0/cw2/store: `128.438`
- z shared load contribution: `1.109375` floats/output
- x/y global neighbor contribution: `28.000000` floats/output

## Top Candidates

| candidate | mode | p1 floats/out | bytes/out | shared KiB | p_core byte ceiling | sampled-main ceiling |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `[16, 16, 1]` | `zx_shared_y_global` | `17.516` | `82.062` | `3.52` | `1.5651x` | `1.1282x` |
| `[32, 8, 1]` | `zx_shared_y_global` | `17.953` | `83.812` | `3.95` | `1.5324x` | `1.1228x` |
| `[16, 8, 2]` | `zx_shared_y_global` | `19.156` | `88.625` | `5.16` | `1.4492x` | `1.1081x` |
| `[8, 16, 2]` | `zx_shared_y_global` | `19.156` | `88.625` | `5.16` | `1.4492x` | `1.1081x` |
| `[64, 4, 1]` | `zx_shared_y_global` | `19.484` | `89.938` | `5.48` | `1.4281x` | `1.1042x` |
| `[32, 4, 2]` | `zx_shared_y_global` | `20.469` | `93.875` | `6.47` | `1.3682x` | `1.0925x` |
| `[8, 8, 4]` | `zx_shared_y_global` | `21.562` | `98.250` | `7.56` | `1.3073x` | `1.0799x` |
| `[64, 2, 2]` | `zx_shared_y_global` | `23.750` | `107.000` | `9.75` | `1.2004x` | `1.0554x` |
| `[64, 2, 2]` | `zy_shared_x_global` | `23.750` | `107.000` | `9.75` | `1.2004x` | `1.0554x` |
| `[32, 4, 2]` | `zy_shared_x_global` | `25.500` | `114.000` | `11.50` | `1.1266x` | `1.0367x` |

## Decision

- decision: `allow_cuda_prototype`
- candidate: `CUDA3D_P_CORE_SHARED_ZX_PLANE`
- reason: A z+x shared-plane p_core kernel can reduce estimated p1 global loads enough to pass the >=5% sampled-main modeling gate.  This is a dataflow change, not another simple block/register sweep.

## Guardrails

- macro default-off
- preserve current second-order p0/p1/cw2 math
- profile/debug/correctness/perf_1gpu_6shots repeat before acceptance
- stop immediately if repeat speedup <5% or correctness exceeds tolerance
