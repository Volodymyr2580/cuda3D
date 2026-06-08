# P-Core Shared-Plane Calibrated Gate

## Calibration Anchor

- tested shape: `[16, 16, 1]` / `zx_shared_y_global`
- p_core share: `31.47%`
- modeled p_core local speedup: `1.5651x`
- modeled sampled-main speedup: `1.1282x`
- observed WP global speedup: `0.7845x`
- observed Gradient global speedup: `0.7893x`
- inferred WP-local p_core speedup: `0.5339x`
- inferred Gradient-local p_core speedup: `0.5411x`
- WP model-to-observed factor: `0.3411x`
- Gradient model-to-observed factor: `0.3457x`

## Top Calibrated Candidates

| shape | mode | model sampled | calibrated WP sampled | calibrated Gradient sampled | shared KiB |
| --- | --- | ---: | ---: | ---: | ---: |
| `[16, 16, 1]` | `zx_shared_y_global` | `1.1282x` | `0.7845x` | `0.7893x` | `3.52` |
| `[32, 8, 1]` | `zx_shared_y_global` | `1.1228x` | `0.7768x` | `0.7817x` | `3.95` |
| `[16, 8, 2]` | `zx_shared_y_global` | `1.1081x` | `0.7565x` | `0.7614x` | `5.16` |
| `[8, 16, 2]` | `zx_shared_y_global` | `1.1081x` | `0.7565x` | `0.7614x` | `5.16` |
| `[64, 4, 1]` | `zx_shared_y_global` | `1.1042x` | `0.7511x` | `0.7560x` | `5.48` |
| `[32, 4, 2]` | `zx_shared_y_global` | `1.0925x` | `0.7355x` | `0.7404x` | `6.47` |
| `[8, 8, 4]` | `zx_shared_y_global` | `1.0799x` | `0.7189x` | `0.7238x` | `7.56` |
| `[64, 2, 2]` | `zx_shared_y_global` | `1.0554x` | `0.6878x` | `0.6927x` | `9.75` |
| `[64, 2, 2]` | `zy_shared_x_global` | `1.0554x` | `0.6878x` | `0.6927x` | `9.75` |
| `[32, 4, 2]` | `zy_shared_x_global` | `1.0367x` | `0.6648x` | `0.6697x` | `11.50` |

## Decision

- decision: `reject_current_shared_plane_family`
- reason: The failed 16x16x1 prototype shows shared-fill/control/warp-mapping overhead overwhelms the byte-model savings.  Applying that empirical calibration pulls every current shared-plane candidate below the >=5% gate.
- reopen condition: Reopen only with a materially different warp/coalescing design and a source/profile model that separately accounts for shared fill, synchronization, and control overhead.
