# PML Len16 Compact DZZ16 Commit Result

## Scope

Exact-FP32 default-off prototype `CUDA3D_PML_LEN16_COMPACT_STATE` routes only accepted pressure len16 `memory_dzz` through compact `dzz16` state. Full `memory_dz` old/next remains authoritative.

## Final Build

- binary SHA256: `3bba4756a8bac7493be1c50455e39f7eb3c8a4c92dba44f393d66ed1eb41e094`
- debug-only mirror-back exists only when `CUDA3D_PML_DEBUG_DUMP` is also enabled; normal performance build does not write full `memory_dzz` for accepted len16.

## Correctness

- correctness compare pass: `True`
- max rel L2: `0.000000e+00`

## Perf Repeat

- baseline mean WP: `2.004982s`
- candidate mean WP: `1.981516s`
- WP speedup: `1.011842x`
- baseline mean Gradient: `2.118638s`
- candidate mean Gradient: `2.093934s`
- Gradient speedup: `1.011798x`
- all perf output compare pass: `True`

| tag | WP s | Gradient s | compare | max rel L2 |
| --- | ---: | ---: | ---: | ---: |
| `a` | `1.976281` | `2.092469` | `True` | `0.000000e+00` |
| `b` | `1.984154` | `2.094078` | `True` | `0.000000e+00` |
| `c` | `1.984113` | `2.095256` | `True` | `0.000000e+00` |

## Decision

Reject as a performance candidate because repeat WP speedup is below the `>=1.02x` disabled-candidate keep gate. Keep the macro default-off as a tested exact-FP32 negative result.

## Next Step

Do not expand this dzz16-only path. If compact state continues, the next design must prove safe handling of `memory_dz` old/next halo ownership before writing a commit prototype.
