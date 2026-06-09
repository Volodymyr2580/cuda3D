# PML Len16 Compact DZ Old/Next Design

## Status

- Route: exact-FP32 single-GPU CUDA optimization.
- Design name: `CUDA3D_PML_LEN16_COMPACT_DZ16_OLD_NEXT`.
- Current status: design-only gate passed by static halo ownership audit.
- Baseline: `current_best_v_pml_len16`.

This design does not use relaxed precision.

## Prerequisites

Required flags:

```text
CUDA3D_PML_RECOMPUTE_Z
CUDA3D_PML_ZMEM_IN_P
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
CUDA3D_PML_LEN16_COMPACT_STATE
```

`CUDA3D_PML_LEN16_COMPACT_STATE` is the existing dzz16-only path. It is a
negative standalone result, but it remains the scaffold for the deeper compact
state prototype.

## Ownership Result

The static audit:

```text
tools/pml_len16_dz_halo_ownership_audit.py
```

shows that `perf_1gpu_6shots` has:

- len16 halo reads outside compact write set: `0`,
- residual reads of len16-written z-state: `0`,
- residual writes of len16-written z-state: `0`.

Therefore accepted pressure len16 tiles can own their 16 central `memory_dz`
state entries without residual pressure PML consuming stale full-array values.

The `correctness` case has zero pressure len16 tiles, so it validates only
residual fallback and macro wiring; it does not validate compact len16 state.

## Compact State

Add two compact buffers:

```text
compact_dz_old16
compact_dz_next16
```

Indexing matches the dzz16 compact index:

```text
compact_line = blockIdx.x * PmlTileBlockSize2 * PmlTileBlockSize3 + local_line
compact_z16 = compact_line * 16 + local_z
```

For accepted pressure len16 central positions:

```text
new_mem = compact_dz_old16[compact_z16] * coef + value * (coef - 1)
compact_dz_next16[compact_z16] = new_mem
value += new_mem
```

Halo z-cache entries may keep the existing full-array helper with
`write_owned=false`. Under the audited topology those halo calls do not touch
valid PML z-state. Keeping the fallback preserves exact behavior if a future
case violates the current topology assumption.

## Buffer Swap

At the same point where full `d_memory_dz` and `d_memory_dz_next` are swapped,
swap:

```text
d_p_len16_compact_dz_old16
d_p_len16_compact_dz_next16
```

The compact next buffer is fully written by accepted pressure len16 central
positions every timestep. A debug-only fill/check path should be added before
promotion:

- fill compact next with a non-finite sentinel before pressure update,
- verify all compact next entries are finite before swap,
- fail fast on any unwritten entry.

## Full-Array Fallback

Residual pressure PML remains full-array authoritative.

Normal performance builds must not duplicate accepted len16 central writes into
full `memory_dz_next`, because that would erase most of the intended traffic
reduction. Debug builds may mirror compact state back to full arrays before
dumping so existing debug dump comparison tools remain useful.

## Validation Gate

Required before keeping the prototype:

- build current-best flags plus `CUDA3D_PML_LEN16_COMPACT_DZ16_OLD_NEXT`,
- build current-best flags with the new macro disabled,
- debug dump or equivalent profile case covering len16 path at step `0/1/2`,
- `correctness` output compare rel L2 `<=1e-5`,
- `perf_1gpu_6shots` output compare rel L2 `<=1e-5`,
- `perf_1gpu_6shots` repeat x3.

Promotion rules:

- WP speedup `<1.02x`: reject and keep macro default-off.
- WP speedup `>=1.02x` but `<1.05x`: keep as disabled candidate only if
  profiler evidence shows a clear next step.
- WP speedup `>=1.05x` and Gradient speedup positive: eligible for current-best
  review.

## Stop Conditions

Stop immediately if:

- compact next coverage check reports any unwritten entry,
- debug dump step `0/1/2` exceeds tolerance,
- output comparison exceeds relative L2 `1e-5`,
- profiler shows added control or descriptor overhead dominates the removed
  state traffic.
