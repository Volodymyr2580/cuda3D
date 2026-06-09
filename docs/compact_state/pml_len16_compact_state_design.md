# PML Len16 Compact-State Design

## Status

- Route: exact-FP32 single-GPU CUDA optimization.
- Design name: `CUDA3D_PML_LEN16_COMPACT_STATE`.
- Current gate: allowed to proceed to debug mirror after Phase 1 audit.
- Baseline: `current_best_v_pml_len16`.

This design does not use relaxed precision.

## Why This Route Is Still Exact-FP32

The compact-state route changes memory ownership and indexing for selected CPML
state arrays.  It must not change:

- pressure field storage,
- velocity field storage,
- CPML state precision,
- stencil order,
- source injection,
- receiver extraction,
- MPI behavior.

All state remains `float`.  Output comparison remains against exact-FP32
`current_best_v_pml_len16` with relative L2 `<= 1e-5`.

## Scope

Allowed:

- accepted pressure len16 PML line path,
- `memory_dzz` state update for pressure len16,
- `memory_dz` / `memory_dz_next` read window used by pressure z-recompute cache,
- descriptor-side compact base offsets for len16 tiles,
- debug mirror against full-array state.

Not allowed:

- compact pressure wavefields `p0/p1/cw2`,
- compact `vx/vy/vz` arrays,
- residual non-len16 PML path rewrite,
- relaxed precision,
- direct `p1` second derivative replacement,
- z-face VP fusion,
- shared VP retry,
- pressure branch-only split,
- length-23 or length-32 descriptor retry.

## Existing Ownership

Current accepted pressure len16 kernel:

```text
cuda_fd3d_p_pml_len16_halfwarp_ns
```

Mapping:

- one CUDA block owns one accepted len16 tile,
- one warp row processes two active z-lines,
- each active z-line has 16 active z positions,
- `threadIdx.x` maps to the 16 active z positions,
- `threadIdx.y` selects one of four warp rows,
- `local_line = threadIdx.y * 2 + (threadIdx.x >> 4)`,
- `local_x = local_line & 3`,
- `local_y = local_line >> 2`.

The accepted pressure len16 path already avoids inactive z lanes.  The next
target is state indexing and state traffic, not lane packing.

## Proposed Compact Descriptor

The current `PmlTile` is:

```c
typedef struct PmlTile {
  int z0;
  int x0;
  int y0;
  unsigned int mask;
} PmlTile;
```

Compact state needs a second descriptor type, default-off:

```c
typedef struct PmlCompactTile {
  int z0;
  int x0;
  int y0;
  unsigned int mask;
  unsigned int compact_base_line;
} PmlCompactTile;
```

`compact_base_line` is the first compact line owned by this tile.  A line means
one `(y, x)` line with 16 active z points.

Per thread:

```text
line_offset = compact_base_line + local_line
z16_offset = line_offset * 16 + local_z
z23_offset = line_offset * 23 + window_z
```

No global shell linear index division/modulo should be used inside the kernel.

## Compact Arrays

Initial prototype arrays:

```text
compact_memory_dzz16
compact_memory_dz_old23
compact_memory_dz_next23
```

Rationale:

- `compact_memory_dzz16` stores the pressure CPML `memory_dzz` value for the 16
  active central z positions.
- `compact_memory_dz_old23` and `compact_memory_dz_next23` store the z-recompute
  read window needed by the pressure len16 z-cache.
- A 23-slot window is conservative: 16 central points plus 4 left halo and 3
  right halo.

The first commit prototype may keep full arrays allocated for residual fallback
and for debug mirror.  The goal is to route accepted len16 pressure state
traffic through compact arrays while residual PML continues to use the existing
full arrays.

## Old/Next Behavior

Current exact-FP32 flags include:

```text
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_PML_ZMEM_IN_P
```

Therefore:

- `memory_dz` is the old z-memory state for this step,
- `memory_dz_next` is the next z-memory state produced this step,
- full-array old/next swapping remains the authoritative behavior for residual
  PML.

For compact state:

1. mirror mode fills compact old/next from full arrays and compares after the
   full-array kernel update.
2. commit mode routes accepted pressure len16 reads/writes through compact
   arrays.
3. residual pressure PML keeps full-array old/next.
4. full-array `memory_dz_next` may still be updated for residual and optional
   debug mirror, but accepted len16 pressure state must avoid duplicate
   authoritative writes.

## Debug Mirror Plan

Macro:

```text
CUDA3D_PML_LEN16_COMPACT_STATE_MIRROR
```

Behavior:

- allocate compact arrays,
- build compact descriptors,
- full-array path remains authoritative,
- after selected timesteps, gather corresponding full-array values for accepted
  pressure len16 lines into compact mirror,
- compare compact mirror against full state,
- report max abs and relative L2,
- do not change `p0/p1/vx/vy/vz` outputs.

Required validation:

- `smoke_1gpu`,
- debug dump step `0/1/2` if enabled,
- `correctness`,
- compact-vs-full rel L2 `<= 1e-6`,
- no NaN/Inf.

## Commit Prototype Plan

Macro:

```text
CUDA3D_PML_LEN16_COMPACT_STATE
```

Behavior:

- accepted pressure len16 kernel uses compact arrays for:
  - `memory_dzz`,
  - z-recompute old/next memory window.
- residual pressure PML still uses full arrays.
- full pressure fields remain unchanged.
- debug mode can mirror compact state back to full state for verification.

Acceptance:

- correctness rel L2 `<= 1e-5`,
- `perf_1gpu_6shots` repeat x3,
- output compare against current-best,
- WP mean speedup `>=1.02x`: keep candidate disabled,
- WP mean speedup `>=1.05x` and Gradient positive: promote branch,
- WP mean speedup `<1.02x`: reject commit path and keep mirror only.

## Phase 1 Audit Anchor

Phase 1 audit result:

- pressure len16 tiles: `67392`,
- pressure len16 active points: `8626176`,
- pressure len16 compact lines: `539136`,
- compact pressure-state bytes: `127.512 MiB`,
- full pressure-related state bytes x shots: `274.324 MiB`,
- NCU p_len16 duration: `67.04 us`,
- modeled whole sampled-main speedup ceiling: `1.0965x`,
- decision: `allow_commit_prototype_after_design`.

The model is optimistic.  This route may still fail if compact descriptor loads,
halo handling, or mirror synchronization costs exceed state-traffic savings.
