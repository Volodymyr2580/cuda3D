# CUDA3D_WAVESTEP_ENGINE_V2 Design

Date: 2026-06-08

## Goal

`CUDA3D_WAVESTEP_ENGINE_V2` is a dataflow rewrite project, not another kernel micro-sweep.

The stable RTX 5090 baseline remains:

```text
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DCUDA3D_PML_ZMEM_IN_P
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
```

The next realistic speed pool is PML ownership:

```text
p_curr -> v_pml_tile -> global vx/vy/vz -> p_pml_tile -> p_next
```

Phase 1 only cleans CPML velocity memory ownership. It must not perform PML fusion.

## Current ZMEM Timestep Flow

In the stable zmem path, one timestep is:

```text
1. v_pml_tile reads p1.
2. v_pml_tile updates vx/vy global velocity and memory_dx/memory_dy in-place.
3. v_pml_tile skips z velocity memory update because ZMEM_IN_P is active.
4. p_core updates strict core pressure.
5. p_pml_tile recomputes z velocity from p1 and memory_dz old.
6. p_pml_tile writes memory_dz_next for owned z velocity memory.
7. p_pml_tile reads global vx/vy for x/y pressure divergence.
8. p_pml_tile updates pressure CPML memory_dxx/dyy/dzz in-place.
9. p_pml_tile writes p0.
10. host swaps memory_dz and memory_dz_next.
11. source injection writes p0.
12. receiver extraction reads p0.
13. host swaps p0 and p1.
```

## Pressure State

Stable zmem still uses the legacy two-pressure-buffer schedule:

```text
p0: old p(t-1), overwritten as p(t+1)
p1: current p(t)
```

The pressure triple-buffer branch proved this can become:

```text
p_prev: p(t-1), read-only
p_curr: p(t), read-only
p_next: p(t+1), write target
```

Phase 1 does not require pressure triple-buffer. It prepares CPML memory ownership for later PML fusion.

## CPML Velocity Memory State

Current stable zmem:

```text
memory_dx: velocity x CPML memory, in-place
memory_dy: velocity y CPML memory, in-place
memory_dz: velocity z CPML memory old
memory_dz_next: velocity z CPML memory next
```

Phase 1 target under `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`:

```text
memory_dx_old -> memory_dx_next
memory_dy_old -> memory_dy_next
memory_dz_old -> memory_dz_next
```

After every timestep:

```text
swap(memory_dx_old, memory_dx_next)
swap(memory_dy_old, memory_dy_next)
swap(memory_dz_old, memory_dz_next)
```

## CPML Pressure Memory State

Pressure CPML memory remains in-place in Phase 1:

```text
memory_dxx
memory_dyy
memory_dzz
```

These are produced and consumed only inside the pressure PML update for the current timestep. They are not double-buffered in Phase 1.

## Global Velocity State

Global velocity arrays remain present in Phase 1:

```text
vx
vy
vz
```

Stable zmem behavior:

```text
vx/vy are written by v_pml_tile and read by p_pml_tile.
vz is not required by p_pml_tile when ZMEM_IN_P is active.
```

Phase 1 keeps this behavior. The future fused VP phase is the first stage allowed to remove fused-owned `vx/vy/vz` global round trips.

## Producer/Consumer Table

| State | Producer | Consumer | Phase 1 Ownership |
|---|---|---|---|
| `p1` | previous timestep pressure swap | `v_pml_tile`, `p_core`, `p_pml_tile`, source/receiver schedule | unchanged |
| `p0` | `p_core`, `p_pml_tile`, source injection | receiver extraction, next host swap | unchanged |
| `vx` | `v_pml_tile` | `p_pml_tile` | unchanged |
| `vy` | `v_pml_tile` | `p_pml_tile` | unchanged |
| `vz` | skipped under `ZMEM_IN_P` | unused by zmem pressure path | unchanged |
| `memory_dx_old` | previous host swap | `v_pml_tile`, optional recompute debug paths | read-only |
| `memory_dx_next` | `v_pml_tile` | next timestep after swap | write-only |
| `memory_dy_old` | previous host swap | `v_pml_tile`, optional recompute debug paths | read-only |
| `memory_dy_next` | `v_pml_tile` | next timestep after swap | write-only |
| `memory_dz_old` | previous host swap | `p_pml_tile` z recompute | read-only |
| `memory_dz_next` | `p_pml_tile` | next timestep after swap | write-only |
| `memory_dxx/dyy/dzz` | `p_pml_tile` | `p_pml_tile` next timestep | unchanged, in-place |
| source injection | source kernel | `p0` | unchanged |
| receiver extraction | receiver kernel | `p0` | unchanged |

## Arrays That Can Be Double-Buffered

Phase 1 allows only velocity CPML memory:

```text
memory_dx
memory_dy
memory_dz
```

These are the old/new state needed before PML velocity-pressure fusion.

## Arrays That Must Stay Residual Fallback

Do not double-buffer or remove in Phase 1:

```text
vx/vy/vz global velocity
memory_dxx/memory_dyy/memory_dzz pressure CPML memory
p0/p1 pressure buffers
source/receiver interpolation arrays
MPI/halo state
```

## First Fused VP Region Definition

Not implemented in Phase 1. The first future fused VP region should be:

```text
pure z-PML face
x/y inside core-safe columns
no edge/corner
no source/receiver in safety band
single GPU / single MPI rank
```

That future stage must prove that fused-owned velocity can become a local intermediate rather than a global state.

## Correctness Gates

Phase 1 must pass:

```text
debug dump step 0/1/2 vs zmem_reference
smoke_1gpu
correctness
all output values finite
relative L2 <= 1e-5
```

Debug fill:

```text
CUDA3D_CPML_VMEM_DEBUG_FILL fills next velocity CPML buffers with NaN-like bytes before a timestep.
Coverage check verifies next buffers contain no NaN after v/p PML ownership writes.
```

## Performance Gates

Phase 1:

```text
perf_1gpu_6shots repeat slowdown <= 2%: continue
slowdown 2% to 5%: diagnose overhead
slowdown > 5%: stop and report
```

Future PML fused VP z-face:

```text
meaningful case repeat speedup >= 10%
perf_1gpu_6shots repeat speedup >= 5%
```

## Phase 2 Z-Face Result

`CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY` was tested in two forms:

```text
1. Separate fused z-face kernel.
2. Inline p_pml fused branch with no extra kernel launch.
```

Both forms passed smoke, correctness, and perf6 repeat output comparison, but both failed performance:

```text
zmem mean WP:              2.434461s
separate zface mean WP:    2.660077s
inline zface mean WP:      2.692579s
```

Conclusion:

```text
Direct p1-based x/y second derivatives are not an acceptable replacement for the vx/vy global round trip on RTX 5090.
Do not repeat this route unless the next design keeps velocity intermediates CTA-local with shared-memory reuse or is backed by new profiler evidence.
```
