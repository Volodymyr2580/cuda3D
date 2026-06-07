# CUDA Core Dependency Map

> Generated: 2026-06-06
> Target: `fd_3d_f` time-stepping loop in `rem_fd.cu`
> Active kernels: `cuda_fd3d_v_pml_ns`, `cuda_fd3d_p_core_ns`, `cuda_fd3d_p_pml_ns`

---

## 1. Per-Time-Step Kernel Launch Sequence

```
for it = 0 .. nt-1:
  1. cuda_fd3d_v_pml_ns   <<<full domain>>>
  2. cuda_fd3d_p_core_ns  <<<core box only>>>
  3. cuda_fd3d_p_pml_ns   <<<full domain (core early-exit)>>>
  4. lint3d_inject_bell_extract_gpu_zz  <<<1 block>>>
  5. swap(p0, p1)
```

Implicit device-wide synchronization exists **between every kernel** (different launches).
No synchronization exists **inside a single kernel** across blocks.

---

## 2. Array Inventory

| Array | Size (elements) | Domain | Meaning |
|-------|-----------------|--------|---------|
| `d_p0`, `d_p1` | `nxyzpad = nypad * nxpad * nzpad` | Full padded grid | Pressure wavefield (double buffer) |
| `d_vy`, `d_vx`, `d_vz` | `nxyzpad` | Full padded grid | Velocity components (intermediate) |
| `d_cw2` | `nxyzpad` | Full padded grid | Model: `v^2 * dt^2` or equivalent |
| `d_memory_dy`, `d_memory_dx`, `d_memory_dz` | PML face volumes | PML shell only | Memory variables for **velocity** PML update |
| `d_memory_dyy`, `d_memory_dxx`, `d_memory_dzz` | PML face volumes | PML shell only | Memory variables for **pressure** PML update |
| `d_ay/bz` etc. | `npml` (constant mem) | PML coefficients | Already in `__constant__` |

**Full-domain arrays:** `p0, p1, vy, vx, vz, cw2`
**PML-only arrays:** `memory_dy/dx/dz/dyy/dxx/dzz`

---

## 3. Producer / Consumer Matrix

| Array | Producer | Consumers | Notes |
|-------|----------|-----------|-------|
| `p1` | Previous step's swap | `v_pml_ns` (reads pressure gradients) | Old pressure field |
| `p0` | `p_core_ns`, `p_pml_ns`, `inject` | Next step's swap (becomes `p1`) | New pressure field |
| `vz` | `v_pml_ns` | `p_pml_ns` | Z-velocity, written in step 1, read in step 3 |
| `vx` | `v_pml_ns` | `p_pml_ns` | X-velocity, written in step 1, read in step 3 |
| `vy` | `v_pml_ns` | `p_pml_ns` | Y-velocity, written in step 1, read in step 3 |
| `mem_dz` | `v_pml_ns` | `v_pml_ns` (next step) | Accumulating PML memory state |
| `mem_dx` | `v_pml_ns` | `v_pml_ns` (next step) | Accumulating PML memory state |
| `mem_dy` | `v_pml_ns` | `v_pml_ns` (next step) | Accumulating PML memory state |
| `mem_dzz`| `p_pml_ns` | `p_pml_ns` (next step) | Accumulating PML memory state |
| `mem_dxx`| `p_pml_ns` | `p_pml_ns` (next step) | Accumulating PML memory state |
| `mem_dyy`| `p_pml_ns` | `p_pml_ns` (next step) | Accumulating PML memory state |

**Critical data flow:**
```
p1 --> v_pml_ns --> [vz, vx, vy] --> p_pml_ns --> p0
         |                               |
         v                               v
    mem_d{z,x,y}                    mem_d{zz,xx,yy}
```

---

## 4. Cross-Block Dependency: Why `v_pml + p_pml` Fusion Failed

### 4.1 Stencil footprints

**`v_pml_ns`** computes velocity at point `(z, x, y)` by reading `p1` at:
- `vz`: `[z, z+1, z+2, z+3, z+4]` and `[z-1, z-2, z-3]` (8-point stencil in z)
- `vx`: `[x, x+1, x+2, x+3, x+4]` and `[x-1, x-2, x-3]` (8-point in x)
- `vy`: `[y, y+1, y+2, y+3, y+4]` and `[y-1, y-2, y-3]` (8-point in y)

**`p_pml_ns`** computes pressure divergence by reading `vz/vx/vy` at:
- `vzz`: `vz[z]` and `vz[z-1]` (backward difference in z)
- `vxx`: `vx[x]` and `vx[x-1]` (backward difference in x)
- `vyy`: `vy[y]` and `vy[y-1]` (backward difference in y)
Plus wider stencil: `vz[z+1..+3]`, `vx[x+1..+3]`, `vy[y+1..+3]`.

### 4.2 The race condition

If `v_pml` and `p_pml` were fused into one kernel:
- Block A writes `vz[outIndex]`.
- Block B (neighbor in z) needs `vz[outIndex-1]` and `vz[outIndex+1]` for its divergence.
- **There is no `__syncthreads()` across blocks.** Block B may read `vz` before Block A has written it.
- Result: pressure divergence uses **stale velocity values** from the previous time step.

This is why the previous `vp_fused` experiment failed correctness.

### 4.3 Acceptable fusion prerequisites

Fusion is only safe if one of the following holds:
1. The consumer stencil footprint is fully contained within the same block's write footprint (no halo dependency).
2. Domain decomposition is changed so that halo tiles live in shared memory / cluster-scope sync.
3. A multi-stage kernel explicitly writes halo to global, syncs, then reads back.
4. Data layout is changed so PML update does not need freshly-written neighbor velocity.

---

## 5. Memory Traffic Per Grid Point

### 5.1 Notation

- `n1 = nbz = nz + 2*nbd` (Z / fast axis)
- `n2 = nbx = nx + 2*nbd` (X / middle axis)
- `n3 = nby = ny + 2*nbd` (Y / slow axis)
- `npml = nbd` (PML thickness)
- `CorePmlMargin = 4` (safety margin between core and PML)
- `core_lo = npml + CorePmlMargin`, `core_hi = n{1,2,3} - core_lo`
- `radius = 4` (8th-order first derivative stencil half-width)
- `CoreStencilRadius = 7` (14th-order second derivative stencil half-width in `p_core_ns`)

### 5.2 `cuda_fd3d_v_pml_ns` -- Full domain, core early-exit

**Scope:** All `(n1, n2, n3)` threads launch. Core threads return immediately.
Only shell threads (within `CorePmlMargin` of core boundary or inside PML) compute.

**Per active thread (shell point):**

| Operation | Count | Note |
|-----------|-------|------|
| Read `p1` | 8--24 | 8 per needed component (`need_vz`, `need_vx`, `need_vy`) |
| Write `vz` | 0--1 | Only if `need_vz` |
| Write `vx` | 0--1 | Only if `need_vx` |
| Write `vy` | 0--1 | Only if `need_vy` |
| R/W `mem_dz` | 0--2 | One per active PML face (z-lo, z-hi) |
| R/W `mem_dx` | 0--2 | One per active PML face (x-lo, x-hi) |
| R/W `mem_dy` | 0--2 | One per active PML face (y-lo, y-hi) |

**Typical active shell point:** ~16 `p1` reads + 2--3 velocity writes + 2--4 memory R/W.
All loads use `__ldg` (read-only cache).

### 5.3 `cuda_fd3d_p_core_ns` -- Core box only

**Scope:** `(core_n1, core_n2, core_n3)` threads.

**Per core point:**

| Operation | Count | Note |
|-----------|-------|------|
| Read `p1` | 15 (z) + 14 (x) + 14 (y) = 43 | 14th-order second derivative in each axis |
| Read `p0` | 1 | For `2*p1 - p0` update |
| Read `cw2` | 1 | Model coefficient |
| Write `p0` | 1 | Updated pressure |

Shared memory caches the z-plane (15 points), reducing redundant global loads at block boundaries.
x and y loads come directly from global memory but are **coalesced** (adjacent threads differ in z).

### 5.4 `cuda_fd3d_p_pml_ns` -- Full domain, core early-exit

**Scope:** Same grid as `v_pml_ns`. Core threads return immediately.

**Per active thread (shell point):**

| Operation | Count | Note |
|-----------|-------|------|
| Read `vz` | 8 | If z-divergence needed |
| Read `vx` | 8 | If x-divergence needed |
| Read `vy` | 8 | If y-divergence needed |
| Read `p1` | 2 | `2*__ldg(p1+outIndex)` and `__ldg(p1+outIndex)` |
| Read `p0` | 1 | Old pressure |
| Read `cw2` | 1 | Model coefficient |
| Write `p0` | 1 | Updated pressure |
| R/W `mem_dzz` | 0--2 | PML face (z-lo, z-hi) |
| R/W `mem_dxx` | 0--2 | PML face (x-lo, x-hi) |
| R/W `mem_dyy` | 0--2 | PML face (y-lo, y-hi) |

**Typical active shell point:** ~16 velocity reads + 4 pressure R/W + 2--4 memory R/W.
All velocity loads use `__ldg`.

### 5.5 `lint3d_inject_bell_extract_gpu_zz`

- Atomically adds source wavelet into `p0` (8 `atomicAdd`s).
- Reads `p0` for receiver extraction (8 reads).
- Negligible traffic compared to FD kernels.

### 5.6 Summary bandwidth estimate (perf_1gpu)

Config: `384 x 384 x 95`, `nt=1501`, `npml=12`, `nbd=12`

```
n1 = 119, n2 = 408, n3 = 408
Total points per step = 119 * 408 * 408 ~ 19.8M
Core box = (119-32) * (408-32) * (408-32) ~ 87 * 376 * 376 ~ 12.3M
Shell points per step ~ 7.5M
```

| Kernel | Active points/step | Est. bytes/point | Est. bytes/step |
|--------|-------------------|------------------|-----------------|
| `v_pml_ns` | 7.5M | ~100 B | ~0.75 GB |
| `p_core_ns` | 12.3M | ~180 B | ~2.2 GB |
| `p_pml_ns` | 7.5M | ~120 B | ~0.9 GB |
| **Total** | | | **~3.9 GB** |

Over 1501 steps: ~5.8 TB of global memory traffic.
At baseline 0.545 s: effective ~10.6 TB/s.

> **Note:** Actual global DRAM traffic is lower because `__ldg` and L1/L2 cache exploit spatial and temporal reuse (neighboring points share stencil footprints). The kernel is likely **memory-latency-bound** rather than purely bandwidth-bound, given the high L1 hit rate from coalesced access and warp-level locality.

---

## 6. Full-Domain vs PML-Only Arrays

### 6.1 Full-domain arrays (allocate `nxyzpad`)

```
p0, p1, vy, vx, vz, cw2
```

These arrays span the entire padded volume including PML and `radius` halo.
**Problem:** `vy, vx, vz` are written for **every** full-domain point in `v_pml_ns`, but only the shell values are ever read by `p_pml_ns`. Core values are computed and then never used.

Current `v_pml_ns` mitigates this by early-exiting core threads (they do not write), but the kernel is still launched over the full domain.

### 6.2 PML-only arrays (boundary faces)

| Array | Shape | Effective volume (floats) |
|-------|-------|---------------------------|
| `mem_dz` | `2 * npml * n2 * n3` | Two z-faces |
| `mem_dx` | `n1 * 2 * npml * n3` | Two x-faces |
| `mem_dy` | `n1 * n2 * 2 * npml` | Two y-faces |
| `mem_dzz` | `2 * npml * n2 * n3` | Two z-faces |
| `mem_dxx` | `n1 * 2 * npml * n3` | Two x-faces |
| `mem_dyy` | `n1 * n2 * 2 * npml` | Two y-faces |

**Problem:** These face arrays are accessed with **non-uniform strides** depending on which face is active. The indexing formulas differ for lo-face vs hi-face, creating branch divergence and non-coalesced access patterns for some faces (especially y-face, where `n1*n2` stride is large).

Example `mem_dz` indexing (z-lo face):
```
pind = gtid3 * n2 * npml + gtid2 * npml + gtid1
```
Adjacent threads (varying `gtid1`) access adjacent memory -- coalesced.

Example `mem_dy` indexing (y-lo face):
```
pind = gtid3 * n2 * n1 + gtid2 * n1 + gtid1
```
Adjacent threads (varying `gtid1`) access adjacent memory -- coalesced.

However, the **y-face memory arrays** are accessed with `gtid3` as the slowest varying index inside the face, while the kernel grid has `gtid1` (z) as the fastest varying index. Since `gtid3` varies across blocks but not within a warp (blockDim.z=1), a warp sees the same `gtid3`, and adjacent threads vary `gtid1` -- coalesced for the face arrays too.

So the face array access is actually coalesced. The bigger issue is **divergence** (different faces take different branches) and **redundant full-domain velocity storage**.

---

## 7. Structural Observations for Next Phase

### 7.1 `core_opt` opportunities

1. **Register blocking / z-line sliding window:** `p_core_ns` already uses shared memory for z, but block is only `128x2x1`. A larger y-tile could enable y-direction caching in shared memory or registers, reducing the 14 y-strided loads per point.

2. **Warp shuffle for y-direction:** Since `blockDim.z = 1`, all threads in a block share the same `gtid3`. But `p_core_ns` loads `p1[base +/- k*stride3]` (y-direction) directly from global memory. A warp-shuffle approach could reuse y-neighbor values if multiple y-planes were processed by the same warp, but current block size prevents this.

3. **Temporal blocking prototype:** Core-only region could accumulate multiple time steps in registers/shared memory before writing back to global memory. Halo expansion per step = `CoreStencilRadius = 7`. For `T` steps, halo expands `7*T` in each direction. Shared memory budget must be explicitly computed.

### 7.2 `pml_opt` opportunities

1. **Avoid full-domain velocity writes:** `v_pml_ns` writes `vz/vx/vy` into full-domain arrays, but `p_pml_ns` only reads shell values. A boundary-specialized layout could store shell velocities in a compact buffer, eliminating the full-domain global round-trip.

2. **PML face array reordering:** Current face arrays pack lo-face and hi-face into one contiguous buffer with offset arithmetic. Splitting into separate arrays per face could simplify indexing and reduce branch divergence.

3. **`v_pml --> p_pml` dependency reduction:** The divergence computation in `p_pml_ns` needs velocity at `base` and `base-1`. If `v_pml_ns` computed **fluxes** (e.g., `flux_z = vz[base] - vz[base-1]`) and stored them, `p_pml_ns` would only need to read the flux, not both velocity points. However, the stencil is wider than 2 points (it uses `base`, `base-1`, `base+1`, `base+2`, `base+3`), so a simple 2-point flux is insufficient. But a **stencil-compressed representation** (e.g., precomputed divergence contributions at each point) could decouple the kernels.

4. **PML slab launch:** Instead of launching over full domain and letting core threads early-exit, launch only over the union of PML slabs. This reduces thread launch count and eliminates the core-boundary branch check.

---

## 8. Synchronization Points

```
Step it:
  v_pml_ns  --> implicit device sync --> p_core_ns
  p_core_ns --> implicit device sync --> p_pml_ns
  p_pml_ns  --> implicit device sync --> inject/extract
  inject    --> implicit device sync --> swap(p0,p1)
```

Between steps, the only state carried forward is:
- `p0 <-> p1` (swapped)
- `mem_d{z,x,y}` and `mem_d{zz,xx,yy}` (persistent PML state)

All other arrays (`vz, vx, vy`) are **fully recomputed** each step.
