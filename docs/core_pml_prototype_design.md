# Core \& PML Optimization Prototype Designs

> Date: 2026-06-06
> Author: Kimi (CUDA implementation engineer)
> Pre-requisite: `docs/cuda_core_dependency_map.md`

---

## Design Principles

1. **Do not repeat already-reverted experiments.** The baseline has already absorbed obvious micro-optimizations (`__restrict__`, `__ldg`, cache flags, simple shared tiles, block-size sweeps).
2. **Measure L1 vs explicit-shared trade-off.** If the L1/L2 hierarchy already captures reuse, explicit shared memory will regress due to `__syncthreads()` and halo-thread overhead.
3. **Target the cross-kernel round-trip.** The `v_pml_ns` → `p_pml_ns` velocity round-trip is the largest identifiable structural inefficiency.
4. **Preserve numerical correctness.** No stencil-order changes, no precision reduction, no domain-size changes.

---

## Prototype A: `core_opt` -- 3D Shared-Memory Tile for `p_core_ns`

### A.1 Current State

`cuda_fd3d_p_core_ns` uses a 1D shared tile in **z only**:

```cpp
__shared__ float z_tile[PBlockSize3][PBlockSize2][PBlockSize1 + 2 * CoreStencilRadius];
```

With `PBlockSize3=1, PBlockSize2=2, PBlockSize1=128`, the block is `128×2×1 = 256` threads.

x and y neighbors are loaded **directly from global memory** (coalesced, but not shared across threads). Each core point performs ≃43 `p1` loads.

### A.2 Hypothesis

L1 cache hit rate for x/y neighbors is imperfect because:
- A warp of 32 threads loads 32 consecutive z-values at fixed `(x, y)`.
- The x-neighbor loads (`base ± k*stride2`) are also coalesced, but the **same x-neighbor values are needed by all 32 threads in the warp** (just offset in z). L1 should capture this.
- The y-neighbor loads (`base ± k*stride3`) are coalesced but **different warps in the same block load disjoint y-planes**. With `PBlockSize3=1`, there is no intra-block y-sharing at all.

If we increase `PBlockSize3` to 4 or 8, threads in the same block cover multiple y-planes. A 3D shared tile can cache y-neighbors explicitly, reducing y-strided global loads from 14 per point to 0.

### A.3 Design

**New kernel:** `cuda_fd3d_p_core_opt_ns`

**Block size:** `64 × 4 × 4 = 1024` threads (max block size).

**Shared memory layout:**
```cpp
__shared__ float smem[4 + 2*CoreStencilRadius][4 + 2*CoreStencilRadius][64 + 2*CoreStencilRadius];
// dimensions: [y][x][z]
// size = 18 * 18 * 78 = 25,272 floats ≈ 101 KB
```

All p1 loads for the tile come from shared memory. Only the tile + halo is loaded from global once per block.

**Thread mapping:**
- `threadIdx.x` → z (0..63)
- `threadIdx.y` → x (0..3)
- `threadIdx.z` → y (0..3)

**Collaborative load:** Each thread loads its own center value. Halo threads (within 7 of any face) load additional halo values. After `__syncthreads()`, all stencil accesses are shared-memory reads.

### A.4 Expected Impact

| Metric | Baseline `p_core_ns` | Prototype A |
|---|---|---|
| Global `p1` loads / point | ~43 | ~1 + halo amortization |
| Shared memory / block | 1.1 KB | ~101 KB |
| Blocks for core | ~70,688 | ~17,672 |
| Registers / thread | ~32 (est.) | ~48 (est.) |

**Best case:** If L1 was missing y-neighbor reuse, speedup on `p_core_ns` could be 1.5–2×.
**Overall impact:** `p_core_ns` is ~26% of GPU time. A 2× speedup → **~13% total speedup**.

**Risk:** If L1 already captures y-neighbor reuse (e.g., across warps on the same SM), the extra `__syncthreads()` and halo-thread overhead could regress performance. The previous `p_core_2dtile` experiment (0.898s) suggests caution.

### A.5 Risk Mitigation

- Keep the old `p_core_ns` path. Add a runtime or compile-time flag to select.
- Test on `smoke_1gpu` first, then `correctness`, then `perf_1gpu`.
- Profile with `nsys` to compare L1 hit rate.

---

## Prototype B: `pml_opt` -- Intra-Block 3D Tile for `v_pml_ns`

### B.1 Current State

`cuda_fd3d_v_pml_ns` launches over the **full domain** (`n1 × n2 × n3`). Each thread:
1. Checks `need_vz / need_vx / need_vy` (early-exit for core).
2. Loads `p1` neighbors via `__ldg` (up to 24 loads).
3. Writes `vz, vx, vy` to global.

The failed `v_pml_ytile` experiment (0.665s → 0.548s) used shared memory only in the y-direction and did not improve performance.

### B.2 Hypothesis

The y-tile alone failed because:
- The block was still `128 × 2 × 1` in z-x, so **z-neighbors were not shared** within the block.
- `__ldg` already provides caching; the benefit of shared memory was outweighed by `__syncthreads()` and halo-thread overhead.

If we use a **genuine 3D tile** (non-trivial size in all 3 axes), the number of halo threads relative to interior threads decreases, and the amortization of collaborative loading improves.

Crucially, `v_pml_ns` does NOT need shared memory for output (`vz/vx/vy` are written to global). It only needs shared memory for **input** (`p1`). A 3D input tile with `By > 1` enables sharing in y, and with `Bx > 2` enables sharing in x.

### B.3 Design

**New kernel:** `cuda_fd3d_v_pml_opt_ns`

**Block size:** `32 × 8 × 2 = 512` threads.

**Shared memory layout:**
```cpp
__shared__ float smem_p1[2 + 2*radius][8 + 2*radius][32 + 2*radius];
// dimensions: [y][x][z]
// size = 10 * 16 * 40 = 6,400 floats ≈ 25 KB
```

**Collaborative load:** All 512 threads load their center `p1` value. Halo threads load additional values at the block boundaries. After `__syncthreads()`, each thread reads all 24 `p1` neighbors from shared memory.

**Key difference from previous y-tile attempt:**
- Previous: `128 × 2 × 1` block, y-tile added shared memory only for y-direction.
- New: `32 × 8 × 2` block, shared memory covers **all 3 directions**. The halo ratio is lower: unique values = 6,400, threads = 512, so only ~12.5 values per thread vs 24 loads without sharing.

### B.4 Expected Impact

| Metric | Baseline `v_pml_ns` | Prototype B |
|---|---|---|
| Global `p1` loads / active point | ~16–24 | ~1 + halo amortization (≅12.5) |
| Shared memory / block | 0 | ~25 KB |
| Blocks for full domain | ~83,232 | ~20,808 |

**Best case:** 30–50% reduction in `p1` global loads for `v_pml_ns`.
**Overall impact:** `v_pml_ns` is ~37% of GPU time. A 1.3–1.5× speedup → **~10–18% total speedup**.

**Risk:** Same as A -- if L1 already captures most reuse, shared memory overhead wins.

### B.5 Why this is different from the reverted `v_pml_ytile`

The reverted `v_pml_ytile` added a y-direction shared tile to the existing `128×2×1` block configuration. Because `blockDim.z` was still 1, there was no actual **intra-block** y-sharing; the tile was loaded by threads across the z-x plane and then shared via `__syncthreads()`. With `blockDim.z = 2` (or more), threads genuinely occupy different y-planes and can share y-neighbors directly.

---

## Prototype C: `pml_opt` -- Fused Face-Kernel with On-the-Fly Gradient Recompute (long-term)

### C.1 Concept

Instead of writing `vz, vx, vy` to global and reading them back in `p_pml_ns`, we can **eliminate the velocity arrays entirely** for PML face points by recomputing the pressure gradients (`c1, c2, c3`) directly inside the pressure kernel.

For a z-face point:
- Current: `v_pml_ns` computes `c1 = dz(p1)`, stores `vz = c1 + mem_dz`.
- Current: `p_pml_ns` reads `vz` neighbors, computes `vzz = dz(vz)`.
- Proposed: `p_pml_ns` recomputes `c1 = dz(p1)` at the point and its neighbors, reads `mem_dz` neighbors, and computes `vzz = dz(c1 + mem_dz)` on the fly.

This trades **velocity global loads** for **p1 global loads + mem_dz global loads**.

### C.2 Trade-off Analysis

| Operation | Current (loads/point) | Proposed (loads/point) |
|---|---|---|
| Read `p1` | 2 (in `p_pml_ns`) | 2 + 8×3 = 26 (recompute 3 gradients + neighbors) |
| Read `vz/vx/vy` | 24 | 0 |
| Read `mem_dz/dx/dy` | 0 | ~6 (face neighbors) |
| Write `vz/vx/vy` | 3 (in `v_pml_ns`) | 0 |

**Net:** Eliminate 27 velocity memory ops, add ~30 p1/mem ops. **Not a clear bandwidth win** unless p1 is significantly hotter in cache than velocity (which it isn't, since both are equally recently touched).

### C.3 Verdict

This prototype is **architecturally interesting but numerically risky and not clearly a bandwidth win**. It is listed here as a **long-term candidate** to be evaluated only if Prototypes A and B fail to yield sufficient speedup. It requires explicit approval from the lead architect before implementation because it changes the fundamental data flow and introduces strided `mem_*` neighbor reads that are currently hidden behind coalesced `vz/vx/vy` loads.

---

## Implementation Order

| Order | Prototype | Rationale |
|---|---|---|
| 1 | **A** (`core_opt` 3D tile) | Lowest risk, clean shared-memory pattern, teaches us about L1 vs explicit cache. |
| 2 | **B** (`pml_opt` 3D tile for `v_pml`) | Medium risk, directly targets the largest kernel (37%). |
| 3 | **C** (fused face kernel) | High risk, only if A+B are insufficient. |

## Rollback Strategy

For each prototype:
1. Add new kernel(s) with `_opt` suffix.
2. Add a compile-time or runtime switch (e.g., `#ifdef CUDA3D_USE_CORE_OPT`) in `rem_fd.cu`.
3. Old kernels remain the default.
4. If correctness fails, switch back to default and investigate.

## Success Criteria

| Case | Metric | Threshold |
|---|---|---|
| `correctness` | rel L2 | ≤ 1e-5 |
| `correctness` | NaN/Inf | None |
| `perf_1gpu` | WP time | < 0.545 s (baseline) |
| `perf_1gpu` | Speedup | ≥ 1.05× to justify keeping |

If a prototype meets correctness but is slower than baseline, it will be **reverted** and the failure mechanism documented.
