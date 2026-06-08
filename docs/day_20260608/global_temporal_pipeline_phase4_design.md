# Global Temporal Pipeline Phase 4 Design Gate

Date: 2026-06-08

Status: design gate opened, no CUDA prototype yet.

## Why This Phase Exists

Phase 1 showed that `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` is a real but small gain:

- WP speedup: `1.032329x`
- Gradient speedup: `1.028370x`
- output rel L2: `0`

Phase 2 stopped compact-state storage:

- current CPML state is already axis-slab compact
- static compact-state ceiling: `1.005x`
- pressure PML duration did not improve with CPML double-buffer

The remaining structural direction is not another local z-face patch. It must change how wave steps move data through time.

## Current Kernel Timing Snapshot

Short NCU SpeedOfLight profile on zmem:

| kernel | duration | share of sampled main kernels | read |
| --- | ---: | ---: | --- |
| `cuda_fd3d_p_pml_tile_ns` | 189.562 us | 53.43% | dominant sampled kernel |
| `cuda_fd3d_p_core_ns` | 93.670 us | 26.40% | memory SOL `96.810%` |
| `cuda_fd3d_v_pml_tile_ns` | 71.610 us | 20.18% | CPML_DBUF already improves this path |

Sampled main-kernel total: `354.842 us`.

To get a `>=5%` speedup on this sampled set, a candidate must save at least:

```text
354.842us * (1 - 1/1.05) = 16.897us
```

If only `p_core` is improved, it needs at least:

```text
16.897us / 93.670us = 18.04% p_core reduction
```

That is possible in theory, but not enough to justify a toy-only core trick: the full WP path also includes PML/source/receiver/workflow overhead.

## Temporal-Blocking Geometry

The pressure stencil radius is `7`. A `K`-step temporal pipeline needs a dependency halo of roughly `K*7` cells from the region boundary.

For the current `perf_1gpu_6shots` geometry:

- pressure core lengths: z/x/y = `87/376/376`
- K=2 deep-core output lengths after one extra radius shrink: z/x/y = `73/362/362`
- K=2 deep-core share of original core: about `77.7%`
- K=3 deep-core output lengths after two extra radius shrinks: z/x/y = `59/348/348`
- K=3 deep-core share of original core: about `58.1%`

So a deep-core K=2 temporal pipeline has enough geometric coverage to matter only if it really reduces memory traffic. K=3 loses too much z-depth on this model unless the implementation is very efficient.

## Implementation Constraint

Normal CUDA kernels do not provide a grid-wide barrier. A second-order time update needs neighboring `p(t+1)` values to compute `p(t+2)`.

That means the easy-looking fused two-step kernel is not valid across CTA boundaries unless it is one of these:

- a CTA-local tile with halo, which is already a forbidden direction from previous failed attempts;
- a cooperative/persistent kernel with a real grid or wavefront synchronization strategy;
- a split-domain algorithm where only a provably safe deep interior is temporally advanced and boundary/PML is reconciled every step.

The next prototype must not silently read partially written neighbor data from another CTA.

## Allowed Prototype Shape

Allowed next prototype family: `CUDA3D_WAVESTEP_ENGINE_V2_TEMPORAL_PIPELINE`.

Minimum design requirements before CUDA code:

- target K=2 first, not K=3;
- restrict to deep core only at first;
- leave PML and source/receiver handling on the original per-step path;
- explicitly prove the output region dependency cone does not cross the original per-step boundary;
- include a correctness compare against zmem and CPML_DBUF;
- require `>=5%` meaningful-case WP speedup before expanding scope.

The first prototype should be rejected immediately if it needs:

- per-thread division/mod in the inner loop;
- dynamic branch-heavy classification per grid point;
- extra full-size wavefield copies that cancel the saved traffic;
- a CUDA cooperative launch whose resident grid cannot cover the required domain safely.

## Current Gate Read

Proceed to a design prototype only after a more detailed byte model answers:

1. How many bytes does `p_core` actually save for K=2 deep core after halo overhead?
2. How much of the total WP path can the deep-core path cover?
3. What synchronization mechanism is used between substeps?
4. Does the prototype avoid the previously failed CTA-local two-step pattern?

Until those are answered, do not write a temporal kernel. The immediate next work item is a byte/synchronization model, not CUDA code.
