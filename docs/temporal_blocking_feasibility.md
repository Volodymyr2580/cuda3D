# Temporal Blocking Feasibility

Date: 2026-06-07

## Context

The `CUDA3D_CORE_ZPENCIL_SHARED` source-profile gate failed because the current `p_core` already caches the z/fast-dimension stencil in shared memory. The remaining local kernel bottlenecks are x/y global neighbor loads, pressure update traffic, and synchronization/latency effects.

This document records whether a larger temporal/dataflow rewrite is plausible.

## Current Time-Step Flow

The main time loop in `src/rem_fd.cu` has this order:

```text
for each time step:
  1. update PML velocity fields from p1
  2. update core pressure p0 from p1
  3. update PML pressure p0 from p1 and PML velocity/memory fields
  4. swap/advance ZMEM_IN_P auxiliary buffer if enabled
  5. inject source into p0 and extract receiver data from p0
  6. swap p0 and p1
```

Important arrays:

- `p1`: current pressure field read by velocity and pressure kernels.
- `p0`: next pressure field written by core/PML pressure kernels, then source-injected.
- `vy/vx/vz`: PML velocity state.
- `memory_dy/memory_dx/memory_dz`: PML velocity memory state.
- `memory_dyy/memory_dxx/memory_dzz`: PML pressure memory state.
- `memory_dz_next`: next z-memory buffer under `CUDA3D_PML_ZMEM_IN_P`.
- `d_est`: receiver output written each time step.

## Why Naive Two-Step Blocking Is Hard

A two-step pressure update is not just "run p_core twice inside one kernel."

Reasons:

1. The second pressure step needs the first step's updated field as its input.
2. With a 7-point radius, a two-step tile needs at least a 14-cell spatial halo for exact interior results.
3. The PML pressure update depends on velocity and memory fields that are updated every step.
4. Source injection happens after pressure update and before the `p0/p1` swap, so a blocked second step must see injected source values from the first step.
5. Receiver extraction records data every time step from the post-injection `p0`; skipping the intermediate extraction would change outputs.
6. Multi-GPU/MPI correctness would require halo exchange at every blocked boundary unless the prototype is restricted to a strict interior region.

## Feasible Prototype Scope

A safe first temporal prototype should be deliberately narrow:

```text
CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE
```

Recommended constraints:

- single GPU first;
- core interior only, excluding a guard margin of at least `2 * CoreStencilRadius` from PML boundaries;
- no source points inside the blocked region during the two-step window, or explicit source injection support inside the fused kernel;
- receiver extraction kept exact for every timestep;
- PML kernels remain unblocked and execute every timestep;
- MPI/multi-GPU path disabled until the single-GPU numerical contract is proven.

The minimal correctness target would be a small correctness case where the blocked region excludes source and receivers. After that, gradually admit source injection into the blocked kernel.

## Expected Benefit

The potential benefit is larger than local p_core micro-optimization because temporal blocking can reduce repeated global reads/writes of pressure fields across timesteps. It targets the whole dataflow:

```text
p1 global read -> stencil -> p0 global write
next step p0/p1 swap -> reread the same values from global
```

However, the risk is also much higher:

- exact boundary handling is complex;
- source/receiver timing must remain bitwise/logically aligned;
- PML memory state makes full-domain temporal blocking substantially harder;
- MPI halo exchange has to be redesigned around the temporal depth.

## Recommendation

Do not start full temporal blocking immediately.

The next high-leverage engineering path should be:

1. Build a read-only dependency map for `p_core`, PML pressure, PML velocity, source injection, receiver extraction, and swap timing.
2. Add a debug mode that compares per-timestep interior regions between baseline and a candidate.
3. Prototype a single-GPU, core-interior two-step pressure-only kernel with conservative guard margins.
4. Only after that works, decide whether to integrate source injection or expand toward PML.

This is the first direction in the project so far that could plausibly produce a structural speedup beyond a few percent, but it should be treated as a new architecture project rather than a local kernel tweak.

