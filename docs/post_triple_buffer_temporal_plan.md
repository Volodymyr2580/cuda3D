# Post Triple-Buffer Temporal Plan

Date: 2026-06-07

## Purpose

This is not an implementation plan for the current turn. It records how temporal blocking can be reconsidered after `p_prev / p_curr / p_next` are explicit.

## Why Triple Buffer Helps

The current in-place pressure update blocks safe fused temporal work:

```text
p0 is old p(t-1) input
p0 is also new p(t+1) output
```

Triple buffer removes this alias:

```text
p_prev read-only
p_curr read-only
p_next write-only
```

That makes it safe to design future kernels where `p_next` production and `p(t+2)` prediction are reasoned about without old-p0 overwrite races.

## Phase-2 Scope

Phase2 is deliberately narrow:

```text
Only triple-buffer-based temporal pipeline work is allowed.
```

Do not restart:

```text
PML face split
PML block/mask/prune sweeps
p_core block/register sweeps
simple CUDA Graph replay
simple memory-pool optimization
MPI temporal blocking
full-domain temporal blocking
```

## Possible Phase-2 Directions

### Split Out-of-Place Step Kernels

Candidate sequence:

```text
p_core_step1_outofplace:
    p_next = F(p_curr, p_prev)

p_core_step2_interior:
    p2 = F(p_next, p_curr) on strict safe interior
```

This is still two launches, but avoids old-p0 snapshot and in-place races.

### Fused Step1 And Interior Step2

A future fused kernel can:

```text
read p_prev and p_curr
write p_next
stage local p_next
write p2 for an interior region
```

The old-p0 race is gone, but the `R=7` halo and shared-memory surface/volume problem remains.

### Region-Level Dataflow Reuse

Triple buffer may make it easier to define owned regions:

```text
region owns p_next writes
later region predicts p2
commit/skip uses explicit buffer roles
```

This should still be gated by actual commit ratio, not just theoretical eligible region.

## Is old-p0 Snapshot Still Needed?

For correctness, no:

```text
p_prev is already a read-only old-p0 equivalent.
```

For debug fallback, an additional snapshot should not be necessary unless comparing against legacy in-place behavior.

## Expected Savings

Potential future savings can only come from:

```text
reducing repeated p_curr/p_next global reads
skipping validated pressure work
improving launch/graph scheduling after buffer roles are clean
```

Triple buffer alone is expected to be neutral or slightly slower.

## Remaining Blockers

- PML pressure still runs every timestep.
- PML/core boundary integration remains difficult.
- Source injection modifies `p_next` after pressure update.
- Receiver extraction reads post-injection `p_next` every timestep.
- MPI/halo behavior must be audited before multi-rank temporal blocking.
- `R=7` still creates heavy halo erosion for CTA-local temporal tiles.

## Gate Before Any Phase-2 Implementation

Triple-buffer standalone has passed correctness, but it did not clear the repeat performance gate by itself. Any phase2 candidate must be judged against `zmem_reference`, not against standalone triple-buffer.

Do not continue a phase2 temporal prototype unless the meaningful case clears:

```text
single GPU / single MPI rank first
correctness rel_l2 <= 1e-5
finite outputs, identical output count and sizes
perf_1gpu_6shots repeat WP speedup >= 5% vs zmem_reference
```

If the meaningful case is `<5%`, stop immediately and write the failure report. Do not expand to MPI, full-domain temporal blocking, PML face split, or further block/register sweeps.
