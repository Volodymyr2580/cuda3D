# CUDA3D Micro-Bank Policy

## Purpose

The current best proves that small, clean CUDA changes can accumulate.  Future
work should not kill every sub-5% idea automatically.  Instead, separate
low-risk composable micro patches from heavy local architecture routes.

## Branch

Use a dedicated branch for this class of work:

```text
exp/micro-bank-current-best
```

## Admission Rules

A patch may enter the micro-bank only if all mandatory rules pass:

- Correctness passes with the current tolerance policy.
- `perf_1gpu_6shots` repeat mean WP speedup is `>=1.010x`, or a stack-level
  ablation after several patches is `>=1.020x`.
- No single repeat run is below `0.995x` WP.
- Mean Gradient does not regress.
- Register count does not increase by more than `4` registers/thread.
- Static shared memory does not increase by more than `2 KiB` unless explicitly
  justified.
- The patch is local, macro-gated, and easy to revert.
- It does not create a new ownership family.
- It does not block future pressure/PML dataflow work.

## Required Evidence

For every accepted micro-bank patch:

- Build flags.
- Binary hash.
- 3-round A/B repeat.
- Output comparison summary.
- NCU or ptxas evidence if resource usage changes.
- Short ablation note explaining why it composes with the current stack.

After every three banked patches, run an ablation:

```text
current_best
current_best + patch A
current_best + patch B
current_best + patch C
current_best + A+B+C
```

Only keep the combined stack if the aggregate result is positive.

## Accepted Current Stack

These are already accepted into the current best:

- `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`
- `CUDA3D_CPML_VMEM_DISABLE_MPI`
- `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
- `CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK`
- `CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK`

## Non-Bankable Routes

These are not micro-bank candidates.  They are heavy architecture routes and
remain prohibited unless a new model reopens them:

- direct z-face VP fusion
- shared VP retry
- residual pressure branch-only split
- pressure length-32 / length-23 descriptor retry
- v-PML descriptor / point-list expansion after v-len16
- current p-core shared-plane/block/register family
- ordinary K=2 CUDA temporal
- cooperative-grid temporal prototype
- cluster-local DSM K=2 temporal prototype
- same-GPU multi-rank oversubscription
- host/setup micro-prototypes without a new `>=5%` measured hotspot

## Interpretation

Low-risk micro patches may be banked at `1%` to `3%` because they can compose.
Heavy ownership, fusion, temporal, cluster, or scheduling routes still need a
larger gate because their complexity can easily make the whole program slower.
