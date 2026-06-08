# Pressure PML Z-Recompute Cache Prototype

## Scope

Implemented macro-default-off prototype:

```text
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

The prototype changes only `cuda_fd3d_p_pml_tile_ns`.  It builds a
CTA-local shared z-line cache for `recompute_vz_after_update_from_old_mem`
values and preserves `memory_dz_next` ownership: only tile-owned active
central z positions write next z CPML memory.

It does not enable `RECOMPUTE_X/Y/XYZ`, tile-mask fastpath, z-face
specialization, or z-face fusion.

## Validation

Test platform:

```text
/work/wenzhe/cuda3D_codex_day_20260608
RTX 5090, sm_120
baseline: zmem_reference flags
candidate combo: zmem_reference + CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
                 + CUDA3D_CPML_VMEM_DISABLE_MPI
                 + CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

Clean-worktree note:

```text
The original /work/wenzhe/cuda3D worktree had uncommitted experiment
changes, so testing used a non-destructive clean worktree under /work/wenzhe.
The perf_1gpu_6shots velocity .dir was linked from the original data
location, and a missing d_obs output directory was created.
```

Correctness:

```text
debug dump step 0/1/2       pass
correctness rel L2          0 for all 6 output files
NaN/Inf                     none detected by compare_outputs.py
perf6 repeat comparisons    pass in all 3 rounds
```

## Performance

Standalone z-cache candidate:

```text
mean WP speedup             1.044955x
mean Gradient speedup       1.045506x
verdict                     useful but below standalone >=5% gate
```

Combined candidate with Phase 1 CPML vmem scaffold:

```text
mean WP speedup             1.083390x
mean Gradient speedup       1.080857x
verdict                     pass meaningful >=5% gate
```

Perf repeat table:

| round | baseline WP | candidate WP | WP speedup | baseline Gradient | candidate Gradient | Gradient speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.435633 | 2.249627 | 1.082683 | 2.545943 | 2.357701 | 1.079841 |
| 2 | 2.413101 | 2.227910 | 1.083123 | 2.533939 | 2.346707 | 1.079785 |
| 3 | 2.416663 | 2.228645 | 1.084364 | 2.542785 | 2.348029 | 1.082944 |

## Failed Aggressive Variant

Tested but removed from code:

```text
shared vx/vy pressure-neighbor cache inside the same p_pml tile
```

Result:

```text
mean WP speedup             0.419906x
mean Gradient speedup       0.426565x
```

Decision:

```text
Do not reopen pressure-PML vx/vy shared-neighbor cache in the current
block shape.  It is correctness-safe but performance-catastrophic because
the shared-memory fill/sync overhead dominates the saved global loads.
```

## Next Gate

The current accepted candidate is the combination:

```text
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_CPML_VMEM_DISABLE_MPI
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

Next work should profile this combined candidate against zmem and decompose
the remaining `cuda_fd3d_p_pml_tile_ns` time.  Do not expand to shared vx/vy
caches unless new profiler evidence changes the conclusion above.

