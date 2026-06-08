# Phase 2 Compact-State Gate Decision

Date: 2026-06-08

Decision: stop compact-state CUDA prototype for now.

## Inputs

- Static audit: `docs/day_20260608/pml_compact_state_audit.md`
- Static JSON: `reports/day_20260608/pml_compact_state_audit.json`
- NCU summary: `docs/day_20260608/pml_state_ncu_summary.md`
- Raw NCU CSV:
  - `benchmarks/profiles/day_20260608/zmem_pml_state_ncu.csv`
  - `benchmarks/profiles/day_20260608/cpml_dbuf_pml_state_ncu.csv`

## Static Findings

The current code already stores CPML memory as axis slabs:

- `memory_dy`: `2*npml*n2*n1`
- `memory_dx`: `n3*2*npml*n1`
- `memory_dz`: `n3*n2*2*npml`
- pressure mirrors: `memory_dyy/memory_dxx/memory_dzz`

So the obvious optimization "do not allocate full padded-domain CPML state" is already done.

For `perf_1gpu_6shots`:

- CPML double-buffer state footprint: `72.391 MiB`
- six padded wavefield/cw2 arrays floor: `503.039 MiB`
- state footprint share vs those six arrays: `14.39%`
- safe z-face compact region covers `84.93%` of `memory_dz`
- residual z edge/corner state still required: `602112` elements

Static traffic floor:

- mandatory CPML state update traffic: `96.521 MiB/step`
- zmem `memory_dz` old reads from recompute path: `111.762 MiB/step`
- pressure PML vx/vy load estimate: `458.344 MiB/step`
- pressure PML p0/p1/cw2/store floor: `114.586 MiB/step`

The static estimated compact-state WP speedup ceiling is only `1.005x`, below the required `>=1.05x` prototype gate.

## NCU Findings

Short NCU profile used:

- `--section SpeedOfLight`
- `--section MemoryWorkloadAnalysis`
- `--section SourceCounters`
- `--section WarpStateStats`
- `--launch-skip 10`
- `--launch-count 12`
- kernel filter: `cuda_fd3d_[vp]_pml_tile`

The CSV did not expose a fine-grained global-load/global-store sector table on this setup, but it did provide aggregate memory and duration metrics.

| kernel | zmem duration | CPML duration | read |
| --- | ---: | ---: | --- |
| `cuda_fd3d_p_pml_tile_ns` | 189.840 us | 190.293 us | pressure PML unchanged |
| `cuda_fd3d_v_pml_tile_ns` | 71.493 us | 66.000 us | velocity PML improves |

This supports the interpretation that the CPML double-buffer gain comes from velocity PML memory ownership/write behavior, not from pressure PML state layout.

## Gate

Stop compact-state prototype.

Reasons:

- The current state allocation is already slab-compact.
- A safe z-face compact layout cannot delete residual edge/corner state.
- Static upper-bound speedup is `1.005x`, below the `>=1.05x` meaningful prototype gate.
- NCU does not show pressure PML improving with CPML double-buffer, so pressure-side compact layout is not currently the dominant lever.

## Next Direction

Enter a stricter global-region temporal pipeline phase.

Constraints for the next phase:

- Do not reopen local z-face fusion, shared z-face VP, direct x/y derivative recompute, p-only shared tile, or block-size/register sweeps.
- Do not implement CTA-local core two-step or z-pencil duplicate paths already rejected in earlier notes.
- Require a design-level byte/traffic model before CUDA code.
- Require a meaningful-case prototype gate: `>=5%` speedup before expanding scope.
- Keep `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` available as an ownership scaffold, but do not make it the optimization endpoint.
