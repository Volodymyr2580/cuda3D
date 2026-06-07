# Core 2-Step Interior Result

Date: 2026-06-07

## Status

Scaffolding started.

Implemented in this phase:

- stable architecture decision docs;
- dependency map;
- debug dump macros for strict core interior;
- `tools/create_core_2step_case.py`;
- `tools/compare_core_interior_dumps.py`;
- design document for debug-only and future commit mode.

Not implemented yet:

- `CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE` compute kernel;
- debug-only `p(t+2)` prediction;
- commit mode;
- performance mode.

## Current Acceptance

This phase is accepted only if:

- default zmem build still compiles;
- debug-dump build compiles;
- `core_2step_interior_1gpu` runs with `ALL DONE`;
- dump metadata reports `source_in_region=0` and `receivers_in_region=0`;
- comparing a dump directory with itself passes.

Detailed run results are appended to `AGENT_LOG.md`.

