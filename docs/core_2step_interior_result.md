# Core 2-Step Interior Result

Date: 2026-06-07

## Status

Debug-only `p(t+2)` strict-interior prediction validated.

Implemented in this phase:

- stable architecture decision docs;
- dependency map;
- debug dump macros for strict core interior;
- `tools/create_core_2step_case.py`;
- `tools/compare_core_interior_dumps.py`;
- design document for debug-only and future commit mode.
- `CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE` debug-only predictor kernel.

Not implemented yet:

- commit mode;
- performance mode.

## Current Acceptance

This phase is accepted:

| Check | Result |
|---|---|
| default zmem build compiles | pass |
| debug-dump build compiles | pass |
| `core_2step_interior_1gpu` runs with `ALL DONE` | pass |
| dump metadata reports `source_in_region=0` | pass |
| dump metadata reports `receivers_in_region=0` | pass |
| comparing a dump directory with itself passes | pass |
| debug-only `p2(it)` vs next `p0(it+1)` shifted comparison | pass |
| post-restore non-debug run completes | pass |

## Validation Runs

Default build after adding the scaffolding:

```text
binary SHA256 = 496e09b9021ea03c1461b242cee400b90d5782970b3dafaccc86a8601c9a0d45
```

The first debug run used the initial generated case with `xpad=0.05`. It failed usefully:

```text
ERROR invalid CUDA3D_CORE_2STEP debug region z=[26,70) x=[26,-1) y=[26,-1), n=(96,25,25)
```

This showed that acquisition-based subdomain cropping can make the strict interior empty. The case generator was fixed to use `xpad=0.5`.

Debug-dump build:

```text
binary SHA256 = afa215446262be563c2a753a79c04f115afd050aff4bfe0bf4ce7e7b19d8d244
```

Successful debug run:

```text
Gradient TIME all = 0.004658 s
WP computing time = 0.003526 s
ALL DONE
dump files = 18
```

Metadata sample:

```text
nby=61
nbx=61
nbz=96
z0=26
z1=70
x0=26
x1=35
y0=26
y1=35
count=3564
source_z=9
source_x=30
source_y=30
source_in_region=0
receiver_count=9
receivers_in_region=0
```

Self-compare:

```text
report = benchmarks/reports/core2step_debug_self_compare_xpad_20260607/comparison.md
pass = True
files compared = 12
```

The server binary was restored to a non-debug zmem build:

```text
binary SHA256 = e2e48089353443fbbf3088ef7e1131bec9023e4721f25e9ff4a1f3e90a8a045a
```

Post-restore run on the same case:

```text
Gradient TIME all = 0.002856 s
WP computing time = 0.001342 s
ALL DONE
```

## Predictor Validation

Debug-only predictor build:

```text
flags += -DCUDA3D_CORE_2STEP_DEBUG_DUMP -DCUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE
binary SHA256 = b3ba2be23e7f07aa4b7593ba154649bf1b4c616b87c880548ef38dc29bc90f36
```

Successful debug-only predictor run:

```text
run = benchmarks/runs/core_2step_p2_debug_20260607_152450
Gradient TIME all = 0.005677 s
WP computing time = 0.004523 s
Elapsed wall clock = 0:02.49
ALL DONE
dump files = 23
p2 dump files = 5
```

Shifted predictor comparison:

```text
report = benchmarks/reports/core2step_p2_shift_compare_20260607_152450/comparison.md
mode = p2-shift
pass = True
files compared = 5
rel_l2 = 0.0 for all compared timesteps
max_abs = 0.0 for all compared timesteps
```

Post-restore non-debug run after gating the predictor kernel behind `CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE`:

```text
run = benchmarks/runs/core_2step_p2_postrestore_default_20260607_152450
Gradient TIME all = 0.002968 s
WP computing time = 0.001357 s
Elapsed wall clock = 0:02.48
ALL DONE
binary SHA256 after this rebuild = b0996e463e6a89c8adfaf5daf84c6441d3bf6289356ec2f3637110791858289b
```

Note: repeated non-debug rebuilds on this server did not produce identical binary SHA256 values, so binary hash is recorded for traceability but is not used as the sole acceptance criterion for this step.

## Next Step

The next implementation step is commit mode: reuse the validated `p(t+2)` strict-interior result and skip recomputing that interior region on the following timestep. The first commit-mode version must stay single-GPU, reject source/receiver-in-region cases, and keep the baseline path for PML and guard regions.
