# Inject/Extract Launch Overhead Gate

Date: 2026-06-08

Context:

```text
Current best is the direct-fill pressure z-cache combo:
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_CPML_VMEM_DISABLE_MPI
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

## NCU Profile

Target kernel:

```text
lint3d_inject_bell_extract_gpu_zz
```

Report:

```text
reports/day_20260608/inject_extract_ncu_summary.md
reports/day_20260608/inject_extract_ncu_summary.json
```

Summary:

| metric | value |
| --- | ---: |
| mean duration | 5.109us |
| SOL compute | 0.040% |
| SOL memory | 6.699% |
| SOL DRAM | 1.414% |

Nsight Compute rule:

```text
The kernel grid is too small to fill available resources, producing 0.0 full
waves across all SMs.
```

Interpretation:

```text
This is a small-kernel / launch-scheduling signal, not evidence that the
main pressure-PML math path can be improved by changing this helper kernel's
local block geometry.
```

## BS512 Candidate

Candidate:

```text
CUDA3D_INJECT_EXTRACT_BS512
```

Change:

```text
Use BS=512 instead of BS=1024 for lint3d_inject_bell_extract_gpu_zz launch
geometry.
```

Correctness:

```text
pass, 6 outputs
rel L2 = 0 for every output
```

Perf repeat:

```text
all compare pass                       true
mean WP speedup vs direct-fill          0.999684x
mean Gradient speedup vs direct-fill    0.998963x
```

| round | direct WP | candidate WP | WP speedup | direct Gradient | candidate Gradient | Gradient speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.210482 | 2.207253 | 1.001463 | 2.317219 | 2.319892 | 0.998848 |
| 2 | 2.188303 | 2.191148 | 0.998702 | 2.305490 | 2.309288 | 0.998355 |
| 3 | 2.189624 | 2.192060 | 0.998889 | 2.308208 | 2.308930 | 0.999687 |

Decision:

```text
Reject BS512.  It is numerically safe, but it misses the >=2% small-candidate
gate and is slightly slower on mean.
```

Follow-up boundary:

```text
Do not retry inject/extract block-size-only tuning.  If this area is reopened,
the candidate must be a broader CUDA Graph, launch aggregation, or wave-step
scheduling design with repeat evidence.
```
