# WAVESTEP V2 Night Sprint Final Report

## Verdict

Gate result: `stop_shared_vp_continue_cpml_scaffold`

What survived:

- `zmem_reference` remains the stable baseline and final restored binary.
- `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` remains a useful default-off scaffold, with about `1.0335x` same-session WP speedup.

What stopped:

- direct fused z-face VP with global `p1` x/y second derivatives;
- S2 p-only shared pressure-tile VP;
- S4 p-only shared pressure-tile VP;
- S4 staged-V shared velocity-intermediate VP.

## Baseline

Same-session `zmem_reference`:

| run | WP computing time | Gradient TIME all | wall |
| --- | ---: | ---: | ---: |
| zmem perf6 a | `2.447898s` | `2.559258s` | `0:04.73` |
| zmem perf6 b | `2.449256s` | `2.562289s` | `0:04.84` |
| zmem mean | `2.448577s` | `2.560774s` | - |

## CPML Double Buffer

Result: `continue_as_scaffold`

| run | WP computing time | WP speed ratio | Gradient TIME all | Gradient speed ratio |
| --- | ---: | ---: | ---: | ---: |
| cpml mean | `2.369180s` | `1.033512x` | `2.486222s` | `1.029986x` |

Correctness: pass, all compared `rel_l2 = 0`.

Report:

```text
docs/wavestep_v2/cpml_vmem_double_buffer_all_result.md
```

## NCU Forensics

Nsight Compute short profiles showed why direct fused z-face failed.

`cuda_fd3d_p_pml_tile_ns`:

| metric | zmem | cpml_vmem | direct_inline |
| --- | ---: | ---: | ---: |
| duration | `188.856us` | `189.896us` | `248.200us` |
| duration ratio vs zmem | `1.000x` | `1.006x` | `1.314x` |
| SOL compute | `55.745%` | `55.788%` | `69.227%` |
| SOL memory | `55.745%` | `55.788%` | `69.227%` |

Interpretation:

- CPML double-buffer improves velocity-side ownership and leaves pressure PML nearly unchanged.
- direct inline fusion moves too much work into the pressure kernel and makes the pressure critical path about `31.4%` slower.
- direct z-face fusion is structurally stopped.

Reports:

```text
docs/wavestep_v2/phase2_fused_zface_forensics.md
docs/wavestep_v2/ncu_forensics_summary.md
reports/wavestep_v2_night_20260608/ncu_forensics_summary.json
```

## Shared VP Prototype

Budget passed only as a design gate:

```text
docs/wavestep_v2/pml_zface_shared_tile_budget.md
docs/wavestep_v2/pml_zface_shared_vp_design.md
```

Implementation/testing result:

| variant | correctness | WP | WP speed ratio | Gradient | Gradient speed ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| S2 p-only | pass | mean `3.007605s` | `0.814129x` | mean `3.169875s` | `0.807847x` |
| S4 p-only | pass | `3.039426s` | `0.805605x` | `3.188930s` | `0.803020x` |
| S4 staged-V | pass | mean `3.090552s` | `0.792278x` | mean `3.236344s` | `0.791255x` |

All output comparisons passed with `rel_l2 = 0`, but all performance gates failed badly. The route is stopped.

Detailed result:

```text
docs/wavestep_v2/shared_vp_debug_result.md
```

## Final Restore

Remote binary was restored to `zmem_reference` and smoke-tested:

```text
0e54c4938ea60bdb606fa67e450a5fc992b71ffe5823d2238a1414e0f30e9d6d  bin/cuda_3D_FM

/work/wenzhe/cuda3D/benchmarks/runs/smoke_1gpu_final_restored_zmem_smoke_20260608_020705
returncode=0
outputs=3
```

## Decision Log

Updated:

```text
AGENTS.md
docs/architecture_decision_log.md
docs/wavestep_v2/post_shared_vp_fallback_plan.md
```

Next allowed work:

- keep CPML double-buffer as scaffold;
- stop local z-face fusion work;
- move to PML compact-state audit or global-region temporal pipeline only after profiler/byte-budget evidence.
