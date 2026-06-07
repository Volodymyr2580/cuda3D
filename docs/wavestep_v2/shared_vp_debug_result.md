# Shared VP Debug Result

## Verdict

Gate: `stop`

`CUDA3D_PML_ZFACE_SHARED_VP_DEBUG` is numerically correct in the tested variants, but all variants are much slower than `zmem_reference`. Do not continue this local z-face shared-tile VP route without a new profiler reason and a materially different dataflow.

## Baseline

Same-session `zmem_reference` mean:

- WP `2.448577s`
- Gradient `2.560774s`

## Tested Variants

| variant | build notes | correctness | WP | WP speed ratio | Gradient | Gradient speed ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| S2 p-only | `12x16x12`, p shared `81,120B`, 256 threads | pass | mean `3.007605s` | `0.814129x` | mean `3.169876s` | `0.807847x` |
| S4 p-only | `12x12x12`, p shared `70,304B`, 256 threads | pass | `3.039426s` | `0.805605x` | `3.188930s` | `0.803020x` |
| S4 staged-V | `12x12x12`, p+vx+vy shared `92,192B`, 256 threads | pass | mean `3.090552s` | `0.792278x` | mean `3.236345s` | `0.791255x` |

All output comparisons passed with `rel_l2 = 0`.

## Evidence

Run paths:

```text
reports/wavestep_v2_night_20260608/shared_vp_debug_perf_paths.txt
reports/wavestep_v2_night_20260608/shared_vp_debug_s4_paths.txt
reports/wavestep_v2_night_20260608/shared_vp_debug_s4_stage_v_paths.txt
```

Comparison summaries:

```text
reports/wavestep_v2_night_20260608/shared_vp_debug_perf6_repeat_vs_zmem_comparison.md
reports/wavestep_v2_night_20260608/shared_vp_debug_s4_perf6_vs_zmem_comparison.md
reports/wavestep_v2_night_20260608/shared_vp_debug_s4_stage_v_perf6_repeat_vs_zmem_comparison.md
```

Binary hashes:

```text
S2 p-only:
143a3a19fa7e57ddadb0c1cb80b10397c7e2b4b6263df2723c5f621f7ac7b324  bin/cuda_3D_FM

S4 p-only:
e7cdd11d3d0de5654d836679dbef5242b0adc6232deaa2b28a0b0f570c960ef4  bin/cuda_3D_FM

S4 staged-V:
288103e236d3c4bba160073f372a2b9bc61cf6486fcf8c439d8b094dd3e2202b  bin/cuda_3D_FM
```

## Interpretation

The p-only prototype was slow because each pressure output repeatedly recomputed `vx/vy` from shared `p1`. That avoided global velocity traffic but moved too much work into the pressure update.

The staged-V prototype was closer to the intended architecture, but the required shared-memory footprint was still too high. S4 staged-V used `92,192B` dynamic shared memory per CTA, giving effectively one CTA per SM; the extra staging loops and synchronizations outweighed saved global traffic.

## Decision

Keep the code macro-gated and default-off for traceability. Do not promote it to baseline.

The remote binary was restored to `zmem_reference` after this test:

```text
0e54c4938ea60bdb606fa67e450a5fc992b71ffe5823d2238a1414e0f30e9d6d  bin/cuda_3D_FM
```

Final restored smoke:

```text
/work/wenzhe/cuda3D/benchmarks/runs/smoke_1gpu_final_restored_zmem_smoke_20260608_020705
returncode=0
outputs=3
```
