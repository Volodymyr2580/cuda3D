# Wave-Step Async Streams Prototype

## Context

The accepted current-best candidate before this experiment was:

```text
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_CPML_VMEM_DISABLE_MPI
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
```

The stream-overlap model allowed a first prototype because `p_core` is
dependency-independent from the PML chain in a single wave step:

```text
p_core: reads p1/cw2, writes p0 core only
v_pml: reads p1, writes vx/vy and CPML velocity state
p_pml: reads vx/vy, writes p0 PML only
```

Model report:

```text
docs/day_20260608/wavestep_stream_overlap_model.md
reports/day_20260608/wavestep_stream_overlap_model.json
```

## Prototype

Temporary macro tested remotely:

```text
CUDA3D_WAVESTEP_ASYNC_STREAMS
```

Implementation tested in remote clean worktree:

```text
/work/wenzhe/cuda3D/.codex_worktrees/async_streams_20260608_1738
```

Prototype schedule:

```text
stream_core: p_core
stream_pml:  v_pml -> p_pml_len16 -> p_pml_residual
default:     wait core+pml -> source injection/extraction -> record next-step event
```

The prototype changed only host-side launch scheduling and event waits.  It did
not change CUDA math kernels.

## Validation

Build:

```text
make -B -f makefile.rtx5090 test NVFLAGS="... -DCUDA3D_WAVESTEP_ASYNC_STREAMS"
```

Async candidate binary SHA256:

```text
78ea9c9ee37328ff913e9a403b8abe3ec3e3c2f232790ccb104e3dcdcd2e0f86
```

Smoke:

```text
benchmarks/runs/smoke_1gpu_async_streams_smoke_datafixed_flags_20260608_174937
returncode 0
outputs 3
ALL DONE
```

Correctness:

```text
baseline:  benchmarks/runs/correctness_len16_base_for_async_20260608_175017
candidate: benchmarks/runs/correctness_async_streams_candidate_20260608_175026
compare:   reports/day_20260608/wavestep_async_correctness_compare_20260608_175029
```

Result:

```text
pass: true
6 outputs
max rel L2: 0
```

Operational note:

```text
The first smoke/perf attempts in the new worktree failed because test output
or velocity data files were missing from the isolated worktree, not because of
the CUDA stream code.  After adding only the missing test output directory and
velocity symlink, smoke/correctness/perf ran normally.
```

## Performance

Repeat report:

```text
reports/day_20260608/wavestep_async_perf6_repeat_20260608_175407/summary.md
reports/day_20260608/wavestep_async_perf6_repeat_20260608_175407/summary.json
```

Summary:

| round | base WP | async WP | WP speedup | base Gradient | async Gradient | Gradient speedup | compare |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2.046013s | 2.037689s | 1.004085x | 2.163786s | 2.158926s | 1.002251x | True |
| 2 | 2.033079s | 2.021030s | 1.005962x | 2.156490s | 2.150619s | 1.002730x | True |
| 3 | 2.032439s | 2.021318s | 1.005502x | 2.156099s | 2.148397s | 1.003585x | True |

Mean:

```text
WP speedup        1.005183x
Gradient speedup  1.002855x
all compare pass  true
```

## Decision

Decision:

```text
reject_cuda_prototype
```

Reason:

```text
The prototype is numerically correct, but the observed repeat speedup is only
about 0.5% WP and 0.3% Gradient, far below the >=5% meaningful prototype gate.
The profiler model was a valid upper-bound gate, but hardware resource
contention leaves almost no realized overlap in this workload.
```

Source status:

```text
The temporary CUDA3D_WAVESTEP_ASYNC_STREAMS source changes were removed from
the local mainline after validation.  Only this report and the model/tool are
kept.
```

Do not continue:

```text
Do not reimplement the same two-stream p_core-vs-PML overlap prototype.
Do not write CUDA Graph / launch aggregation for this single-GPU case based on
this result.
Do not open three-stream pressure residual/len16 fanout until Nsight Systems
shows real concurrent execution headroom and a new model proves >=5% repeat
speedup after contention.
```

Allowed next direction:

```text
Move back to memory-ownership changes that reduce real global memory work, not
only host-side scheduling.  The strongest remaining accepted signal is pressure
PML final p0/cw2 writeback and CPML z-state dependency.
```
