# Cluster / Cooperative Primitive Gate

## Summary

RTX 5090 / CUDA 13 exposes both cooperative launch and thread-block
cluster launch, but this does **not** reopen the previous ordinary
K=2 temporal CUDA prototype.

Decision:

```text
reject_direct_cooperative_grid_k2_temporal_reopen
design_only_until_cluster_local_ownership_model_passes
ordinary CUDA prototype allowed = false
cluster CUDA prototype allowed = false
```

## Probe Evidence

Probe source:

```text
tools/cuda_cluster_capability_probe.cu
```

Remote worktree:

```text
/work/wenzhe/cuda3D/.codex_worktrees/cluster_probe_20260609_0132
```

Raw stdout:

```text
reports/day_20260609/cluster_probe_stdout_20260609_0132.txt
```

Device:

```text
name                         NVIDIA GeForce RTX 5090
compute capability           12.0
SM count                     170
CUDA                         13.0
cooperative_launch           1
cluster_launch               1
```

Cooperative launch capacity:

```text
block size                    128
active blocks / SM            12
cooperative grid ceiling      2040 blocks
previous K=2 required blocks  70688 blocks
over-capacity factor          34.6510x
```

Cluster launch probe:

| cluster size | active clusters | launch |
| ---: | ---: | --- |
| 1 | 340 | pass |
| 2 | 170 | pass |
| 4 | 85 | pass |
| 8 | 41 | pass |
| 16 | -1 | cluster misconfiguration |

## Interpretation

- Full-grid cooperative temporal blocking remains infeasible: the grid
  that needs a global barrier is still far larger than the resident
  cooperative launch capacity.
- Thread-block clusters are real on this platform and can synchronize
  inside one cluster, but they are not a grid-wide barrier.
- A cluster route can only reopen after a separate model proves
  cluster-local ownership for `p_mid`/velocity/CPML state and handles
  source injection, receiver extraction, shell/PML reconciliation, and
  cross-cluster boundary dependencies.

## Next Gate

Allowed next work is design-only:

```text
cluster-local ownership byte/synchronization model
```

Do not write a cluster CUDA prototype until that model shows a
`>=5%` repeat-speedup ceiling after boundary and synchronization
costs are included.
