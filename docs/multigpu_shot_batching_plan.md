# Multi-GPU Shot Batching Plan

## Goal

Prepare for true multi-GPU shot batching when a platform with `>=2` visible
GPUs is available.

This is an application-level throughput route, not a single-GPU kernel speedup.

## Current Constraint

The current RTX 5090 server exposes one GPU:

```text
GPU 0: NVIDIA GeForce RTX 5090
```

Therefore this plan is prepared but not validated on the current platform.

## FreeOSC Slurm Target

The department `freeosc` cluster was surveyed read-only on 2026-06-09.  See:

```text
docs/freeosc_cluster_survey_20260609.md
```

Important scheduler facts:

- GPU jobs must be submitted through Slurm with explicit GPU GRES requests.
- `ExclusiveUser=NO`, so multiple users can share one node.
- `OverSubscribe=NO`, so correctly requested GPU GRES should not be
  oversubscribed by Slurm.
- Do not run CUDA benchmarks on the login node.
- Current best `sm_120` binaries require RTX 5090 nodes (`gpu5/gpu6`).

Survey result:

- `gpu1` and `gpu3` were powered and mixed, but all configured GPU GRES were
  allocated.
- `gpu4`, `gpu5`, and `gpu6` were `down*` with reason `power_cut`.
- `gpu5/gpu6` must be restored before RTX 5090 batching can be validated.

For the current six-shot performance case, the preferred validation target is:

```bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:rtx5090:6
```

This gives the ideal `[1,1,1,1,1,1]` shot distribution.  If `gpu6` is restored
and eight GPUs are available, still use six GPUs first for the formal
six-shot benchmark; eight GPUs need a larger shot count to be meaningful.

## Code Requirements

The existing program reads `gpus_p_node` from the input file and maps MPI ranks
with:

```text
cudaSetDevice(mytid % gpus_p_node)
```

True one-rank-per-GPU batching requires these three values to agree:

```text
mpirun -np N
CUDA_VISIBLE_DEVICES contains N devices
input file last line gpus_p_node = N
```

Do not use `tools/run_benchmark.py --gpus` alone as a complete multi-GPU
configuration; it only sets `CUDA_VISIBLE_DEVICES`.

## Shot Independence

The `perf_1gpu_6shots` case has six shots.  Shot assignment follows rank count.
Ideal shot balance:

| GPUs/ranks | shot counts | ideal speedup | efficiency |
| ---: | --- | ---: | ---: |
| 1 | `[6]` | `1.0000x` | `100%` |
| 2 | `[3,3]` | `2.0000x` | `100%` |
| 3 | `[2,2,2]` | `3.0000x` | `100%` |
| 4 | `[2,2,1,1]` | `3.0000x` | `75%` |
| 6 | `[1,1,1,1,1,1]` | `6.0000x` | `100%` |

Because there are only six shots, four GPUs do not improve the ideal bound over
three GPUs for this case.

## Output Collision Rules

Before running multi-GPU batching, audit per-rank writes for:

- `d_obs`
- gradient outputs
- run logs
- temporary shot-domain files

Every output must be either rank/shot-specific or generated in an isolated run
directory.  Do not allow multiple ranks to write the same file name unless the
code has an explicit safe reduction step.

## Validation Metrics

Scheduling metrics:

- `/usr/bin/time -v` elapsed wall clock
- `Gradient TIME all`

Diagnostic metric only:

- root-rank printed `WP computing time`

Never claim a multi-rank speedup using only root-rank printed WP.

## Validation Gate

For `N` GPUs:

- Use current-best binary and flags.
- Create an input copy whose final non-empty line is `N`.
- Run with `mpirun -np N`.
- Set `CUDA_VISIBLE_DEVICES` to exactly `N` visible GPUs.
- Run 3 repeats.
- Compare combined outputs against the single-GPU current-best baseline.
- Accept only if all comparisons pass and elapsed or `Gradient TIME all`
  speedup is `>=1.05x`.

## Runner

A safe runner stub exists:

```text
tools/run_multigpu_batching.py
```

It refuses to run on fewer than two visible GPUs and will not fake a true
multi-GPU test by oversubscribing one GPU.
