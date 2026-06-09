# FreeOSC Cluster Survey 2026-06-09

## Purpose

Prepare the next CUDA3D phase on the department Slurm cluster, with focus on
true multi-GPU shot batching.

This survey is read-only.  No GPU job was submitted and no cluster files were
modified.

## Connection

- Host: `162.105.91.194`
- Port: `26537`
- User: `shengwz`
- Login node: `mu01`
- Home: `/home/shengwz`
- Slurm: `24.05.3`

Password material must not be written into repository files or shell startup
files.  Automation should pass it only through a temporary process environment.

## Slurm GPU Partition

Command summary:

```bash
sinfo -p gpu -eN -O "NodeHost:8,StateLong:14,CPUsState:18,Gres:90,GresUsed:90,Reason:40"
scontrol show partition gpu
squeue -p gpu
sinfo -R
```

Observed partition settings:

- Partition: `gpu`
- Nodes: `gpu[1-6]`
- Max wall time: `2-00:00:00`
- `OverSubscribe=NO`
- `ExclusiveUser=NO`
- Total configured GPU TRES: `41`
- Configured GPU types include `v100s`, `a40`, `a100`, `rtx4090`, `rtx5090`,
  and `10gb` slices.

Interpretation:

- Multiple users can share one physical node because `ExclusiveUser=NO`.
- GPU sharing should still be controlled by Slurm GRES.  A job that correctly
  requests `--gres=gpu:<type>:N` should receive allocated GPU devices instead
  of silently sharing the same GPU with another job.
- For CUDA3D tests, every sbatch script must request GPU GRES explicitly.  Do
  not rely on manual `CUDA_VISIBLE_DEVICES` alone.

## Node State At Survey Time

Survey time on cluster: `2026-06-09 14:50 CST`.

| Node | Hardware | State | GPU allocation | Notes |
| --- | --- | --- | --- | --- |
| `gpu1` | `4x V100S 32GB + 4x A40 48GB` | `mixed` | all 8 GPUs allocated | usable only after current jobs finish |
| `gpu2` | `4x A100 40GB` | `down*` | none | `power_cut`, not responding |
| `gpu3` | `3x A100 40GB + 4x 10gb slices` | `mixed` | all 7 GRES allocated | usable only after current jobs finish |
| `gpu4` | `8x RTX 4090 24GB` | `down*` | none | `power_cut`, not responding |
| `gpu5` | `6x RTX 5090 32GB` | `down*` | none | `power_cut`, not responding |
| `gpu6` | `8x RTX 5090 32GB` | `down*` | none | `power_cut`, not responding |

`gpu5` and `gpu6` are the best match for the current `sm_120` RTX 5090 build,
but both were down during the survey.  `ResumeAfterTime=None` was reported for
the down GPU nodes, so this does not look like a user-triggered automatic
power-up state.

## Current Queue Signal

The GPU partition had multiple running jobs on `gpu1` and `gpu3`, plus pending
jobs.  Example pending estimates included:

- `qijq/work2`: expected start `2026-06-09T20:51:41` on `gpu3`
- several `pengjx/<JOBNAME>` jobs: expected start `2026-06-11T09:31:39`

This means the currently powered GPU nodes are busy.  The RTX 4090/5090 nodes
need administrative recovery before they can be used.

## Software Modules

Observed module families:

```text
cuda/10.1
cuda/11.5
cuda/12.3
cuda/12.6-gpu5
cuda/12.8-gpu6
cuda/13.1-gpu6
oneapi/compiler-2021.4.0
oneapi/mpi-2021.4.0
oneapi/mkl-2021.4.0
intel/2020.4.304
```

Before compiling on an RTX 5090 node, verify inside the allocated GPU job:

```bash
module load cuda/13.1-gpu6
nvcc --version
nvcc --list-gpu-arch | grep sm_120
nvidia-smi
```

If testing on non-RTX5090 nodes, the binary must be rebuilt for the matching
architecture:

| GPU | Suggested architecture |
| --- | --- |
| V100S | `sm_70` |
| A100 | `sm_80` |
| A40 | `sm_86` |
| RTX 4090 | `sm_89` |
| RTX 5090 | `sm_120` |

Do not run an `sm_120`-only binary on A100, A40, V100S, or RTX 4090 nodes.

## Storage

- Home is on BeeGFS: `2.1P` total, `225T` available, `90%` used.
- User quota reported `10.46 GiB` used and `unlimited` hard quota.
- `/home/scratch` exists and is world-writable, but should be treated as
  temporary scratch according to the cluster guide.

## Recommended Next Step

1. Ask the cluster administrator to restore either `gpu5` or `gpu6`.
2. After the node is up, submit a tiny `sbatch` GPU probe, not a full benchmark:
   - request `--partition=gpu`
   - request `--gres=gpu:rtx5090:1`
   - set a short time limit such as `00:03:00`
   - print `hostname`, `nvidia-smi`, loaded CUDA module, and `nvcc --list-gpu-arch`
3. Only after that probe passes, upload/build CUDA3D and run the multi-GPU
   batching validation.

For true batching, prefer one homogeneous node:

```bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:rtx5090:6
```

Six GPUs matches the current six-shot `perf_1gpu_6shots` case with ideal shot
balance `[1,1,1,1,1,1]`.
