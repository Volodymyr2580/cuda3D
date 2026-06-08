# True Multi-GPU / Multi-Job Batching Protocol

## Decision

- decision: `defer_true_multigpu_validation_current_platform_single_gpu`
- available GPUs on current RTX 5090 server: `1`

The current platform cannot validate true multi-GPU batching because it exposes only one GPU.
Same-GPU multi-rank oversubscription has already been rejected, so this protocol is for a future multi-GPU run.

## Current-Best Anchor

- alias: `len16_current_best`
- mean elapsed: `2.970s`
- mean Gradient: `2.155902s`
- mean WP: `2.031753s`
- WP speedup vs zmem: `1.1928x`

## Existing Code Requirements

- `src/main.cu` maps ranks with `cudaSetDevice(mytid % gpus_p_node)`.
- `gpus_p_node` is read from the input file, not inferred from `CUDA_VISIBLE_DEVICES`.
- For true one-rank-per-GPU runs, `mpirun -np N`, `CUDA_VISIBLE_DEVICES` with `N` devices, and the input file's last line `gpus_p_node=N` must agree.
- Shot assignment uses `sht_num[is * ntids + mytid]`, so rank count directly changes shot distribution.
- Printed `WP computing time` is root-rank local for multi-rank runs; elapsed and `Gradient TIME all` are the scheduling metrics.

## Shot-Balance Model

| GPUs/ranks | runnable here | active shots/rank | ideal speedup | ideal efficiency | ideal elapsed | ideal Gradient |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `True` | `[6]` | `1.0000x` | `100.0%` | `2.970s` | `2.155902s` |
| 2 | `False` | `[3, 3]` | `2.0000x` | `100.0%` | `1.485s` | `1.077951s` |
| 3 | `False` | `[2, 2, 2]` | `3.0000x` | `100.0%` | `0.990s` | `0.718634s` |
| 4 | `False` | `[2, 2, 1, 1]` | `3.0000x` | `75.0%` | `0.990s` | `0.718634s` |
| 6 | `False` | `[1, 1, 1, 1, 1, 1]` | `6.0000x` | `100.0%` | `0.495s` | `0.359317s` |

## Gate

- Do not run more same-GPU oversubscription probes for this case.
- Do not claim multi-rank speedup from root-rank printed `WP computing time`.
- Promote true multi-GPU batching only after a 3-round repeat where all output comparisons pass and elapsed or `Gradient TIME all` speedup is `>=1.05x`.

## Minimal Future Command Shape

```bash
source ./env_5090.sh
# create an input copy whose last line is N, e.g. 3 for three visible GPUs
CUDA_VISIBLE_DEVICES=0,1,2 python3 tools/run_benchmark.py \
  --case perf_1gpu_6shots --input input_perf_1gpu_6shots_gpus3.in \
  --np 3 --gpus 0,1,2 --tag true_multigpu_np3
```

The input override is necessary because `run_benchmark.py --gpus` controls `CUDA_VISIBLE_DEVICES`, while the CUDA device mapping inside the program uses the input file's `gpus_p_node`.
