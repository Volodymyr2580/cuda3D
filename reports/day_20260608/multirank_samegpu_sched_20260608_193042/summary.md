# Same-GPU Multi-Rank Scheduling Probe

## Context

Current-best binary was run on one RTX 5090 with `CUDA_VISIBLE_DEVICES=0` and `np=1/2/3`.
This tests whether multiple MPI ranks sharing one GPU can improve six-shot wall-clock throughput.

Important: printed `WP computing time` is root-rank local for `np>1`, so elapsed and `Gradient TIME all` are the scheduling metrics.

## Runs

| run | np | elapsed s | Gradient s | printed WP s | elapsed speedup | Gradient speedup | outputs | shots seen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `np1` | 1 | 2.990 | 2.165543 | 2.048052 | 1.0000x | 1.0000x | 6 | `[0, 1, 2, 3, 4, 5]` |
| `np2` | 2 | 3.370 | 2.311468 | 2.443532 | 0.8872x | 0.9369x | 6 | `[0, 1, 2, 3, 4, 5]` |
| `np3` | 3 | 3.250 | 2.328266 | 2.158150 | 0.9200x | 0.9301x | 6 | `[0, 1, 2, 3, 5, 4]` |

## Correctness

- `np2` vs `np1`: pass `True`, max rel L2 `0.0`, max abs `0.0`.
- `np3` vs `np1`: pass `True`, max rel L2 `0.0`, max abs `0.0`.

## Gate

- decision: `reject_same_gpu_multirank_probe`
- best elapsed candidate: `np3` / `0.9200x`
- best Gradient speedup: `0.9301x`

Single-round probe is promoted only if elapsed or Gradient speedup is `>=1.05x` and comparisons pass.

## Next

Do not pursue same-GPU multi-rank oversubscription. Move to true multi-GPU / multi-job batching design or exactness-policy alternatives.
