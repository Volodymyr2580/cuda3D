# Application-Level Scheduling Frontier Gate

## Summary

Current single-GPU RTX 5090 platform has no remaining local
application-level scheduling experiment worth running.

Decision:

```text
no_local_application_level_experiment_available_on_single_gpu
```

## Formal Current-Best Anchor

```text
alias                         current_best_v_pml_len16
rounds                        3
mean elapsed                  3.016667s
mean Gradient                 2.111930s
mean WP                       1.988905s
elapsed speedup vs zmem        1.1183x
Gradient speedup vs zmem       1.2066x
WP speedup vs zmem             1.2220x
max rel L2                     6.384336e-07
```

## Same-GPU Multi-Rank

```text
decision                      reject_same_gpu_multirank_probe
best elapsed speedup          0.9200x
best Gradient speedup         0.9301x
```

Same-GPU MPI oversubscription is slower and remains prohibited.

## True Multi-GPU Batching

Available GPUs now: `1`.

| GPUs/ranks | runnable now | active shots/rank | ideal speedup | ideal elapsed | ideal Gradient |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | `True` | `[6]` | `1.0000x` | `3.016667s` | `2.111930s` |
| 2 | `False` | `[3, 3]` | `2.0000x` | `1.508333s` | `1.055965s` |
| 3 | `False` | `[2, 2, 2]` | `3.0000x` | `1.005556s` | `0.703977s` |
| 4 | `False` | `[2, 2, 1, 1]` | `3.0000x` | `1.005556s` | `0.703977s` |
| 6 | `False` | `[1, 1, 1, 1, 1, 1]` | `6.0000x` | `0.502778s` | `0.351988s` |

True multi-GPU batching is the only application-level route with large
theoretical upside, but it cannot be validated on the current one-GPU
server.

## Host / Setup

```text
process-timer elapsed          3.220000s
process-timer Gradient         2.161705s
outside process wrapper        0.547231s
MPI_Init                       0.254292s
gpu_setup/context              0.186226s
cal-loop wavefield_prep        0.049816s
wavefield_prep ceiling         1.0236x
```

Host/setup and pre-FD loop micro routes remain below the `>=5%` gate
unless a new measured hotspot appears.

## Gate

- decision: `no_local_application_level_experiment_available_on_single_gpu`
- reason: same-GPU oversubscription is slower, true multi-GPU batching needs more visible GPUs, and host/setup/pre-FD local micro routes do not meet the >=5% gate.

Allowed next:
- Run true multi-GPU batching when a >=2 GPU platform is available.
- Open precision-relaxation only after an explicit tolerance-policy change.
- Stop CUDA-core sprint at current-best and package results.

Do not continue:
- same-GPU np=2/3 oversubscription reruns for perf_1gpu_6shots
- claiming speedup from root-rank printed WP in multi-rank runs
- host/setup micro-prototypes without a new >=5% measured hotspot
- true multi-GPU benchmark on current single-GPU platform
