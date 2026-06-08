# CUDA3D Current-Best Package

## Status

This is a current-best package, not a speed-threshold archive.

```text
branch                         exp/day-20260608-cpml-compact-temporal
head commit                    f637ba115d52852b493867ab4a957113a01142a5
candidate                      current_best_v_pml_len16
mean elapsed                   3.016667s
mean Gradient                  2.111930s
mean WP                        1.988905s
elapsed speedup vs zmem         1.118261x
Gradient speedup vs zmem        1.206588x
WP speedup vs zmem              1.222023x
max rel L2                      6.384336e-07
max abs                         4.768372e-06
all compare pass                True
```

Milestone:

```text
1.5x archive                    False
additional WP speedup to 1.5x   1.227472x
```

## Current-Best Flags

```text
-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2 -DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL -DCUDA3D_CPML_VMEM_DISABLE_MPI -DCUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE -DCUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK -DCUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
```

## Accepted Stack

- `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`
- `CUDA3D_CPML_VMEM_DISABLE_MPI`
- `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
- `CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK`
- `CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK`

## Closed Frontiers

| frontier | decision | key number | report |
| --- | --- | ---: | --- |
| ordinary exact CUDA | `ordinary_exact_cuda_frontier_exhausted_for_micro_routes` | `0` allowed | `docs/day_20260608/ownership_frontier_gate.md` |
| cluster/cooperative | `reject_direct_cooperative_grid_k2_temporal_reopen` | `34.650980x` over capacity | `docs/day_20260609/cluster_cooperative_frontier_gate.md` |
| cluster-local ownership | `reject_cluster_local_temporal_cuda_prototype` | `1.1602x` byte ratio | `docs/day_20260609/cluster_local_ownership_model.md` |
| application-level local | `no_local_application_level_experiment_available_on_single_gpu` | `1` GPU | `docs/day_20260609/application_level_frontier_gate.md` |

## Allowed Next

- Run true multi-GPU batching when a >=2 GPU platform is available.
- Open precision-relaxation only after an explicit tolerance-policy change.
- Propose a fundamentally different ownership representation and pass a new byte/synchronization model before CUDA code.
- Stop the CUDA-core sprint at current-best and package results.

## Do Not Repeat

- ordinary exact-CUDA micro prototypes under the closed route matrix
- direct cooperative-grid K=2 temporal reopen
- cluster-local K=2 temporal DSM prototype
- same-GPU multi-rank oversubscription
- host/setup micro-prototypes without a new >=5% measured hotspot

## Primary Reports

- `reports/day_20260608/formal_vpmlen16_table_20260608_2359/summary.json`
- `docs/day_20260608/ownership_frontier_gate.md`
- `docs/day_20260609/cluster_cooperative_frontier_gate.md`
- `docs/day_20260609/cluster_local_ownership_model.md`
- `docs/day_20260609/application_level_frontier_gate.md`
- `docs/day_20260609/pro_handoff_current_best_frontier.md`

## Recent Commits

```text
f637ba1 (HEAD -> exp/day-20260608-cpml-compact-temporal, origin/exp/day-20260608-cpml-compact-temporal) test(scheduling): gate application level frontier
652ea41 test(cuda): gate cluster local temporal ownership
f59040f test(cuda): gate cluster cooperative primitives
f6a0cbc docs(pml): add current-best handoff report
929219e test(pml): close exact cuda frontier
76e7468 test(pml): gate residual pressure routes
eaac165 test(pml): record vpmlen16 formal speed table
3355359 test(pml): gate post-vlen16 pressure routes
```
