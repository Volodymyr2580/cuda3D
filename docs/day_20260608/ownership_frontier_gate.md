# Ownership Frontier Gate

## Context

- current best: `current_best_v_pml_len16`
- formal WP speedup vs zmem: `1.2220x`
- formal Gradient speedup vs zmem: `1.2066x`
- formal elapsed speedup vs zmem: `1.1183x`
- max rel L2: `6.384336e-07`
- additional WP speedup needed to reach `1.5x`: `1.2275x`
- sampled main profile anchor: `284.010us`
- saved time required for `>=5%` sampled-main: `13.524us`

## Region Shares

| region | duration | share | local speedup required for 5% sampled-main |
| --- | ---: | ---: | ---: |
| `p_core` | `93.730us` | `33.00%` | `1.1686x` |
| `pressure-PML total` | `138.120us` | `48.63%` | `1.1085x` |
| `pressure len16` | `66.180us` | `23.30%` | `1.2568x` |
| `pressure residual` | `71.940us` | `25.33%` | `1.2315x` |
| `v-PML total` | `52.160us` | `18.37%` | `1.3500x` |

## Route Matrix

| route | model ceiling | required local | decision | reason |
| --- | ---: | ---: | --- | --- |
| `pressure_final_writeback_state_representation` | `1.0762x` | `1.5065x` | `reject_cuda_prototype` | A 2x final-update subgroup win is mathematically enough in isolation, but exact state representation variants have already moved or increased traffic into p_core/v_pml or added state writes. |
| `cpml_recursive_state_traffic` | `1.0323x` | `4.2007x` | `reject_cuda_prototype` | mem_dzz is a recursive per-step state with no intra-step reuse target; even a 2x local win is below gate. |
| `combined_final_plus_mem_dzz_state_redesign` | `1.0730x` | `1.3043x` | `design_only` | Only a broader state/ownership redesign can plausibly reach the gate, but no exact ordinary-CUDA design has yet shown how to remove both traffic groups without larger side effects. |
| `residual_pressure_branch_or_descriptor` | `1.0429x` | `1.2315x` | `reject_cuda_prototype` | Residual pressure branch/predicate/descriptor cleanups are below the 5% sampled-main gate. |
| `v_pml_descriptor_or_micro_packing` | `n/a` | `1.3500x` | `reject_cuda_prototype` | After accepted v-len16, v-PML share is only 18.37%; descriptor expansion has no >=5% overhead model. |
| `ordinary_v_pressure_zface_fusion` | `1.0759x` | `n/a` | `reject_previous_failed_family` | Direct z-face VP fusion/shared-VP has already failed; ordinary CUDA variants duplicate halo/control work or need cross-CTA values not available in shared/register state. |
| `source_aware_temporal_wavefront` | `1.1248x` | `1.1686x` | `reject_ordinary_cuda` | The ideal K=2 source-aware ceiling is meaningful, but ordinary CUDA schedules either materialize p_mid globally, duplicate halos heavily, or need a grid-wide/cross-CTA ownership primitive. |
| `host_launch_or_stream_scheduling` | `1.0052x` | `n/a` | `reject_cuda_prototype` | The tested async stream prototype was correct but reached only about 0.5% WP speedup. |

## Gate

- decision: `ordinary_exact_cuda_frontier_exhausted_for_micro_routes`
- ordinary CUDA allowed count: `0`
- reason: All remaining ordinary CUDA exact routes either fail the >=5% modeled repeat-speedup gate, repeat a measured failed family, or require cross-CTA/global synchronization semantics not present in the current implementation model.

Next allowed:

- write a handoff report to Pro/next agent with current-best and prohibited routes
- investigate concrete cluster/cooperative persistent-kernel primitives before any cross-CTA ownership prototype
- precision-relaxation plan only if the user explicitly changes the tolerance policy
- application-level batching/multi-shot scheduling outside the CUDA-core exactness track

Stop for now:

- do not start another exact ordinary-CUDA micro prototype
- do not reopen residual pressure, v-PML descriptor, z-face fusion, current p-core shared-plane, or K=2 temporal routes
- do not claim 1.5x archive; current formal WP speedup is 1.222023x
