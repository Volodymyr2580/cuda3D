# Post-VLen16 Pressure Next Gate

## Context

- profile: `v_pml_len16_short`
- duration unit: `us`
- sampled main total: `284.010us`
- target local gate: `1.0500x` sampled-main speedup

| region | duration | sampled-main share | local speedup required for 5% sampled-main |
| --- | ---: | ---: | ---: |
| `p_core` | `93.730us` | `33.00%` | `1.1686x` |
| `p_pml_len16` | `66.180us` | `23.30%` | `1.2568x` |
| `p_pml_residual` | `71.940us` | `25.33%` | `1.2315x` |
| `p_pml_total` | `138.120us` | `48.63%` | `1.1085x` |
| `v_pml_len16` | `20.030us` | `7.05%` | `3.0788x` |
| `v_pml_residual` | `32.130us` | `11.31%` | `1.7269x` |
| `v_pml_total` | `52.160us` | `18.37%` | `1.3500x` |

## Pressure Source Anchor

- source profile: `reports/day_20260608/len16_source_profile_20260608_1646/source_page_cuda_sass.txt`
- parsed samples: `15712`

| len16 source group | share of packed pressure kernel samples |
| --- | ---: |
| final `p0/p1/cw2` update | `60.78%` |
| CPML `mem_dzz` update | `26.82%` |
| z-cache shared loads | `1.92%` |
| visible address/branch control | `5.49%` |
| other/unparsed | `4.98%` |

Required source-group speedup to make packed pressure-len16 alone clear the 5% sampled-main gate:

| group | required speedup |
| --- | ---: |
| `final_p0_p1_cw2_if_alone` | `1.5065x` |
| `cpml_mem_dzz_if_alone` | `4.2007x` |
| `final_plus_mem_dzz` | `1.3043x` |

## Scenario Ceilings

| scenario | sampled-main speedup | interpretation |
| --- | ---: | --- |
| `p_pml_total_1_10x` | `1.0463x` | barely below the 5% gate; pressure needs more than a small local win |
| `p_pml_total_1_25x` | `1.1077x` | would be meaningful, but requires real state/writeback reduction |
| `p_pml_len16_1_25x` | `1.0489x` | packed pressure-len16 alone is almost but not quite enough |
| `p_pml_residual_1_25x` | `1.0534x` | barely clears the gate in the model, but requires a large residual-only gain |
| `len16_final_update_2x` | `1.0762x` | mathematically interesting, but syntax variants already failed |
| `len16_mem_dzz_2x` | `1.0323x` | not enough by itself |
| `len16_final_plus_mem_dzz_1_5x` | `1.0730x` | large enough only for a broader state representation change |

## Fusion Sanity Check

| scenario | sampled-main speedup | decision |
| --- | ---: | --- |
| `perfect_remove_v_len16_time` | `1.0759x` | theoretical headroom, but direct z-face VP fusion is a rejected family |
| `perfect_remove_all_v_pml_time` | `1.2250x` | large upper bound, requires a materially new wave-step ownership model |
| `halve_all_v_pml_time` | `1.1011x` | large in theory, but far beyond what descriptor packing has modeled |

## Concrete Route Gate

| route | decision | reason |
| --- | --- | --- |
| `pressure_writeback_syntax_microtuning` | `reject` | p0 __ldg and local new_mem variants were already tested at noise-level speedup. |
| `pressure_len23_or_exact_descriptor` | `reject` | Post-len16 compact descriptor budget predicts only about 1.5% calibrated sampled-main speedup. |
| `pressure_branch_only_specialization` | `reject` | Residual branch efficiency is already 83.32%; branch-only routes do not remove the dominant final update/state dependency. |
| `v_pml_descriptor_expansion` | `reject` | After v-len16, total v-PML share is only 18.37% and descriptor overhead has no >=5% repeat-speedup model. |
| `direct_v_pressure_zface_fusion` | `reject_as_previous_failed_family` | Naive fusion belongs to the already rejected z-face VP fusion/shared-VP family unless a new ownership model avoids duplicate halo/control costs. |
| `new_pressure_or_wave_step_ownership_model` | `allow_design_only` | Pressure-PML remains 48.63% of sampled main; only a model that removes real state/writeback traffic should reach the next CUDA prototype gate. |
| `formal_current_best_benchmark_table` | `allow` | Current cumulative speedup is partly multiplicative across sessions; a same-session zmem/direct-fill/pressure-len16/current-best table is needed before a major phase switch. |

## Gate

- decision: `no_new_micro_cuda_prototype`
- reason: The post-vlen16 profile leaves pressure-PML as the dominant target, but the remaining pressure time is concentrated in required pressure writeback and recursive CPML z-state operations.  Existing syntax/cache/descriptor branches do not meet the >=5% repeat gate.
- next action: Run a formal same-session zmem/direct-fill/pressure-len16/current-best benchmark table, then open only a design-level pressure/wave-step ownership model with an explicit >=5% ceiling.

Allowed next:

- formal same-session benchmark table for zmem, direct-fill, pressure-len16, and current-best
- design-only pressure state/writeback ownership model with equivalence proof
- design-only wave-step ownership model that reduces vx/vy or pressure writeback global traffic

Prohibited next:

- repeat p0 __ldg, local new_mem, ptxas cache-policy, z-cache fill, or shared-z-cache tuning
- pressure length-23 or exact active-point descriptor CUDA prototype
- v-PML descriptor/point-list expansion after accepted v-len16
- direct z-face VP fusion/shared-VP retry without a materially new ownership proof
- random p-core block/register/shared-plane sweeps from the rejected family
