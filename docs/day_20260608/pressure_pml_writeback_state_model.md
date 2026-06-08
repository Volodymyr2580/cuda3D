# Pressure-PML Writeback / CPML State Gate

## Context

- profile: `len16`
- sampled main: `297.248us`
- len16 packed pressure-PML: `65.771us` / `22.13%`
- total pressure-PML: `138.453us` / `46.58%`
- target sampled-main speedup gate: `1.0500x`
- packed len16 kernel speedup required for target: `1.2742x`
- total pressure-PML speedup required for target: `1.1139x`

## Source Sample Groups

Parsed source samples: `15712`

| group | share of len16 source samples | interpretation |
| --- | ---: | --- |
| final `p0/p1/cw2` update | `60.78%` | required second-order pressure writeback and model load |
| CPML `mem_dzz` update | `26.82%` | recursive z-PML state update |
| z-cache shared loads | `1.92%` | visible but no longer dominant after direct-fill cache |
| address/control visible lines | `4.31%` | tile/address/branch overhead visible in source page |
| other/unparsed | `6.17%` | all remaining sampled lines |

Required local speedups if only one group is improved:

| group | local speedup required | note |
| --- | ---: | --- |
| final `p0/p1/cw2` group alone | `1.5482x` | would require removing a large part of the time-update traffic |
| CPML `mem_dzz` group alone | `5.0614x` | effectively requires eliminating most recursive state traffic |
| final + `mem_dzz` together | `1.3257x` | only plausible through a broader state representation change |

## Amdahl Ceilings

| scenario | packed-kernel speedup | sampled-main speedup | status |
| --- | ---: | ---: | --- |
| `eliminate_final_p0_p1_cw2_update_utopian` | `2.5498x` | `1.1554x` | utopian |
| `eliminate_cpml_mem_dzz_update_utopian` | `1.3665x` | `1.0631x` | utopian |
| `eliminate_both_final_and_mem_dzz_utopian` | `8.0657x` | `1.2404x` | utopian |
| `make_mem_dzz_2x_faster` | `1.1549x` | `1.0306x` | modeled |
| `make_final_update_2x_faster` | `1.4366x` | `1.0721x` | modeled |
| `make_final_plus_mem_dzz_1_25x_faster` | `1.2124x` | `1.0403x` | modeled |
| `make_final_plus_mem_dzz_1_5x_faster` | `1.4124x` | `1.0691x` | modeled |

## Concrete Route Check

| route | status | known/measured signal | reason |
| --- | --- | ---: | --- |
| `p0_read_syntax_ldg` | `rejected_by_existing_perf_repeat` | 1.0001x | Old-p0 read-only load was already tested on the pressure-PML path and landed at noise level. |
| `cpml_local_new_mem_accumulation` | `rejected_by_existing_perf_repeat` | 1.0006x | Explicit local new_mem expression did not change the memory dependency enough to matter. |
| `ptxas_cache_policy_sweep` | `rejected_by_existing_perf_repeat` | dlcm_ca=0.9993x, dlcm_cg=0.8593x | Global load cache policy does not remove the final writeback or CPML state dependency. |
| `branch_only_lower_upper_specialization` | `reject_without_cuda` | n/a | Visible branch/control source samples are too small and would add tile-list and launch overhead. |
| `remove_old_p0_or_cw2_traffic` | `requires_math_or_state_representation_redesign` | n/a | Second-order pressure update needs old pressure state, current pressure state, velocity model, and a new pressure write.  A syntax rewrite only moves this traffic; it does not remove it. |
| `remove_mem_dzz_state_traffic` | `requires_cpml_model_redesign_or_precision_relaxation` | n/a | mem_dzz is a recursive CPML state.  It has one use per step and is already contiguous in the len16 z-line mapping, so shared/cache micro-tuning has no proven reuse target. |

## Gate

- decision: `reject_writeback_state_micro_cuda_prototype`
- reason: The only groups large enough to matter are mathematically required pressure writeback traffic and recursive CPML z-state traffic.  Existing syntax/cache variants already show noise-level gains, while a >=5% sampled-main gain would require about 1.2742x speedup of the packed len16 kernel or 1.3257x speedup of the combined final-writeback+mem_dzz group.
- reopen condition: Reopen only for a state-representation or time-integration design that proves the old-p0/cw2 or mem_dzz traffic is actually removed, not merely reloaded through a different syntax, and has a >=5% perf_1gpu_6shots repeat ceiling after extra storage/control costs.

Allowed next directions:

- math-level state representation design for pressure update, with equivalence proof before CUDA
- PML vx/vy round-trip ownership redesign that reduces global traffic without doubling component work
- formal same-session benchmark table for zmem/direct-fill/len16/current best before a larger phase switch

Do not continue:

- len16 p0 __ldg or old-p0 read syntax retry
- len16 explicit local new_mem retry
- ptxas cache-policy retry for this path
- branch-only lower/upper z-PML specialization without a new >=5% model
- shared z-cache or z-cache fill micro-tuning inside the accepted len16 kernel
