# Pressure State Representation Gate

## Context

- sampled main: `297.248us`
- p_core share: `31.47%`
- v_pml share: `21.95%`
- pressure-PML share: `46.58%`
- len16 packed pressure-PML share: `22.13%`
- formal current-best WP speedup vs zmem: `1.1928x`

Current second-order pressure update state traffic per point:

| item | bytes |
| --- | ---: |
| `p_prev_read` | `4` |
| `p_cur_read` | `4` |
| `cw2_read` | `4` |
| `p_next_write` | `4` |
| `total` | `16` |

## Candidate Decisions

| candidate | decision | equivalence | traffic signal | reason |
| --- | --- | --- | --- | --- |
| `delta_pressure_state` | `reject_cuda_prototype` | exact algebra for the time recurrence, but changes stored state | bytes 16 -> 20; sampled effect 0.8957x | It removes the old-p read but adds a delta write; minimum pressure-update bytes rise from 16 to 20 per point before any extra bookkeeping. |
| `scaled_pressure_q_only` | `reject_cuda_prototype` | time update algebra is exact only if p=cw2*q is reconstructed wherever pressure is used | p_core cw2 reconstructions >= 29 per output; p_core+v_pml at risk 53.42% | It can remove one cw2 load at the final update only by adding cw2 reconstruction to p_core/v_pml pressure stencils.  p_core alone would go from one cw2 read per output to at least 29 pressure-value reconstructions per output. |
| `scaled_pressure_dual_p_and_q` | `reject_cuda_prototype` | can preserve existing pressure stencils if both p and q are kept coherent | bytes 16 -> 32; sampled effect 0.6822x | Keeping both representations avoids stencil reconstruction but doubles pressure state write/read work. |
| `first_order_full_domain_velocity_pressure` | `reject_cuda_prototype` | not bitwise equivalent; would replace the current mixed second-order core with a new first-order scheme | save 4B old-p, add >= 24B velocity/core point | It saves one old-p read but introduces full-domain velocity state traffic and a different numerical scheme. |
| `precomputed_cw2dt` | `reject_cuda_prototype` | exact if dt is fixed for a run | The same 4-byte model coefficient is still loaded once per pressure update. | It removes a multiply, not the global-memory traffic that dominates the source profile. |
| `half_or_compressed_cw2` | `reject_for_current_exactness_gate` | requires precision relaxation or a separate quantization proof | ideal len16 cw2-line sampled speedup 1.0282x | Even the ideal len16 cw2-line-only benefit is small, and precision changes are outside the current exactness contract. |
| `cpml_mem_dzz_rescaled_state` | `reject_cuda_prototype` | algebraic rescaling can keep the recurrence but cannot remove one read and one write per step | mem_dzz local speedup required 5.0614x | The CPML state is recursive and has no intra-step reuse target; rescaling changes arithmetic but not state traffic. |

## Gate

- decision: `reject_pressure_state_representation_cuda_prototype`
- reason: No exact pressure-state representation removes old-p/cw2 or mem_dzz traffic without moving larger traffic into p_core/v_pml stencils, adding another state write, changing the numerical scheme, or relaxing precision.

Allowed next directions:

- PML vx/vy round-trip ownership design with a >=5% model before CUDA
- source-aware multi-step or wavefront design only if it solves synchronization/halo ownership
- precision-relaxation study only with an explicit new tolerance policy

Do not continue:

- q=p/cw2 pressure-state prototype under the current variable-cw2 stencil path
- delta pressure state prototype
- first-order full-domain velocity-pressure rewrite as a drop-in optimization
- precomputed cw2dt or compressed cw2 prototype under the current exactness gate
- CPML mem_dzz algebraic rescaling without state-traffic removal
