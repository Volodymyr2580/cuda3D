# Source-Aware Wavefront Synchronization Gate

## Current-Best Rebase

- sampled main: `297.248us`
- p_core: `93.547us` / `31.47%`
- formal current-best WP speedup vs zmem: `1.1928x`
- ideal K=2 p_core pair reduction: `35.25%`
- ideal K=2 sampled-main speedup on current best: `1.1248x`
- p_core reduction required for 1.05x sampled-main: `15.13%`
- fraction of ideal saving required: `42.92%`

## Source / Receiver Compatibility

- aggregate K=2 deep-core share: `73.22%`
- source overlap shots: `0`
- receiver overlap shots: `0`
- verdict: `compatible_for_this_case`

## Synchronization Facts

- p_core grid blocks: `70688`
- conservative resident block capacity: `1360`
- cooperative-grid over-capacity factor: `51.98x`

## Candidate Schedules

| candidate | ordinary CUDA | speedup ceiling | decision | reason |
| --- | ---: | ---: | --- | --- |
| `safe_global_middle_two_kernel` | `True` | `1.0000x` | `reject_cuda_prototype` | Writes p(t+1) globally and reloads its stencil for step 2, so it keeps the global p_mid traffic that temporal blocking was meant to remove. |
| `cooperative_grid_full_core_k2` | `False` | `1.1248x` | `reject_cuda_prototype` | The full p_core grid exceeds conservative resident capacity by about 52x, so a cooperative launch cannot cover the current grid. |
| `cta_local_diamond_k2` | `True` | `0.0886x` | `reject_cuda_prototype` | Concrete CTA-local candidates require 11.29x to 21.30x baseline pair bytes after p_mid halo duplication. |
| `multi_kernel_global_wavefront` | `True` | `1.0000x` | `reject_cuda_prototype` | Layered global wavefronts still need global p_mid materialization between layers and add many small wavefront launches; no p_mid global stencil traffic is removed. |
| `persistent_wavefront_without_global_barrier` | `False` | `1.1248x` | `reject_cuda_prototype` | A persistent kernel cannot let one CTA safely read another CTA's p_mid from shared/register state in ordinary CUDA. |
| `ideal_no_dup_source_aware_wavefront` | `False` | `1.1248x` | `reject_not_ordinary_cuda` | This is the only meaningful ceiling, but it requires non-duplicating p_mid ownership across CTA boundaries without global reloads. |

## Gate

- decision: `reject_source_aware_wavefront_cuda_prototype`
- reason: Source and receiver placement are compatible, and the ideal current-best K=2 ceiling is meaningful. However, every ordinary CUDA schedule either materializes p_mid globally, duplicates p_mid halos by 11x or more, or lacks the grid-wide/cross-CTA ownership primitive required to read p_mid safely.

Allowed next directions:

- application-level multi-shot batching or scheduling
- precision-relaxation study only with explicit tolerance policy
- future hardware/runtime-specific cross-CTA ownership only after a concrete primitive is identified

Do not continue:

- ordinary CUDA K=2 source-aware wavefront prototype
- multi-kernel global-middle wavefront prototype
- CTA-local diamond temporal prototype
- persistent-kernel wavefront relying on cross-CTA shared/register values
