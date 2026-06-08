# Host / Setup Overhead Gate

## Decision

- decision: `profile_before_host_setup_prototype`

## Timing Anchor

- current best: `len16_current_best`
- mean elapsed: `2.970s`
- mean Gradient TIME all: `2.155902s`
- mean WP: `2.031753s`
- elapsed - Gradient: `0.814098s` / `27.41%`
- elapsed - WP: `0.938247s` / `31.59%`

Current accepted speedups vs zmem:

- elapsed: `1.1560x`
- Gradient: `1.1792x`
- WP: `1.1928x`

## 5% Gate

- saved time required for `1.05x` elapsed speedup vs current best: `0.141429s`
- required fraction of `elapsed - Gradient`: `17.37%`

| removable fraction | saved s | candidate elapsed | speedup vs current best | speedup vs zmem |
| ---: | ---: | ---: | ---: | ---: |
| `10%` | `0.081410` | `2.888590` | `1.0282x` | `1.1886x` |
| `25%` | `0.203524` | `2.766476` | `1.0736x` | `1.2410x` |
| `50%` | `0.407049` | `2.562951` | `1.1588x` | `1.3396x` |
| `100%` | `0.814098` | `2.155902` | `1.3776x` | `1.5925x` |

## Boundary

- `Gradient TIME all` starts inside `cal_fwi_grad_3d` after early allocation/setup.
- elapsed is the whole MPI program execution measured by `/usr/bin/time` in `run_benchmark.py`.
- output copying done by `run_benchmark.py` happens after the timed command, so optimizing it does not improve this elapsed metric.

## Gate

Do not make blind host/setup code changes yet.  Reopen only after Nsight Systems, CPU sampling, or targeted timers identify a concrete host/setup hotspot with `>=5%` elapsed-speedup ceiling.

Forbidden shortcuts:

- Do not move timing markers and call that a speedup.
- Do not skip output generation or correctness work.
- Do not optimize runner output copy for this elapsed metric.
