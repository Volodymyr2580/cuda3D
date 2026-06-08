# Host / Setup Timer Probe

## Program Timing

- elapsed: `3.220s`
- Gradient TIME all: `2.161705s`
- WP computing time: `2.045140s`
- elapsed - Gradient: `1.058295s`

## Process Timers

| stage | seconds |
| --- | ---: |
| `mpi_init` | `0.254292` |
| `main_after_mpi_to_pre_finalize` | `2.418194` |
| `mpi_finalize` | `0.000283` |
| `process_total` | `2.672769` |
| `elapsed_minus_process_total` | `0.547231` |

## Main Setup Timers

| stage | seconds |
| --- | ---: |
| `input_scan` | `0.000023` |
| `gpu_setup` | `0.186226` |
| `input_bcast` | `0.000001` |
| `coeff_init` | `0.000012` |
| `static_alloc` | `0.000182` |
| `root_model_read` | `0.018050` |
| `model_bcast` | `0.000001` |
| `acqui_read` | `0.000739` |
| `acqui_bcast` | `0.000000` |
| `lint` | `0.000039` |
| `shot_list` | `0.022546` |
| `total_pre_gradient` | `0.227820` |
| `gradient_call_total` | `2.186518` |
| `post_gradient_barrier_and_free` | `0.001155` |
| `total_after_mpi_to_pre_finalize` | `2.415493` |

## Cal Setup Timers

| stage | seconds |
| --- | ---: |
| `pre_gradient_init` | `0.022299` |

## Accounting

- measured pre-Gradient setup: `0.250119s`
- known non-Gradient time including process shell/MPI/finalize/post-free: `1.053080s`
- unaccounted elapsed-minus-Gradient: `0.808176s`
- residual after known non-Gradient timers: `0.005215s`
