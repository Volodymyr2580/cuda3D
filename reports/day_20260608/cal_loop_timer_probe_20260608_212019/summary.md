# Host / Setup Timer Probe

## Program Timing

- elapsed: `2.990s`
- Gradient TIME all: `2.164033s`
- WP computing time: `2.044622s`
- elapsed - Gradient: `0.825967s`

## Process Timers

| stage | seconds |
| --- | ---: |
| `mpi_init` | `0.253563` |
| `main_after_mpi_to_pre_finalize` | `2.418320` |
| `mpi_finalize` | `0.000242` |
| `process_total` | `2.672125` |
| `elapsed_minus_process_total` | `0.317875` |

## Main Setup Timers

| stage | seconds |
| --- | ---: |
| `input_scan` | `0.000026` |
| `gpu_setup` | `0.180329` |
| `input_bcast` | `0.000001` |
| `coeff_init` | `0.000012` |
| `static_alloc` | `0.000194` |
| `root_model_read` | `0.018371` |
| `model_bcast` | `0.000001` |
| `acqui_read` | `0.000743` |
| `acqui_bcast` | `0.000000` |
| `lint` | `0.000040` |
| `shot_list` | `0.022557` |
| `total_pre_gradient` | `0.222276` |
| `gradient_call_total` | `2.191751` |
| `post_gradient_barrier_and_free` | `0.001556` |
| `total_after_mpi_to_pre_finalize` | `2.415582` |

## Cal Setup Timers

| stage | seconds |
| --- | ---: |
| `pre_gradient_init` | `0.023527` |

## Cal Loop Timers

| stage | seconds |
| --- | ---: |
| `shots` | `6.000000` |
| `obs_setup` | `0.002679` |
| `domain_setup` | `0.000010` |
| `wavefield_prep` | `0.049816` |
| `fd_call` | `2.089376` |
| `output_write` | `0.004491` |
| `cleanup` | `0.002593` |
| `post_loop_sync` | `0.000004` |
| `copy_reduce` | `0.015053` |

## Accounting

- measured pre-Gradient setup: `0.245803s`
- known non-Gradient time including process shell/MPI/finalize/post-free: `0.819039s`
- unaccounted elapsed-minus-Gradient: `0.580164s`
- residual after known non-Gradient timers: `0.006928s`
