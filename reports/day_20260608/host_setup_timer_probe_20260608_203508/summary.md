# Host / Setup Timer Probe

## Program Timing

- elapsed: `2.980s`
- Gradient TIME all: `2.162907s`
- WP computing time: `2.046621s`
- elapsed - Gradient: `0.817093s`

## Main Setup Timers

| stage | seconds |
| --- | ---: |
| `input_scan` | `0.000024` |
| `gpu_setup` | `0.174303` |
| `input_bcast` | `0.000001` |
| `coeff_init` | `0.000011` |
| `static_alloc` | `0.000190` |
| `root_model_read` | `0.018118` |
| `model_bcast` | `0.000002` |
| `acqui_read` | `0.000739` |
| `acqui_bcast` | `0.000000` |
| `lint` | `0.000039` |
| `shot_list` | `0.022419` |
| `total_pre_gradient` | `0.215846` |
| `gradient_call_total` | `2.188155` |
| `post_gradient_barrier_and_free` | `0.001178` |
| `total_after_mpi_to_pre_finalize` | `2.405179` |

## Cal Setup Timers

| stage | seconds |
| --- | ---: |
| `pre_gradient_init` | `0.022553` |

## Accounting

- measured pre-Gradient setup: `0.238399s`
- unaccounted elapsed-minus-Gradient: `0.578694s`
