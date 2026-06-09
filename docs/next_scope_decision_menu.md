# Next Scope Decision Menu

## Current State

```text
current best                  current_best_v_pml_len16
WP speedup vs zmem             1.222023x
Gradient speedup vs zmem       1.206588x
estimated total vs original    about 2.3x, not a direct formal table
single-GPU exact CUDA frontier closed
```

## Menu

### A. Stop And Package Current Best

Use the current-best package as the deliverable.

Choose this if the next phase is writing, reporting, or thesis integration.

### B. Wait For >=2 GPUs And Run True Multi-GPU Shot Batching

This is the most realistic route to exceed `3x` at application throughput.

Requirements:

- at least two visible GPUs
- one MPI rank per GPU
- input `gpus_p_node=N`
- 3-round repeat
- output comparison pass
- elapsed and `Gradient TIME all` speedup reported

### C. Approve Precision Relaxation

Choose this only if scientific tolerance can be relaxed beyond the current
`rel_l2 <= 1e-5` policy.

First target should be CPML auxiliary memory storage, not pressure `p_curr` or
receiver outputs.

### D. New Ownership Representation

A new CUDA-core route must first pass a byte/synchronization model.  It must not
be a renamed version of:

- ordinary K=2 temporal
- cooperative-grid temporal
- cluster-local DSM temporal
- shared VP
- z-face direct fusion

## Recommendation

If the goal is fastest path to `3x`, choose B when multi-GPU hardware is
available.

If the goal is single-GPU speed only, choose C only after approving a tolerance
tier.  Otherwise stop at A and package the current-best result.
