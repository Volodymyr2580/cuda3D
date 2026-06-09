# Precision Relaxation Policy Proposal

## Status

No mixed-precision CUDA code is authorized yet.

This document is a proposal only.  Implementation requires explicit user
approval of a relaxed tolerance policy.

## Current Policy: Tier 0

```text
relative L2 <= 1e-5
finite outputs only
no NaN / Inf
receiver output FP32
```

This is the policy used by `current_best_v_pml_len16`.

## Proposed Tolerance Tiers

### Tier 0: Exact Current

- Relative L2 `<= 1e-5`.
- Current acceptance policy.
- Use for release-quality exact-FP32 comparisons.

### Tier 1: Conservative Relaxed

- Relative L2 `<= 1e-4`.
- Add max-absolute bound per output.
- Receiver waveform visual check.
- Gradient sanity check.
- Only accept if downstream inversion behavior remains scientifically stable.

### Tier 2: Aggressive Exploratory

- Relative L2 `<= 1e-3`.
- Exploratory speed study only.
- Not a release default.
- Requires explicit labeling in reports and outputs.

## Candidate Arrays

| array class | first-study precision | risk | upside |
| --- | --- | --- | --- |
| CPML velocity memory | FP16/BF16 storage, FP32 compute | medium | memory traffic reduction |
| CPML pressure memory | FP16/BF16 storage, FP32 compute | medium-high | pressure-PML traffic reduction |
| PML velocity auxiliary fields | FP16/BF16 storage | medium | v-PML memory reduction |
| coefficients | FP16/BF16 or quantized FP32 table | medium | cache footprint reduction |
| pressure `p_curr/p_next` | keep FP32 initially | high | do not touch first |
| receiver outputs | keep FP32 | high | preserve validation path |

## Required Study Before CUDA Code

For each candidate array:

- bytes allocated
- read/write frequency per step
- NCU/source hotness
- conversion cost
- expected numerical sensitivity
- expected speed ceiling

## First Safe Experiment If Approved

Start with CPML auxiliary memory storage only:

```text
storage precision        FP16 or BF16
compute precision        FP32
pressure p0/p1           FP32
receiver output          FP32
correctness tiers        Tier 1 and Tier 2 reported separately
```

Do not mix this with unrelated CUDA kernel rewrites in the same experiment.
