# CUDA3D - 3D CUDA Wavefield Forward Modeling

## Project Overview

3D acoustic wavefield forward modeling using finite-difference method with PML boundaries, MPI multi-shot scheduling, and CUDA GPU acceleration.

**Current Platform:** RTX 5090 single-GPU server (`/work/wenzhe/cuda3D`)
**Baseline Goal:** 5x speedup on `perf_1gpu` (WP 0.545s → ~0.109s)
**Correctness Threshold:** rel L2 <= 1e-5, no NaN/Inf

## Quick Start

```bash
cd /work/wenzhe/cuda3D
source ./env_5090.sh
cd src
make -B -f makefile.rtx5090 test
```

Run smoke test:
```bash
python tools/run_benchmark.py --case smoke_1gpu --tag mytest
```

## Directory Structure

```
src/                    # CUDA source and build files
  main.cu               # Entry point
  rem_fd.cu             # Host-side timestepping driver
  single_solver.cu      # CUDA kernels (v_pml, p_pml, p_core)
  makefile.rtx5090      # RTX 5090 build (CUDA 13.0, sm_120)
include/inc3D/          # Headers
  single_solver.h       # Kernel declarations
  cu_common.h           # Block sizes, constants
  common.h, alloc.h     # MPI/IO helpers
tools/                  # Benchmark and utility scripts
  run_benchmark.py      # Unified benchmark runner
  compare_outputs.py    # Numerical comparison (rel L2)
  create_benchmark_cases.py
  create_smoke_case.py
  sweep_blocksize.py
  archive_speedup.py
benchmarks/
  baselines/current_runnable/   # Frozen baselines
  cases/                        # Test case inputs
  runs/                         # Benchmark run outputs
  reports/                      # Comparison reports
  profiles/                     # nsys profiles
archives/speedups/        # Threshold archives (1.5x, 2.0x provisional)
```

## Architecture

### Time-Stepping Loop (`fd_3d_f` in `rem_fd.cu`)

Each time step launches:
1. `cuda_fd3d_v_pml_ns` — velocity update (full domain, PML branches)
2. `cuda_fd3d_p_core_ns` — pressure update (core box only)
3. `cuda_fd3d_p_pml_ns` — pressure update (PML region only)
4. `lint3d_inject_bell_extract_gpu_zz` — source injection / receiver extraction

### Key Kernels (`single_solver.cu`)

| Kernel | Domain | Primary Bottleneck |
|--------|--------|-------------------|
| `cuda_fd3d_v_pml_ns` | Full grid | ~37% GPU time, global memory traffic |
| `cuda_fd3d_p_pml_ns` | PML shell | ~36% GPU time, PML branch divergence |
| `cuda_fd3d_p_core_ns` | Core box | ~26% GPU time, already optimized |

### PML Data Flow

- 12 PML coefficient arrays stored in **CUDA constant memory** (optimized)
- Velocity arrays `vy, vx, vz` and pressure `p0, p1` stored in global memory
- Each kernel reads/writes global memory for all grid points it touches

## Baselines (RTX 5090)

| Case | WP Time | Gradient Time | Purpose |
|------|---------|---------------|---------|
| `smoke_1gpu` | ~0.002s | ~0.003s | Sanity check |
| `correctness` | ~0.011s | ~0.013s | Numerical validation |
| `perf_1gpu` | **0.545397s** | 0.576524s | **Speedup target** |

 perf_1gpu config: `384×384×95`, `nt=1501`, 1 shot, `npml=12`, `vmax=4.0`

## Optimization History

### Already Effective (retained in mainline)

1. PML coefficients → constant memory
2. Remove unused PML coefficient device allocations
3. `p_core` launch only core box (not full domain)
4. `--use_fast_math` compiler flag
5. Block size `128x2x1`

### Already Tried and Reverted (do not repeat)

- PML 6-slab launch, shared-memory full tile, `__restrict__`, per-shot CUDA Graph,
  compact shell 1D mapping, `-dlcm=ca/cg`, pressure fusion, pressure streams,
  PML block `16x16x1`, `CorePmlMargin<=2`, block-level core skip

See `AGENT_LOG.md` for full experimental history.

## Workflow Rules

1. **Correctness first:** Every change must pass `correctness` case (rel L2 <= 1e-5)
2. **Profile before optimizing:** Use `nsys profile -t cuda ...`
3. **Benchmark after changing:** Compare against frozen baseline
4. **Log everything:** Append to `AGENT_LOG.md`
5. **Archive milestones:** Create `archives/speedups/` entry at 0.5x thresholds

## Key Scripts

```bash
# Run benchmark
python tools/run_benchmark.py --case perf_1gpu --tag myopt

# Compare against baseline
python tools/compare_outputs.py \
  --baseline benchmarks/baselines/current_runnable/perf_1gpu_rtx5090_baseline_YYYYMMDD_HHMMSS/outputs \
  --candidate benchmarks/runs/perf_1gpu_myopt_YYYYMMDD_HHMMSS/outputs \
  --out benchmarks/reports/perf_1gpu_myopt_latest

# Profile
nsys profile -t cuda --force-overwrite=true -o benchmarks/profiles/myprofile \
  mpirun -np 1 ./bin/cuda_3D_FM < benchmarks/cases/perf_1gpu/input_perf_1gpu.in

# Sweep block sizes
python tools/sweep_blocksize.py --case profile_1gpu --gpus 0 --variants 128x2x1 64x4x1 ...
```

## Environment

- CUDA 13.0 (`/usr/local/cuda-13.0`)
- GPU: NVIDIA GeForce RTX 5090 (32GB), `sm_120`
- MPI: Intel MPI 2021.18
- Python: conda env `cuda3d` (Python 3.11, numpy)
- GCC: 13.3.0
