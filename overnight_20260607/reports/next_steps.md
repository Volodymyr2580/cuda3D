# Next Steps For CUDA Core Rewrite

## What We Learned

The stable gain is `CUDA3D_PML_ZMEM_IN_P`, which moves z-memory update responsibility into the pressure PML path and removes redundant z-memory traffic from the velocity PML path. On RTX 5090 `perf6_repeat`, this reproduced about 1.049x WP speedup and 1.047x Gradient speedup against `current_best_reference`.

The follow-up micro-optimizations did not pay off:

- `CUDA3D_PML_ZMEM_V_TILE_PRUNE`: correct, but this workload pruned 0 tiles.
- `CUDA3D_PML_TILE_MASK_FASTPATH`: correct, but extra branching/metadata was slower.
- PML tile block shape sweep: no stable improvement over `32x4x2`.
- p_core block shape sweep: `128x1x2` was only about 1.002x on repeat, below the 2% acceptance threshold.
- `-maxrregcount`: consistently slower, suggesting register capping causes spill or reduces compiler freedom.

## Recommended Structural Directions

1. Build a profiler-guided kernel inventory with Nsight Compute for `p_core`, `v_pml_tile`, and `p_pml_tile`. Capture memory throughput, achieved occupancy, issue stalls, register count, and instruction mix for final zmem binary.
2. Rewrite PML as region-specialized kernels instead of one mixed tile kernel. Separate pure z-face, x-face, y-face, edges, and corners so each kernel has straight-line math and fewer per-point conditionals.
3. Consider fusing velocity PML and pressure PML only where data reuse is real. Do not fuse blindly; first inspect whether memory_d* and velocity fields can stay in registers/shared memory across the update boundary.
4. For core pressure update, test a deeper z-pencil/shared-memory rewrite only if profiler shows global memory bandwidth or z-neighbor reloads dominate. The current simple block-shape sweep suggests launch geometry alone is not the lever.
5. Keep correctness gates unchanged: output file count/size, finite values, rel_l2 <= 1e-5, and debug dump comparison for the first few timesteps when changing PML math.

## Current Best Build

```bash
cd /work/wenzhe/cuda3D
source ./env_5090.sh
cd src
make -B -f makefile.rtx5090 test NVFLAGS="-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2"
```
