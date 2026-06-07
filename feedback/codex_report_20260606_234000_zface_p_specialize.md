# Codex Report: z-face PML Pressure Specialization

时间：2026-06-06 23:40 +08:00

## 目标

按 Pro 的下一步路线执行：

1. 对当前 best 做更细 profile。
2. 实现受控实验 `CUDA3D_PML_ZFACE_P_SPECIALIZE`。
3. 只把 z-PML face 中 `x/y` 位于 core 区域的 pressure update 拆到专用 kernel。
4. `x/y` face、edge、corner 和 residual 区域继续走 generic PML tile-list。
5. 跑 4 个 block shape，并按 `perf_1gpu_6shots` 决策是否继续 `ZFACE_V_SPECIALIZE`。

## Profile 结果

当前 best 宏：

```bash
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
```

Nsight Systems：

- profile dir: `benchmarks/profiles/pml_recompute_z_tile_20260606`
- `cuda_fd3d_p_pml_tile_ns`: 46.7% GPU kernel time
- `cuda_fd3d_v_pml_tile_ns`: 30.0% GPU kernel time
- `cuda_fd3d_p_core_ns`: 22.3% GPU kernel time
- `cudaLaunchKernel`: 2004 calls, total 80.37 ms

判断：

- `p_pml_tile` 仍是最大单项，zface pressure 专用化值得测试。
- launch API 时间不是绝对禁止增加一个 kernel 的程度。

Nsight Compute：

- 命令已执行，但失败于系统权限：
  - `ERR_NVGPUCTRPERM`
  - 当前用户没有 NVIDIA GPU performance counter 访问权限。
- 因此没有可用的 register/occupancy/DRAM throughput NCU CSV。

## 实现

新增/修改文件：

- `include/inc3D/cu_common.h`
  - 新增 `PmlZFaceBlockSize1/2/3` 默认宏。
- `include/inc3D/single_solver.h`
  - 新增 `cuda_fd3d_p_pml_zface_ns` 声明。
- `src/single_solver.cu`
  - 新增 `pml_zface_p_special_point`。
  - 新增 `cuda_fd3d_p_pml_zface_ns`。
  - generic `p_pml`/`p_pml_tile` 在宏开启时跳过 zface-special 点，避免重复更新。
  - 后续优化：将 skip 放在 core early-return 后，并加 tile-level gate，减少无关线程开销。
- `src/rem_fd.cu`
  - 新增 zface pressure tile-list 构建。
  - 新增 zface pressure kernel launch。
  - 新增 zface tile-list 释放。

实验宏：

```bash
-DCUDA3D_PML_ZFACE_P_SPECIALIZE
```

## Debug Dump

使用当前 best 作为 debug baseline：

- baseline: `RECOMPUTE_Z + TILE_LIST`
- candidate: `RECOMPUTE_Z + TILE_LIST + ZFACE_P_SPECIALIZE + 8x8x4`

结果：

| Step | Report | Pass |
|---:|---|---:|
| 0 | `benchmarks/reports/debug_dump_zface_p_8x8x4_vs_best_it0` | True |
| 1 | `benchmarks/reports/debug_dump_zface_p_8x8x4_vs_best_it1` | True |
| 2 | `benchmarks/reports/debug_dump_zface_p_8x8x4_vs_best_it2` | True |

说明：

- zface-special 与当前 best 在 smoke debug dump 上通过。
- correctness 输出非 bitwise identical，但 `rel L2` 约 `1e-7` 量级，低于 `1e-5` 门槛。

## Sweep 结果

当前 best 6-shot baseline：

- run: `benchmarks/runs/perf_1gpu_6shots_recompute_z_tile_all_20260606_230008`
- `WP = 2.506537s`
- `Gradient = 2.629992s`

| Variant | correctness | perf1 compare | 6shot compare | perf1 WP | perf1 Grad | 6shot WP | 6shot Grad | 6shot WP vs best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `8x8x4` | Pass | Pass | Pass | 0.585157 | 0.614705 | 2.865097 | 2.994406 | 0.875x |
| `16x8x2` | Pass | Pass | Pass | 0.555189 | 0.584692 | 2.723829 | 2.854656 | 0.920x |
| `16x4x4` | Pass | Pass | Pass | 0.554979 | 0.585131 | 2.694538 | 2.824677 | 0.930x |
| `8x16x2` | Pass | Pass | Pass | 0.586863 | 0.617514 | 2.844403 | 2.984499 | 0.881x |
| `16x4x4 gated` | Pass | Pass | Pass | 0.553700 | 0.583606 | 2.712018 | 2.838111 | 0.924x |

结论：

- 所有 zface pressure candidates 数值通过。
- 所有 zface pressure candidates 都慢于当前 best。
- 即使把 generic skip 后移并加 tile-level gate，仍然慢于 current best。

## 决策

按 Pro 的决策规则，本轮属于：

```text
情况 C：
ZFACE_P_SPECIALIZE 变慢
=> 回滚/停止；说明多一个 launch 或 tile 切分成本超过收益。
```

执行结果：

- 不继续做 `CUDA3D_PML_ZFACE_V_SPECIALIZE`。
- 不进入 core + z-PML pressure fusion 的前置条件。
- 远端二进制已恢复为当前 best：

```bash
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
```

恢复后 quick perf：

- run: `benchmarks/runs/perf_1gpu_restore_best_after_zface_20260606_233741`
- `WP = 0.506107s`
- `Gradient = 0.535563s`
- output compare vs frozen perf baseline: Pass

## 当前哈希

远端 `/work/wenzhe/cuda3D` 当前二进制是恢复后的 current best 构建，源码保留未启用的 zface 实验宏。

```text
bin/cuda_3D_FM                5d2b7c5e4e0fdfd1590bb6e736c21ba11b0d85ba92dc564d4c6a0227ff305a71
include/inc3D/cu_common.h     b184a1b52697982f1e18caa73db8b7b0127f81241bfd1d3458b40b8dc2293180
include/inc3D/single_solver.h 34b50fad827e6895be922274fc88c8a140cecf9ad3ed1d898e2fed79d99f6fe2
src/single_solver.cu          cfa076ea6a863e7e1f3f920d8c4409950d10ea13841101456c298fa409f77bc7
src/rem_fd.cu                 1bbbfb909b4c0086c14eb1a252b10c2e191358117594d990a773fdb9c5213a4e
```

## 下一步建议

这轮实验证明：在当前 tile-list 框架内继续拆 zface pressure 不是有效路线。

建议下一步不要继续拆 PML faces，而是基于 nsys 结果做更大的方向选择：

1. `p_pml_tile` 仍占 46.7%，但 zface split 变慢，说明拆小 kernel 不是免费收益。
2. `v_pml_tile` 占 30.0%，但只有 z recompute 是正收益，x/y recompute 已退化。
3. `p_core` 已占 22.3%，如果继续压 PML，收益上限可能很快被 core 接管。

更值得进入设计阶段的是：

- 重新设计 PML generic kernel 的 tile shape/occupancy，而不是继续拆 face。
- 或转向 core stencil/time blocking/fusion 的大结构实验。

