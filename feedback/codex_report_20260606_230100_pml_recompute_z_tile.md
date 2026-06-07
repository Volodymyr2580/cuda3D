# Codex Report: PML Velocity Recompute + Tile List

时间：2026-06-06 23:01 +08:00

## 目标

沿 GPT-5.5 Pro 提出的结构化路线，继续尝试减少 PML 计算中的全局内存往返。

核心想法：

- 原路径先在 `v_pml` 写出 `vx/vy/vz`，再在 `p_pml` 读回。
- 新路径允许 `p_pml` 从 `p1 + velocity memory` 直接重算某个方向的 PML velocity。
- 用宏控制方向，避免一次性把所有路径绑死。

## 修改文件

- `include/inc3D/single_solver.h`
- `src/single_solver.cu`
- `src/rem_fd.cu`
- `tools/run_benchmark.py`
- `tools/compare_debug_dumps.py`

## 新增/扩展的实验开关

- `CUDA3D_PML_RECOMPUTE_Z`
  - `v_pml` 不再写 `vz`。
  - `p_pml` 从 `p1 + d_memory_dz` 重算 `vz` 并用于压力更新。

- `CUDA3D_PML_RECOMPUTE_X`
  - `v_pml` 不再写 `vx`。
  - `p_pml` 从 `p1 + d_memory_dx` 重算 `vx`。

- `CUDA3D_PML_RECOMPUTE_Y`
  - `v_pml` 不再写 `vy`。
  - `p_pml` 从 `p1 + d_memory_dy` 重算 `vy`。

- 可与已有 `CUDA3D_PML_TILE_LIST` / `CUDA3D_PML_TILE_LIST_V` / `CUDA3D_PML_TILE_LIST_P` 组合。

默认构建不启用这些宏，因此默认行为保持原路径。

## Debug Harness

新增 `CUDA3D_PML_DEBUG_DUMP`：

- 在 `p_pml` 后、source injection 前 dump PML 状态。
- 支持环境变量：
  - `CUDA3D_PML_DUMP_DIR`
  - `CUDA3D_PML_DUMP_STEP`
- dump 内容包括 `p0/p1/vx/vy/vz`、velocity memory 和 pressure memory。
- `tools/compare_debug_dumps.py` 用于逐数组比较 dump。

验证结果：

- tile-list vs baseline, `it=0`：所有 dump 数组通过。
- recompute-z vs baseline, `it=1`：
  - `p0/p1/vx/vy` 和 memory 数组通过。
  - `vz` dump 不一致是预期行为，因为 recompute-z 有意不再写回 `vz`。
  - 最终输出 correctness 通过，说明被跳过的 `vz` global store 不再是数值依赖。

## 性能矩阵

冻结 RTX 5090 perf baseline：

- baseline run: `benchmarks/baselines/current_runnable/perf_1gpu_rtx5090_baseline_20260606_002902`
- baseline `WP = 0.545397s`
- baseline `Gradient = 0.576524s`

| Variant | Correctness | Perf output compare | WP(s) | Gradient(s) | WP speedup |
|---|---:|---:|---:|---:|---:|
| `RECOMPUTE_X` | Pass | Pass | 0.627311 | 0.657990 | 0.869x |
| `RECOMPUTE_Y` | Pass | Pass | 0.652586 | 0.683030 | 0.836x |
| `RECOMPUTE_Z` | Pass | Pass | 0.518128 | 0.547818 | 1.053x |
| `RECOMPUTE_XYZ` | Pass | Pass | 0.673621 | 0.703244 | 0.810x |
| `RECOMPUTE_Z + TILE_LIST_V` | Pass | Pass | 0.507620 | 0.538299 | 1.074x |
| `RECOMPUTE_Z + TILE_LIST(V+P)` | Pass | Pass | 0.507413 | 0.538789 | 1.075x |

同场复测：

| Case | Variant | WP(s) | Gradient(s) | Output compare |
|---|---|---:|---:|---:|
| `perf_1gpu` | default retest | 0.547345 | 0.578054 | Pass vs frozen baseline |
| `perf_1gpu` | best repeat | 0.506966 | 0.537213 | Pass vs frozen baseline |
| `perf_1gpu_6shots` | default retest | 2.705801 | 2.853960 | reference |
| `perf_1gpu_6shots` | best | 2.506537 | 2.629992 | Pass vs default retest |

6-shot 同场 speedup：

- WP: `2.705801 / 2.506537 = 1.0795x`
- Gradient: `2.853960 / 2.629992 = 1.0852x`

## 当前最佳构建哈希

远端 `/work/wenzhe/cuda3D` 当前二进制为最佳候选宏构建。

```text
bin/cuda_3D_FM                         c67cecd2fe675267c7d32d8ece70300ee74b30ee0c4e77bfe77b7f0a98b7f271
include/inc3D/single_solver.h          b6aa858d68b123f2744ca0b65d68d151089dc6c9432013da0f071d725f051379
src/single_solver.cu                   db7cde40938315fe0891d442f84903cc2d198b4653ca39073b23b0fc4ce7c66b
src/rem_fd.cu                          7bad299265a7781dde3be874f45cb52db7b30229560462458447dfcfb5d13c3a
tools/run_benchmark.py                 90a02d4d2b2cc77ed60bbf75718b86f9a70cf28d1aeb34f46e3ceee6eecfcd5f
tools/compare_debug_dumps.py           238a65ce10458aa90136a20ef80f2d4935681cbf3cea38a776a1be77e681ba65
```

## 结论

当前可保留的最佳候选是：

```bash
NVFLAGS="-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2"
```

它数值通过，1-shot 和 6-shot 都稳定快于默认路径，但只达到约 `1.08x`，没有达到 `1.5x` 存档阈值。

`RECOMPUTE_X/Y` 方向不建议继续作为全局重算方案推进。它们数值正确，但明显变慢，说明跨 x/y stride 重算造成的额外 `p1` 访问和算术开销大于省掉的 velocity store/load。

## 下一步建议

要继续突破瓶颈，应该转向更大结构：

1. 只对 z 方向保留 recompute，继续减少 PML launch/CTA overhead。
2. 在 PML 区做方向专用 kernel，避免一个 kernel 同时带上 x/y/z 三方向复杂分支。
3. 考虑把 core + z-PML 的压力更新融合成更少的 kernel launch，但需要严控 halo 与 source injection 时序。
4. temporal blocking 仍是理论上更大的提速方向，但需要重写时间推进数据流，风险明显高于当前局部结构改写。
