# AGENTS.md

## 项目目标

本项目的长期目标是改写和优化 3D CUDA 波场正演程序，提高计算性能和效率，同时保证数值结果在可接受精度范围内保持一致。

当前阶段的目标不是立刻改 CUDA kernel，而是先建立稳定、可重复、可追溯的测试基准。后续任何性能优化都必须与冻结的 baseline 对比。

## 基准原则

- 当前服务器上的可运行版本作为后续 CUDA 优化的数值金标准。
- 后续优化前必须先冻结 baseline，包括源码哈希、二进制哈希、输入数据哈希、输出数据哈希、运行日志和环境信息。
- 后续修改 CUDA 代码时，必须同时比较性能和数值结果。
- 不能只看运行时间变快，也必须确认输出结果没有超出数值容差。
- 如果测试时 GPU 被其他任务占用，该轮性能数据只能作为参考，不能作为正式性能结论。

## 当前稳定基线

自 2026-06-07 起，RTX 5090 平台的主基线固定为 `zmem_reference`，不再以更早的 `current_best_reference` 作为主比较对象。

当前稳定构建 flags：

```bash
NVFLAGS="-O3 -arch=sm_120 --use_fast_math \
-DCUDA3D_PML_RECOMPUTE_Z \
-DCUDA3D_PML_TILE_LIST \
-DCUDA3D_PML_ZMEM_IN_P \
-DPmlTileBlockSize1=32 \
-DPmlTileBlockSize2=4 \
-DPmlTileBlockSize3=2"
```

当前基线理由：

- `perf6_repeat` 中，`ZMEM_IN_P` 相对 `current_best_reference` 的 WP speedup 约 `1.0493x`，Gradient speedup 约 `1.0469x`。
- zmem 之后的候选没有达到 `>=2%` repeat speedup 门槛。
- 后续所有 CUDA 实验必须相对 `zmem_reference` 报告 correctness、`perf_1gpu_6shots` 和 repeat 结果。

## 下一阶段架构纪律

下一阶段停止随机 CUDA 微调，进入 profiler-guided 的结构重写阶段。

禁止继续投入以下路线，除非有新的 profiler 证据推翻既有结论：

- `CUDA3D_PML_ZMEM_V_TILE_PRUNE`
- `CUDA3D_PML_TILE_MASK_FASTPATH`
- `CUDA3D_PML_ZFACE_P_SPECIALIZE`
- `CUDA3D_PML_ZFACE_V_SPECIALIZE`
- `RECOMPUTE_X` / `RECOMPUTE_Y` / `RECOMPUTE_XYZ`
- PML tile block shape sweep
- `p_core` simple block shape sweep
- `-maxrregcount` / register cap sweep

Profiler gate：

- 没有 Nsight Compute 或等价硬件级 profiler evidence，不启动新的大结构重写。
- PML 数学路径改动必须通过 debug dump step 0/1/2。
- 性能结论必须包含 `perf_1gpu_6shots repeat`。
- 小候选没有 `>=2%` repeat speedup，不进入主线。
- prototype 没有 `>=5%` repeat speedup，不扩展范围。

当前允许的下一阶段 prototype：

1. `CUDA3D_CORE_2STEP_INTERIOR_PROTOTYPE`
   - 只允许 single GPU / single MPI rank。
   - 只允许 core strict interior，默认 guard margin 至少 `2 * CUDA3D_CORE_STENCIL_RADIUS`。
   - dependency map、debug dump、interior compare、debug-only `p(t+2)` prediction 已完成。
   - `CUDA3D_CORE_2STEP_COMMIT_INTERIOR` 已完成 correctness prototype，但性能变慢，只作为调度正确性证明。
   - 下一步只应推进 fused two-step core kernel，目标是在同一 kernel 内计算 `p(t+1)` 与 strict-interior `p(t+2)` 并复用数据。
   - PML、source injection、receiver extraction、`p0/p1` swap 时序必须保持 baseline。
   - source 或 receiver 落入 blocked region 时，第一版必须停止或新建安全 case。
   - 不默认启用，必须宏控制。

已结束或禁止继续的路线：

- `CUDA3D_PML_FUSED_ZSLAB_PROTOTYPE`：correctness pass，但相对 `zmem_reference` repeat 变慢。
- `CUDA3D_PML_FUSED_ZSLAB_SKIP_V_OWNED`：phase 1 已失败，不进入 phase 2。
- `CUDA3D_CORE_ZPENCIL_SHARED`：source-level NCU gate 发现 baseline 已有 z shared tile，不实现。
- `CUDA3D_CORE_2STEP_COMMIT_INTERIOR` standalone predict+copy 性能路线：correctness pass，但不是 speedup 路线，因为 prediction 仍是单独 kernel。
- p_core simple block sweep、PML tile/block/mask/prune sweep、`RECOMPUTE_X/Y/XYZ`、full-domain temporal blocking、MPI temporal blocking。

## 速度阈值存档规则

以 `perf_3gpu` 的冻结 baseline 作为 1.0x：

```text
benchmarks/baselines/current_runnable/perf_3gpu_baseline_heavy_contended_20260601_165602
baseline WP computing time = 3.491814 s
```

速度提升按以下公式计算：

```text
speedup = baseline WP computing time / candidate WP computing time
```

计时口径补充：

- 只有 MPI rank 数、GPU 数、shot 分配方式一致时，才允许直接用 `WP computing time` 判断阈值。
- 如果改变 `mpirun -np`、每 GPU rank 数或 shot 调度方式，`WP computing time` 可能只反映 root rank 的局部计算时间，不能直接代表整作业速度。
- 调度层优化必须同时报告并比较：
  - `WP computing time`
  - `Gradient TIME all`
  - `/usr/bin/time -v` 的 `Elapsed (wall clock) time`
- 调度层版本只有在 `Gradient TIME all` 或 wall-clock 也突破阈值时，才能确认为正式速度阈值存档。
- 若已按 `WP computing time` 误存档，必须追加 validation note，不删除、不覆盖原存档。

存档规则：

- 只有同时通过 `correctness` 与 `perf_3gpu` 数值对比的版本才能存档。
- 每突破一个 `0.5x` 阈值就单独存档，例如 `1.5x`、`2.0x`、`2.5x`、`3.0x`。
- 未达到下一个阈值的中间版本只写入 `AGENT_LOG.md`，不作为正式阈值版本。
- 存档目录统一放在 `archives/speedups/`，命名示例：`1.5x_20260602_013000_opt_name`。
- 每个存档至少包含：
  - 关键源码和头文件快照。
  - benchmark 工具脚本快照。
  - correctness/perf 对比报告。
  - candidate run 的 `manifest.json`、`run.log` 和环境摘要。
  - baseline 与 candidate 的时间、speedup、误差摘要、SHA256。
- 存档操作只允许新增文件和目录，不允许删除或覆盖已有存档。

## 数值正确性要求

默认正确性门槛：

- 输出文件数量必须一致。
- 每个输出 `.dir` 文件尺寸必须一致。
- 所有输出值必须是 finite，不允许出现 `NaN` 或 `Inf`。
- 主判据：相对 L2 误差 `<= 1e-5`。
- 辅助记录：最大绝对误差、最大相对误差、RMS 误差、每炮误差排行。
- 如果某个 baseline 输出全零，则该文件改用绝对误差 `<= 1e-7` 判定。

## 测试层级

后续维护三个层级的测试样例：

1. `smoke`
   - 目的：验证程序链路能跑通。
   - 当前样例：`48 x 48 x 48`，3 炮，每炮 25 个检波点，`nt = 51`。
   - 要求：1GPU 和 3GPU 都能正常完成，日志包含 `ALL DONE`。

2. `correctness`
   - 目的：验证数值结果是否与 baseline 一致。
   - 建议样例：`96 x 96 x 64`，6 炮，每炮 49 个检波点，`nt = 201`。
   - 要求：相对 L2 误差 `<= 1e-5`。

3. `perf_3gpu`
   - 目的：验证 3 张 RTX 4090 上的优化收益。
   - 当前建议样例：`384 x 384 x 95`，9 炮，每炮 441 个检波点，`nt = 1501`。
   - 检波器孔径要足够大，避免程序裁剪出过小子域，导致测到的是 MPI/启动开销而不是 CUDA kernel。
   - 要求：默认每轮测试控制在 10 分钟内。

## 服务器测试环境

### RTX 5090 稳定服务器

后续可将 `/work/wenzhe/cuda3D` 作为新的稳定测试平台。

连接方式：

```bash
ssh -p 25804 -X zz@162.105.95.56
```

项目目录：

```text
/work/wenzhe/cuda3D
```

环境加载：

```bash
cd /work/wenzhe/cuda3D
source ./env_5090.sh
```

当前环境要点：

- 系统：Ubuntu 24.04.4 LTS
- GPU：1 张 NVIDIA GeForce RTX 5090，约 32GB 显存
- CUDA：`/usr/local/cuda-13.0`
- GPU 架构：`sm_120`
- MPI：Intel MPI，`/opt/intel/oneapi/mpi/latest`
- Conda：`/work/wenzhe/miniforge3`
- 项目环境：`cuda3d`
- 构建文件：`src/makefile.rtx5090`
- 可执行文件：`bin/cuda_3D_FM`

常用编译命令：

```bash
cd /work/wenzhe/cuda3D
source ./env_5090.sh
cd src
make -B -f makefile.rtx5090 test
```

当前已完成初始验证：

- `smoke_1gpu` 已在 RTX 5090 上跑通。
- run：`benchmarks/runs/smoke_1gpu_rtx5090_initial_20260606_000133`
- 结果：`ALL DONE`，输出 3 个 `.dir` 文件。
- `WP computing time = 0.002072s`
- `Gradient TIME all = 0.003854s`
- elapsed `0:02.38`

注意：

- 新服务器当前只有 1 张 RTX 5090，因此旧的 `perf_3gpu` 阈值不能直接沿用。
- 需要为 RTX 5090 平台重新冻结 `smoke/correctness/perf_1gpu` baseline。
- 密码不得写入项目文档或脚本；需要自动化时只通过临时环境变量传入。

### 原 RTX 4090 服务器

服务器项目目录：

```text
/data/shengwz/swz/cuda3D
```

当前服务器环境要点：

- 系统：Ubuntu 22.04.5 LTS
- GPU：4 张 NVIDIA GeForce RTX 4090，每张约 24GB 显存
- CUDA 编译器：`/usr/local/cuda-12.2/bin/nvcc`
- MPI：Intel MPI，`/opt/intel/oneapi/mpi/latest`
- 构建文件：`src/makefile.server`
- 可执行文件：`bin/cuda_3D_FM`

常用编译命令：

```bash
cd /data/shengwz/swz/cuda3D/src
make -f makefile.server test
```

常用 3GPU smoke 运行命令：

```bash
source /opt/intel/oneapi/setvars.sh
cd /data/shengwz/swz/cuda3D/bench_smoke
CUDA_VISIBLE_DEVICES=0,1,2 /opt/intel/oneapi/mpi/latest/bin/mpirun -np 3 ../bin/cuda_3D_FM < input_smoke_3gpu.in
```

## 工作日志要求

必须维护根目录下的 `AGENT_LOG.md`。

每次执行以下任一操作后，都要追加日志：

- 修改源码或构建文件。
- 新增、修改测试样例。
- 编译程序。
- 运行 smoke、correctness 或 performance 测试。
- 记录 baseline。
- 对比新旧结果。
- 发现数值误差、性能退化、环境问题或服务器占用问题。

每条日志至少包含：

- 时间。
- 操作目标。
- 修改文件。
- 执行命令。
- 测试结果。
- 输出、哈希或误差摘要。
- 风险与下一步。

## 安全约束

- 禁止批量删除文件或目录。
- 不使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。
- 需要删除文件时，只能一次删除一个明确路径的文件。
- 不在未说明时修改全局系统配置、全局 Git 配置、全局代理或用户 shell 启动文件。
- 优先新增独立测试目录和独立构建文件，不覆盖原始输入数据。
- 后续涉及远程服务器操作时，要记录命令和结果。

## 说明

当前 `orig_code` 目录已确认与当前可运行源码关键文件哈希一致，因此不能作为“未修改原始版”证明。后续若找到真正未修改原始代码，可以补充建立 original baseline。
