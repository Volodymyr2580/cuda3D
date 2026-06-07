# Kimi CUDA Rewrite Directive

本文档用于约束后续 Claude Code + Kimi-K2.6 在 CUDA3D 项目中的工作方式。

Kimi 在本项目中的角色是：高级 CUDA 实现工程师。

Kimi 可以写代码、编译、运行测试、做 profiling、整理实验报告，但不能单独决定项目的总体优化路线。涉及结构性方向选择、数值模型改变、精度降级、stencil 阶数改变、baseline 口径改变时，必须先写反馈报告，交给主架构 agent 和用户审阅。

## 当前远程记录审计

审计时间：2026-06-06 CST。

审计目录：

```text
/work/wenzhe/cuda3D
```

已发现的 Kimi/Claude 工作记录：

- `CLAUDE.md`
- `progress.md`
- `AGENT_LOG.md`
- `benchmarks/runs/`
- `benchmarks/reports/`
- `benchmarks/baselines/current_runnable/`

RTX 5090 单卡 baseline 已建立：

```text
case: perf_1gpu
grid: 384 x 384 x 95
nt: 1501
shots: 1
baseline run: benchmarks/baselines/current_runnable/perf_1gpu_rtx5090_baseline_20260606_002902
WP computing time = 0.545397 s
Gradient TIME all = 0.576524 s
```

Kimi 已记录的主要实验：

| 方向 | 代表 run | WP | 结论 |
|---|---:|---:|---|
| CUDA Graph | `perf_1gpu_cuda_graph_20260606_110054` | 0.546443s | correctness 通过，无加速 |
| block size `32x8x2` | `perf_1gpu_bs_32x8x2_20260606_110528` | 0.577354s | 退化 |
| stream overlap | `perf_1gpu_stream_overlap_20260606_111119` | 0.546808s | correctness 通过，无加速 |
| device memory pool | `perf_1gpu_mem_pool_20260606_120507` | 0.545344s | 单炮无实际收益 |
| `v_pml + p_pml` 简单融合 | `correctness_vp_fused_20260606_122901` | correctness case only | correctness 失败，存在跨 block race 风险 |
| `p_core` LDG/restrict hint | `perf_1gpu_pcore_ldg_20260606_132729` | 0.546795s | 无加速 |
| `p_core` 2D shared tile | `perf_1gpu_pcore_2dtile_20260606_133119` | 0.898886s | 严重退化 |
| `v_pml/p_pml` 寄存器整理 | `perf_1gpu_vp_regopt_20260606_134405` | 0.546916s | 无加速 |
| `v_pml` y-tile shared memory | `perf_1gpu_v_pml_ytile_20260606_135551` | 0.665555s | 退化 |
| `v_pml` y-tile 复测 | `perf_1gpu_v_pml_ytile_20260606_150215` | 0.548109s | 接近 baseline，无实质收益 |
| clean revert | `perf_1gpu_clean_revert_20260606_135908` | 0.546114s | 已回到 baseline 附近 |

审计判断：

- Kimi 的记录支持“小型外围优化和局部 kernel 微调收益很低”。
- 这些记录不能证明“CUDA 核心优化已死”。
- 更准确的判断是：当前代码已经吃掉了显而易见的低风险优化，下一阶段必须从数据依赖、数据布局、PML 状态表示和 time-stepping 结构上重写。
- 后续不得继续随机尝试 `__restrict__`、cache flag、普通 shared tile、block size 微调等低信息量实验，除非先给出新的机制解释和验证指标。

## 工程角色边界

Kimi 必须遵守：

1. Kimi 是实现工程师，不是最终架构决策者。
2. Kimi 可以提出建议，但必须把建议写成可审阅报告。
3. Kimi 不得用一句“方向无效”结束任务。
4. 每个实验必须有假设、代码改动、命令、正确性结果、性能结果、是否保留、是否回退。
5. 如果实验失败，必须解释失败机制，而不是只写“无效”。
6. 不得把服务器密码写入任何项目文件、脚本、日志或报告。
7. 不得批量删除文件或目录。

需要主架构 agent 或用户确认的情况：

- FP16、TF32、mixed precision 等精度降级。
- stencil 阶数改变。
- 物理模型、输入输出格式、baseline 口径改变。
- 删除文件、覆盖 baseline、覆盖 speedup archive。
- 无法恢复的全局环境修改。

## 下一阶段技术路线

下一阶段关注 CUDA 底层计算核心的结构化重写，不优先做 MPI 多炮调度或应用层并行。

### 0. 先写依赖和流量分析

在动代码前，Kimi 必须先写：

```text
docs/cuda_core_dependency_map.md
```

内容必须包括：

- `fd_3d_f` 每个 time step 的 kernel 顺序。
- 每个 kernel 读写哪些数组。
- `p0/p1/vx/vy/vz/memory_*` 的生产者和消费者。
- `v_pml` 到 `p_pml` 的跨 block 依赖。
- 每个网格点大致 global load/store 次数。
- 哪些数组是 full-domain，哪些只在 PML 区域真正有意义。

没有这份依赖图，不允许继续结构性重写。

### 1. 建立新旧并存的 opt path

禁止直接覆盖当前 active solver。

建议新增并行路径，例如：

```text
fd_3d_f_opt
cuda_fd3d_*_opt
```

旧路径继续作为数值金标准。新路径通过 runtime flag、compile flag 或独立入口启用。

### 2. Core 与 PML 分开重写

不要把 core 区和 PML 区混在一个“万能优化”里处理。

建议分成两条实验线：

```text
core_opt: 只处理非 PML core stencil
pml_opt: 只处理 PML 状态和边界更新
```

core 区适合尝试：

- register blocking。
- z-line sliding window。
- warp shuffle。
- 更小范围的 temporal blocking prototype。

PML 区适合尝试：

- PML 状态数组压缩为边界专用 layout。
- 避免 full-domain velocity/PML 辅助场无意义读写。
- 重排 memory variables，使 PML 访问更连续。
- 在不引入跨 block race 的前提下减少 `v_pml -> p_pml` global round trip。

### 3. 禁止重复已失败的简单融合

`v_pml + p_pml` 简单融合已暴露跨 block race：

```text
v_pml 写 vx/vy/vz 后，p_pml 需要读邻近 block 的 velocity。
同一个 kernel 内没有全 grid 同步，所以会读到旧值。
```

因此后续如果再做融合，必须先解决依赖问题。可接受的前置设计包括：

- 只融合不依赖跨 block halo 的局部子问题。
- 重构 domain decomposition，使 tile halo 在同一 block 或 cluster 内可同步。
- 使用多阶段 kernel，显式保留同步边界。
- 改写数据布局，让 PML 更新不需要读刚写出的邻 block velocity。

否则不得再次提交“简单融合”版本。

### 4. Temporal blocking 只能先做 prototype

完整 3D temporal blocking 风险很高。Kimi 不得直接全域替换。

最低要求：

1. 先在 core-only 小区域 prototype。
2. 明确 halo 扩张公式。
3. 明确 shared memory、register、occupancy 预算。
4. smoke 通过后再做 correctness。
5. correctness 通过后再做 perf。

如果 shared memory 预算超过硬件限制，必须写出计算过程，并转向 register/warp-level blocking，而不是继续硬凑 shared tile。

### 5. FP16/TF32 不是默认路线

FP16/TF32 涉及数值精度和物理可信度，不能作为默认优化路线。

只有在以下条件满足时才能实验：

- 用户或主架构 agent 明确批准。
- 保留 FP32 baseline。
- 输出完整误差报告。
- 分别报告 `correctness` 和 `perf_1gpu`。
- 若 rel L2 超过 `1e-5`，必须撤回。

## 每次实验的反馈报告格式

Kimi 每完成一次实验，必须新增一份反馈报告：

```text
feedback/kimi_report_YYYYMMDD_HHMMSS_<tag>.md
```

报告必须使用以下结构：

```markdown
# Kimi Feedback Report: <tag>

## Hypothesis

本实验试图减少什么成本：global memory traffic、kernel launch、register pressure、branch divergence、latency chain，还是其他。

## Code Changes

- 修改文件：
- 新增文件：
- 是否保留旧路径：
- 是否可一键回退：

## Commands

列出实际执行的编译、smoke、correctness、perf、profile 命令。
不得写入密码。

## Correctness

- baseline：
- candidate：
- 输出文件数量：
- rel L2：
- max abs：
- max rel：
- NaN/Inf：
- pass/fail：

## Performance

| metric | baseline | candidate | speedup |
|---|---:|---:|---:|
| WP computing time | | | |
| Gradient TIME all | | | |
| wall-clock elapsed | | | |

## Profiling

- profiler：
- top kernels before：
- top kernels after：
- API overhead before/after：
- 关键变化：

## Decision

选择一个：

- keep
- revert
- keep for later, inactive

并解释理由。

## Failure Mechanism

如果失败，必须解释失败机制。
例如：跨 block race、shared memory 同步开销、occupancy 下降、寄存器溢出、访存不连续、launch 数增加。

## Next Proposed Step

下一步建议，但不得直接替代主架构决策。
```

同时必须追加 `AGENT_LOG.md`，报告路径也要写入日志。

## 给 Kimi 的工作指令

你接下来不要继续做随机微优化。

你的第一项任务是：

```text
写 docs/cuda_core_dependency_map.md
```

你的第二项任务是：

```text
提出 core_opt 和 pml_opt 两条重写路径的最小 prototype 设计。
```

你的第三项任务才是写代码。

每个代码实验结束后，必须写 `feedback/kimi_report_*.md`。

你不是来证明“做不到”的。你是来把每一种可能的结构化重写路线变成可验证、可回退、可审阅的工程实验。
