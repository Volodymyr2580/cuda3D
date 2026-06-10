# CUDA3D 单卡 CUDA 优化阶段汇报

> 汇报目标：说明本阶段 CUDA3D 正演程序的性能优化结果、已尝试路线、当前瓶颈判断，以及后续多 GPU batching 的空间。

## 一句话结论

如果把前期已经完成但没有保留干净原始源码快照的 `1.8x` 提速 anchor 计入，再乘以后续正式封存的两段加速，本项目从“最初最初的实现”到当前 best 的估算总提速为：

```console
estimated WP speedup vs first original
  = early anchor * zmem over current_best_reference * current_best over zmem
  = 1.8 * 1.049300 * 1.222023
  = 2.308x

estimated Gradient speedup vs first original
  = 1.8 * 1.046865 * 1.206588
  = 2.274x
```

这就是给导师汇报时最直接的总结果：**单 GPU exact-FP32 线总体约 `2.31x` WP 提速，Gradient 约 `2.27x` 提速**。

同时需要诚实说明口径：这个 `2.31x` 是基于项目历史 anchor 的总估算，不是“同一台机器上重新编译最初源码后直接测出来”的正式 direct table。原因是仓库当前没有一个可证明、可重建的 very-first original source snapshot；已有记录显示 `orig_code` 与当前 runnable source 的关键文件 hash 已经匹配，不能再当原始基线使用。

在 RTX 5090 单 GPU、`perf_1gpu_6shots`、三轮同机同 session 复现实验中，当前 `current_best_v_pml_len16` 相比冻结的 `zmem` baseline 的正式直接测量结果为：

| 指标 | baseline 平均 | 当前 best 平均 | speedup | 时间下降 |
|---|---:|---:|---:|---:|
| WP computing time | 约 `2.4305s` | `1.988905s` | `1.222023x` | 约 `18.17%` |
| Gradient TIME all | 约 `2.5482s` | `2.111930s` | `1.206588x` | 约 `17.12%` |
| elapsed wall time | 约 `3.3733s` | `3.016667s` | `1.118261x` | 约 `10.58%` |

正确性通过：

```console
max rel L2 = 6.384336e-07
max abs    = 4.768372e-06
```

这里的 `zmem` 是中后期 CUDA 重构阶段冻结的可运行数值金标准，不是项目最初最初的版本。因此本讲义采用双口径：

1. **导师汇报总口径**：估算从最初实现到当前 best，WP `2.308x`，Gradient `2.274x`。
2. **严格复现实测口径**：当前 best 相对冻结 `zmem`，WP `1.222023x`，Gradient `1.206588x`。

这样讲的好处是：既能回答“总共快了多少”，也不会把无法直接重跑的历史版本伪装成正式 direct benchmark。

## 从最初到当前的提速链条

整个优化不是一次完成的，而是分成三段：

| 阶段 | 对比对象 | WP speedup | Gradient speedup | 证据状态 |
|---|---|---:|---:|---|
| 早期工程优化 anchor | very-first original -> early current-best | `~1.800x` | `~1.800x` | 历史项目 anchor，非当前同机 direct table |
| `zmem_reference` 阶段 | `zmem_reference` -> `current_best_reference` | `1.049300x` | `1.046865x` | 项目封存记录 |
| 当前 CUDA-core 阶段 | `current_best_v_pml_len16` -> `zmem_reference` | `1.222023x` | `1.206588x` | 三轮同机 repeat direct benchmark |
| **累计估算** | very-first original -> current best | **`2.308x`** | **`2.274x`** | 前两段 anchor × 当前 direct result |

累计 WP 计算：

```console
1.800 * 1.049300 * 1.222023 = 2.308
```

累计 Gradient 计算：

```console
1.800 * 1.046865 * 1.206588 = 2.274
```

因此，导师汇报可以这样说：

> 当前单 GPU exact-FP32 版本相对项目最初实现，按已记录的阶段性 anchor 估算，WP 计算约提升 `2.31x`；其中最后一段、可严格复现实测的 CUDA-core 重构贡献了 `1.222x`。

## 测试环境和验收口径

核心原则是先保证可重复，再谈加速。

```console
GPU             NVIDIA GeForce RTX 5090
CUDA            13.0
GPU arch        sm_120
case            perf_1gpu_6shots
rounds          3
baseline        zmem
current best    current_best_v_pml_len16
tolerance       rel L2 <= 1e-5, no NaN / Inf
```

正式 current-best flags：

```console
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DCUDA3D_PML_ZMEM_IN_P
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
-DCUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
-DCUDA3D_CPML_VMEM_DISABLE_MPI
-DCUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
-DCUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
-DCUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
```

## 对最初代码结构的认识

这个程序不是一个单 kernel 的简单 stencil，而是一个 3D acoustic / CPML 正演计算管线。每一步 wave-step 大致由三类 kernel 构成：

1. `p_core`
   - 主体内部区域的 pressure 更新。
   - 规则 stencil，计算密度相对高，但很吃 memory/L2 throughput。

2. `v_pml`
   - PML 区域的速度或一阶导相关更新。
   - 边界条件复杂，活跃线程不满，分支和访存不规则。

3. `p_pml`
   - PML 区域的 pressure 更新。
   - 本阶段最关键瓶颈；包含 CPML memory state 更新、`vx/vy/vz` 导数路径、最终 pressure writeback。

从 profiling 看，优化前主要瓶颈集中在 PML，尤其是 pressure-PML：

```console
p_pml sampled-main share       about 53%
p_core sampled-main share      about 26%
v_pml sampled-main share       about 20%
```

这意味着：如果只盯着 core stencil 或编译参数，很难获得大幅提速。真正有价值的方向是重构 PML 数据流和线程所有权。

## 我们做对了什么

### 1. CPML velocity memory double buffer

宏：

```console
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_CPML_VMEM_DISABLE_MPI
```

作用：

- 明确 CPML velocity memory 的 old/next ownership。
- 减少不必要的状态依赖和 global round trip。
- 作为后续 PML 数据流重构的 scaffold。

效果：

```console
WP speedup      about 1.032x
Gradient        about 1.028x
```

### 2. pressure-PML z-recompute shared line cache

宏：

```console
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

背景：

在原 pressure-PML 中，`vz_after_update` 的 z 方向 recompute 被重复调用。静态模型估计：

```console
current recompute calls          152,951,552
shared z-line cache estimate      29,093,740
call reduction                    about 80.98%
```

实现思想：

- 在 pressure-PML tile 内用 shared z-line cache 保存 `vz_after_update` 中间量。
- 保持原有数学路径，不重开已经失败的 z-face direct derivative / fusion 路线。

效果：

```console
directfill vs zmem
WP speedup        1.101172x
Gradient speedup  1.100029x
max rel L2        0
```

### 3. pressure-PML length-16 half-warp packing

宏：

```console
CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
```

背景：

pressure-PML tile 的 active lane 利用率很低，典型 active z-line 有 length-16、length-23、length-32。length-16 可以自然地把两条 z-line 塞进一个 warp：

```console
lane 0..15    line A
lane 16..31   line B
```

这解决的是 “warp 里一半线程闲着” 的 ownership 问题，而不是语法级微调。

效果：

```console
pressure_len16 vs zmem
WP speedup        1.194495x
Gradient speedup  1.179869x
max rel L2        6.384336e-07
```

### 4. velocity-PML length-16 half-warp packing

宏：

```console
CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
```

作用：

- 把 pressure-PML 的 active-segment 思路迁移到 v-PML。
- 收益比 pressure-PML 小，但 repeat 稳定。

最终叠加效果：

```console
current_best_v_pml_len16 vs zmem
WP speedup        1.222023x
Gradient speedup  1.206588x
elapsed speedup   1.118261x
max rel L2        6.384336e-07
```

## 主要尝试路线和结果

| 路线 | 类型 | 结果 | 决策 |
|---|---|---:|---|
| CPML velocity double buffer | ownership scaffold | WP `~1.032x` | 保留 |
| pressure z-recompute line cache | PML 数据复用 | WP `1.101x` vs zmem | 保留 |
| pressure len16 half-warp packing | warp lane ownership | WP `1.194x` vs zmem | 保留 |
| velocity len16 half-warp packing | warp lane ownership | 最终 WP `1.222x` vs zmem | 保留 |
| compact `dzz16` state | compact-state 微调 | WP `1.011842x` vs current-best | 拒绝 |
| compact `dz16 old/next` state | compact-state 微调 | WP `1.017787x` vs current-best | 拒绝 |
| residual pressure branch split | 分支特化 | ceiling `<5%` | 拒绝 |
| length-23 / exact active-point descriptor | descriptor compaction | calibrated `~1.015x` | 拒绝 |
| `p0 __ldg`, local `new_mem`, ptxas cache-policy | 语法/编译微调 | noise-level 或变慢 | 拒绝 |
| z-face VP fusion / shared-VP | kernel fusion | 正确但变慢 | 拒绝 |
| K=2 ordinary temporal blocking | temporal pipeline | 同步 / halo 模型失败 | 拒绝 |
| CUDA Graph / async stream | 调度优化 | `~1.005x` | 拒绝 |

## 为什么认为单卡 exact-FP32 优化基本到头

当前 profiling anchor：

| 区域 | duration | share | 若要再带来 5% sampled-main，需要局部提速 |
|---|---:|---:|---:|
| `p_core` | `93.730us` | `33.00%` | `1.1686x` |
| pressure-PML total | `138.120us` | `48.63%` | `1.1085x` |
| pressure len16 | `66.180us` | `23.30%` | `1.2568x` |
| pressure residual | `71.940us` | `25.33%` | `1.2315x` |
| v-PML total | `52.160us` | `18.37%` | `1.3500x` |

问题在于：剩下的耗时不再是容易消掉的冗余。

pressure-PML source profile 显示，packed pressure kernel 的热点主要是：

```console
final p0 / p1 / cw2 update      about 60.78%
CPML mem_dzz recursive update   about 26.82%
z-cache shared loads             about 1.92%
address/control visible lines    about 4.31%
```

这说明后续瓶颈主要来自数学上必须存在的状态更新：

- 二阶 pressure update 需要 old pressure、current pressure、velocity model、new pressure write。
- CPML memory 是递归状态，每步必须读旧状态、写新状态。
- 单纯换 `__ldg`、调整局部变量、改 cache policy，不能消除这些状态流量。

用 Amdahl 视角看，如果想让整体 sampled-main 再快 5%，必须满足：

```console
pressure-PML total local speedup  >= 1.1085x
pressure len16 local speedup      >= 1.2568x
pressure residual local speedup   >= 1.2315x
v-PML total local speedup         >= 1.3500x
```

而现有可实现的普通 CUDA 路线都达不到这个门槛。项目中的 frontier gate 已经给出结论：

```console
ordinary_exact_cuda_frontier_exhausted_for_micro_routes
ordinary CUDA allowed prototype count = 0
```

这不是说理论上绝对没有更快的算法，而是说：在 exact-FP32、当前数学格式、普通 CUDA kernel 模型下，继续做小修已经缺少可证明的收益上界。

## 为什么 temporal blocking / cluster 也没有立刻打开

我们也检查了更激进的 K=2 temporal blocking 思路。理想情况下，它可以减少中间 `p_mid` 的 global traffic；但普通 CUDA 没有 grid-wide barrier。

曾经的 K=2 cooperative-grid 需求：

```console
required blocks      70688
resident ceiling      2040
over capacity        34.651x
```

RTX 5090 支持 thread-block cluster，但 cluster-local DSM 模型也没有过门：

```console
best DSM local pair byte ratio       1.1602x
required local pair byte ratio      <=0.8557x
estimated sampled-main speedup       0.9498x
```

也就是说，在当前模型下，cluster-local temporal prototype 甚至可能更慢。后续除非有全新的 ownership 表示，不能直接写 cluster CUDA prototype。

## 多 GPU 还有什么空间

原程序已有 MPI 并行能力，但这和我们后续说的 multi-GPU batching 不是完全一回事。

原有 MPI 并行主要问题：

1. root rank 打印的 WP 时间容易误导，不能代表全局 wall-clock throughput。
2. 多 rank 若共享同一张 GPU，会变慢；同 GPU oversubscription 已经被拒绝。
3. 多 GPU 情况下需要严格控制 GPU affinity、输出目录隔离、每轮 rebuild flags、正确性对比和负载均衡。
4. 对多炮正演，真正关心的是 shots/sec 或 Gradient wall-clock，而不仅是单个 rank 的 kernel 时间。

对于 `perf_1gpu_6shots`，如果有多张 GPU，最自然的提升来自 multi-shot batching：

| GPU / ranks | shot 分配 | 理想 speedup | 备注 |
|---:|---|---:|---|
| 1 | `[6]` | `1.0x` | 当前单卡平台 |
| 2 | `[3, 3]` | `2.0x` | 最干净的下一步 |
| 3 | `[2, 2, 2]` | `3.0x` | 6 shots 完美均分 |
| 4 | `[2, 2, 1, 1]` | `3.0x` | 受 longest rank 限制 |
| 6 | `[1, 1, 1, 1, 1, 1]` | `6.0x` | shot 数足够时才接近 |

因此，多卡优化的空间主要不在于 “再把一个 kernel 改快 2 倍”，而在于：

- 每张 GPU 跑 current-best kernel stack。
- 每个 rank 处理独立 shot batch。
- 输出目录和随机环境完全隔离。
- 用 wall-clock / Gradient time 作为正式指标。
- 避免同 GPU oversubscription。
- 对不同 shot 数做负载均衡。

一句话：单卡 CUDA-core 的 exact-FP32 微调已经接近 frontier；多卡 batching 的收益来自并行吞吐，而不是改变单炮数值路径。

## 对导师汇报时可以强调的三点

1. 这不是只调 block size 的优化。
   - 真正有效的是 PML dataflow 和 warp ownership：z-recompute cache、len16 half-warp packing、CPML old/next ownership。

2. 正确性是硬约束。
   - 当前 best 的 max rel L2 为 `6.38e-7`，低于 `1e-5` 门槛。
   - 所有性能结论都来自输出对比通过后的 repeat benchmark。

3. 当前单卡结论是工程上诚实的。
   - 后续不是没有方向，而是普通 exact-CUDA 微调已经没有足够收益上界。
   - 下一阶段应切换到 multi-GPU batching、relaxed precision，或全新 ownership 表示；不能继续随机开 kernel。

## 可引用的核心数据

```console
estimated total WP speedup vs first original        2.308x
estimated total Gradient speedup vs first original  2.274x
total-speedup status                                estimate, not direct table
reason                                              no rebuildable very-first original snapshot

baseline             zmem
current best          current_best_v_pml_len16
case                  perf_1gpu_6shots
rounds                3
mean WP               1.988905s
mean Gradient          2.111930s
WP speedup             1.222023x
Gradient speedup       1.206588x
elapsed speedup        1.118261x
max rel L2             6.384336e-07
```

## 后续建议

如果目标是继续提升总体效率，我建议优先级是：

1. **多 GPU batching 正式实验**
   - 需要至少 2 张稳定 GPU。
   - 先做 2 GPU `[3,3]` 的 6-shot benchmark。
   - 指标用 wall-clock、Gradient time、输出 rel L2。

2. **relaxed precision 单独分支**
   - 只有在导师接受更宽精度策略后再做。
   - 必须维护 exact-FP32 与 relaxed 两条线，不能混报。

3. **新 ownership representation**
   - 先做数学等价证明和 byte / sync model。
   - 没有 `>=5%` repeat ceiling，不写 CUDA prototype。

如果只是继续 ordinary exact-FP32 single-GPU kernel 微调，当前证据不支持继续投入。
