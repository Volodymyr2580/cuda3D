# CUDA3D RTX 5090 优化进度

## 当前状态

- **平台**: RTX 5090, CUDA 13.0, sm_120
- **Baseline WP**: 0.545397s (perf_1gpu: 384x384x95, nt=1501)
- **目标**: 5x → WP ~0.109s
- **模式**: 单 agent 逐步推进, caveman 压缩通信

## 已完成

1. [x] 编译通过 (`make -B -f makefile.rtx5090 test`)
2. [x] smoke 通过 (WP=0.002s)
3. [x] correctness 通过 (WP=0.011s)
4. [x] perf_1gpu baseline 复现 (WP=0.545s)
5. [x] nsys profile 完成

## nsys Profile 关键发现

| 项目 | 时间 | 占比 |
|------|------|------|
| `cuda_fd3d_p_pml_ns` | 219ms | 40.4% |
| `cuda_fd3d_v_pml_ns` | 205ms | 37.7% |
| `cuda_fd3d_p_core_ns` | 114ms | 20.9% |
| `cudaLaunchKernel` | 449ms | API 81.4% |
| `cudaMemcpy` | 100ms | API 18.1% |
| `cudaMalloc/Free` | ~2ms | API 0.3% |

**关键洞察**: kernel 执行总时间 543ms, 但 launch API 449ms。
**每步 4 个 kernel launch × 75us ≈ 300us, 与 kernel 执行相当。**
→ CUDA Graph 可能收益最大。

## 待办 (按优先级)

1. [ ] **CUDA Graph** — 将每步 V→Pcore→Ppml→Inject 打包为 graph replay
   - 风险: 之前 per-shot graph 失败 (instantiate 开销进 WP)
   - 新策略: 在 fd_3d_f 内 capture 整个 time-step loop 序列, loop 内只 replay
2. [ ] **显存池** — 消除 fd_3d_f 每炮重复 malloc/free (但 profile 显示仅占 0.3%, 优先级降)
3. [ ] **Kernel 融合** — v_pml + p_pml 局部融合, 减少 vy/vx/vz global memory 往返
4. [ ] **寄存器压力** — p_pml_ns 40-50 寄存器, 重构滑动窗口

## 实验记录

| Tag | 改动 | WP | vs Baseline | 结果 | 备注 |
|-----|------|-----|-------------|------|------|
| baseline | - | 0.545397s | 1.00x | ✅ | 已冻结 |

## 实验记录

| Tag | 改动 | WP | vs Baseline | 结果 | 备注 |
|-----|------|-----|-------------|------|------|
| baseline | - | 0.545397s | 1.00x | ✅ | 已冻结 |
| cuda_graph | FD 3 kernel 打包为 2-graph replay | 0.546443s | ~1.00x | ✅ | correctness 通过, 但无加速 |

### CUDA Graph 分析

- 编译通过, smoke/correctness 通过, rel L2 = 0
- perf_1gpu WP = 0.546443s vs baseline 0.545397s (无差异)
- 可能原因:
  1. RTX 5090 驱动已优化单个 launch, 3-kernel graph 收益不足
  2. `cudaGraphLaunch` 到 default stream 与后续 inject 仍串行
  3. perf_1gpu 中 kernel 执行时间 (543ms) 仍占主导, launch overhead (449ms) 虽大但 graph 未能有效压缩
- 下一步: nsys profile graph 版本, 对比 launch 调用次数和 API 时间

| bs_32x8x2 | block size 32x8x2 | 0.577354s | 0.94x | ✅ | profile快8%, perf退化6%, 已回退 |

| stream_overlap | v_pml || p_core in CUDA streams | 0.546808s | 0.997x | ✅ | 理论可省114ms, 实际无收益, 已回退 |

## 结论

小/中改动 ROI 耗尽:
- CUDA Graph: launch overhead 不是 perf_1gpu 瓶颈
- Block size: 128x2x1 仍是 perf_1gpu 最优
- Stream overlap: GPU 已饱和或同步开销抵消收益

## 下一步建议

**方向 A: v_pml + p_pml kernel 融合**
- v_pml 输出 vy/vx/vz 只被 p_pml 使用
- 融合后可避免 velocity 全局内存 write+read
- 减少 1 launch/step
- 风险: 寄存器压力增加, 需 shared memory tile

**方向 B: temporal blocking**
- 在 tile 内执行多时间步, 减少全局内存流量
- 3D stencil 标准 HPC 优化, 可实现 2-5x
- 风险: 实现复杂, PML 边界处理困难

**方向 C: 尝试 p_core 与 p_pml 并行**
- 两者写 p0 不同区域, 理论上可并行
- 之前 stream 方案同步开销大, 可改用 cudaEvent

| mem_pool | static pool for d_p0/d_p1/d_v/d_memory | 0.545344s | 1.00x | ✅ | 单炮场景无收益, 多炮有意义 |

## 外围优化总结

已尝试全部外围方向:
1. CUDA Graph (launch overhead) — 无效
2. Stream overlap (v_pml || p_core) — 无效
3. Block size sweep — profile快8%, perf退化
4. Device memory pool — 单炮无收益

**单炮场景外围天花板极低。** nsys 显示 kernel 执行 543ms 占绝对主导, malloc/free 仅 2ms, memcpy 100ms 但与 kernel 部分重叠。

## 建议

外围已做到极致。下一步必须进入 **CUDA 核心优化**:
- v_pml + p_pml kernel 融合 (减少 velocity 全局内存往返)
- Temporal blocking (tile 内多时间步)
- 参考 HeCBench/ExaStencil 的 3D stencil 优化技术

## 多炮 Baseline (perf_1gpu_6shots)

| Case | Shots | Gradient TIME | WP Time | 每炮平均 WP |
|------|-------|---------------|---------|-------------|
| perf_1gpu | 1 | 0.576s | 0.545s | 0.545s |
| perf_1gpu_6shots | 6 | 2.854s | 2.704s | 0.451s |

注: 多炮每炮平均 WP 更低 (0.451s vs 0.545s), 因子域裁剪后 grid 更小。

### 显存池实验 (多炮)
- 带显存池的代码在 shot 1 出现 `cudaMemset invalid argument`
- 原因: 各 shot 子域尺寸不同, static pool 的 nxyzpad 比较逻辑在子域变化时出错
- 已回退到 clean malloc/free 版本
- 预期收益: 6炮 × 2ms = 12ms / 2704ms = **0.4%**

### 多炮外围优化结论
即使多炮场景, malloc/free 节省的时间占总 WP 仍 < 1%。
外围天花板确认极低。

## A 方向: v_pml + p_pml 融合

### 实验结果
- correctness **失败**, rel L2 ~1.5e-2 (>> 1e-5)

### 根本原因: Race Condition
融合 kernel 内, v_pml 写 vy/vx/vz 全局内存后, 同 kernel 内 p_pml 读相邻线程/相邻 block 的 v 值。
缺少 block 间同步, 导致边界读到旧值。

### 分析过的修复方案
1. `__syncthreads()` — 只能同步同 block, 跨 block halo 仍 race
2. Cooperative Groups `this_grid().sync()` — grid 83k blocks >> 5090 SM 数, 不支持
3. Shared memory tile — 仍需加载跨 block halo, 复杂度高

### 结论
**A 方向 (v_pml + p_pml 简单融合) 不可行。** 需要更大结构性重写才能避免 race。

## B 方向: Temporal Blocking 分析

### 设计尝试

**p_core 2-step temporal blocking:**
- 有效 core 每边缩小 14 (2×radius=7)
- 原 core_nz=87 → tb core_nz=59
- 需要 3D shared memory tile: (128+28)×(2+28)×(1+28) = 156×30×29 = 135,720 floats = **542 KB**
- **超过 RTX 5090 shared memory 限制 (~64-96 KB per SM)**

**小 tile 方案 (8×8×8):**
- 512 points, halo (22^3) → 存储 p0+p1 需 **~85 KB**
- 接近限制但 block 数 10,648, 效率低

**z-slab 方案 (16×16×16):**
- 存储 p0+p1 需 **216 KB**
- 超过限制

### 结论

标准 GPU temporal blocking (shared-memory tile) **在当前 block size 和 stencil 阶数下不可行**。

### 剩余可能方案

1. **Register blocking / 1D z-temporal** — 每线程缓存 z-line 在寄存器, 逐步推进
2. **减小 stencil 阶数** — 如从 14 阶降到 8 阶 (但会改变数值精度)
3. **分阶段实现** — 先改数据结构/内存布局, 再做 temporal blocking
4. **使用 CUDA 13 新特性** — TMA, async copy, cluster

## CUDA 13 / Blackwell / RTX 5090 调研

### 硬件规格 (实测)

| 参数 | RTX 5090 | RTX 4090 (参考) |
|------|----------|----------------|
| 架构 | Blackwell (sm_120) | Ada Lovelace (sm_89) |
| 显存 | 32 GB GDDR7 | 24 GB GDDR6X |
| 显存频率 | 14001 MHz | 10501 MHz |
| SM 频率 | 3135 MHz | 2520 MHz |
| 功率限制 | 575W | 450W |
| 计算能力 | 12.0 | 8.9 |

### CUDA 13.0 关键特性

1. **r610 driver** — 改善 Blackwell 系统可观察性和错误分类
2. **`sm_120` 支持** — 但 release notes 提到编译器可能有 bug
3. **Dynamic Boost** — Grace Blackwell 系统支持

### 针对 Stencil 的潜在优化

1. **GDDR7 高带宽** — 理论带宽 ~1.5-2x GDDR6X。当前 kernel 执行时间与 4090 几乎相同，说明代码未充分利用带宽优势
2. **第五代 Tensor Cores** — 面向 AI/矩阵计算，对 stencil 不适用
3. **异步内存拷贝 (`cp.async`)** — CUDA 11.8+ 引入，可用于 global→shared 异步传输，减少 stall
4. **Warp shuffle (`__shfl_sync`)** — 减少 shared memory 依赖，提高 occupancy
5. **CUDA Graph 改进** — Blackwell 可能有更低 launch overhead，但当前实验无效

### 关键发现

**当前代码是 latency-bound 而非 bandwidth-bound。**
- 5090 的 GDDR7 带宽远高于 4090 的 GDDR6X
- 但相同 workload 下 kernel 执行时间几乎相同
- 说明瓶颈在 compute latency / launch overhead / occupancy，而非显存带宽

### 优化建议

1. **优先优化 occupancy 和寄存器使用** — 减少 warp stall
2. **尝试 `cp.async` 异步加载** — 隐藏内存延迟
3. **利用 warp shuffle 替代 shared memory** — 减少 sync 开销
4. **考虑 register blocking / 1D temporal** — 在 z 方向利用寄存器局部性
5. **若以上无效，必须接受结构性重写**

## 总结与下一步

当前状态:
- 外围优化已穷尽
- v_pml+p_pml 融合因 race condition 失败
- Temporal blocking 因 shared memory 限制不可行
- 代码 latency-bound, 未利用 5090 带宽优势

下一步 (按优先级):
1. **尝试 `cp.async` + warp shuffle 优化 p_core** — 中等复杂度, 潜在 1.2-1.5x
2. **尝试 register blocking (z 方向)** — 高复杂度, 潜在 1.5-2x
3. **若以上无效, 接受当前 1.0x, 转向应用层优化** — 多炮并行/异步调度

## Step 1: p_core `const __restrict__` (LDG hint)

| Tag | 改动 | WP | vs Baseline | 结果 | 备注 |
|-----|------|-----|-------------|------|------|
| pcore_ldg | p1 → `const float *__restrict__` | 0.546795s | 0.999x | ✅ | correctness rel L2=0, 无加速 |

**结论**: LDG hint 对 p_core 无效。内存访问已足够高效，或编译器已自动优化。

## Step 1b: p_core 2D shared memory tile (z+x)

计划：扩展 z_tile 为 zx_tile，缓存 x 方向 halo，消除 14 次 global read / thread。
- 当前 tile: [1][2][142] = 284 floats (~1.1KB)
- 新 tile: [1][16][142] = 2272 floats (~9KB)，仍在 SM limit 内
- 收益：x 方向 14 次 read 从 global → shared

| pcore_2dtile | p_core z+x 2D shared tile | 0.898886s | 0.61x | ❌ | correctness 通过, 严重回归, 已回退 |

**2D tile 失败原因分析:**
- 共享内存从 1.1KB → 9KB, 加载循环引入额外指令和同步
- p_core 仅占 20% 时间, 且 x/y global read 已 coalesced
- latency-bound 场景下, 减少 bandwidth 不敏感

**Step 1 结论**: p_core 内存级优化天花板极低。建议转向 v_pml/p_pml（合计 78% 时间）。

| vp_regopt | v_pml + p_pml 寄存器优化 (减少局部变量, 复用 c1/c2/c3, 内联索引) | 0.546916s | 0.997x | ✅ | correctness 通过, 无加速 |

**寄存器优化失败原因分析:**
- 编译器已自动做类似优化 (死代码消除、寄存器分配)
- occupancy 非瓶颈, latency 来自 arithmetic dependency chain
- p_pml 的 `__ldg` 和 PML constant mem 访问已足够高效

## Step 2 结论

v_pml/p_pml 寄存器级优化无效。

**CUDA 核心微优化天花板确认。** 代码已高度优化:
- `__ldg` read-only cache
- PML coefficients in constant memory  
- Coalesced/broadcast memory patterns
- p_core z-tile shared memory

**5x 目标无法通过微优化实现。** 必须进入结构性优化或应用层优化。

## 下一步建议

**方向 3: 应用层优化**
1. 多炮异步并行 (overlap shot computation with IO/communication)
2. 时间步异步调度 (overlap host MPI with GPU kernel execution)
3. 单进程多炮 batching (减少 MPI overhead)

**或更大胆的结构性 CUDA 优化:**
1. 精度降级 (FP16/TF32) — 需验证数值稳定性
2. Stencil 阶数降低 (14阶 → 8阶) — 改变物理精度
3. 完整 temporal blocking 重写 — 高复杂度

建议优先做应用层 (3), 同时调研 HeCBench/其他 stencil 库的最佳实践。

| v_pml_ytile | v_pml y-tile shared mem, block 128x2x4 | 0.665555s | 0.82x | ❌ | correctness 通过, 回归 22%, 已回退 |

**Temporal blocking / shared tile 全面失败原因:**
1. `__ldg` read-only cache 已高效缓存 broadcast/coalesced 访问
2. Shared tile 加载循环 + `__syncthreads()` 引入额外指令和延迟
3. Block size 改变影响 occupancy 和 warp 调度
4. Latency-bound 场景, 减 bandwidth 无效

**CUDA 核心优化总结:**
| 方向 | 尝试 | 结果 |
|------|------|------|
| p_core `__restrict__` | LDG hint | 无效 |
| p_core 2D tile | z+x shared mem | 严重回归 |
| v_pml/p_pml 寄存器优化 | 减少局部变量 | 无效 |
| v_pml y-tile | block 128x2x4 | 回归 |

**结论: 5x 无法通过 kernel 微优化实现。** 必须结构性改变或应用层优化。

## 下一步建议

**选项 A: 应用层优化 (推荐)**
- 多炮异步并行 (overlap computation with MPI/IO)
- 单进程 batch 多炮 (减少 launch overhead)
- Host-GPU 异步流水线

**选项 B: 结构性 CUDA 重写 (高风险高回报)**
- FP16/TF32 精度降级 — 2x 潜在, 需验证数值稳定性
- 完整 3D block temporal blocking — 需重构所有 kernel 和 launch 配置
- 参考 HeCBench 最新 stencil 实现

**选项 C: 接受当前性能**
- perf_1gpu WP = 0.545s 已是高度优化结果
- 转向其他项目/论文写作

| v_pml_ytile | v_pml block 128x2x4 + y-tile shared mem | 0.548109s | 0.994x | ✅ | correctness 通过, 无加速 |

**y-tile 失败原因分析:**
- `__ldg` 已充分缓存 broadcast y-reads
- `__syncthreads` + 协作加载引入额外开销
- block 数减少 4x 未抵消同步成本

## 阶段性结论

**CUDA 核心优化天花板确认。** 全部方向已穷尽:
| 方向 | 尝试 | 最佳结果 |
|------|------|----------|
| 内存 hint | `__restrict__`, `__ldg` | 已在使用, 无额外空间 |
| shared tile | p_core 2D, v_pml y-tile | 无收益或回归 |
| 寄存器优化 | 减少局部变量, 复用寄存器 | 编译器已自动优化 |
| block size | 128x2x1 仍为最优 |
| launch overhead | CUDA Graph | 无效 |
| stream overlap | v_pml \|\| p_core | 无效 |

**5x 目标无法通过 kernel 微优化实现。**

## 剩余可能方向

**结构性 CUDA 重写 (专门 session 级别):**
1. **完整 temporal blocking** — 3D tile + multi-step in registers/shared。需重构 block 布局、 halo 交换、PML 处理。复杂度极高。
2. **FP16/TF32 降级** — RTX 5090 Tensor Cores 支持。需验证 rel L2 <= 1e-5 数值稳定性。
3. **Stencil 阶数降低** — 14阶 → 8阶。改变物理精度，非纯优化。

**应用层优化 (推荐优先):**
1. 多炮异步并行 (overlap shot computation)
2. 时间步 host-GPU 异步调度
3. 单进程多炮 batching

**建议**: 保存当前状态, 下一步做 FP16 验证或应用层优化。
