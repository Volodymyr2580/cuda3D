# 给 Pro 的反馈：Pressure Triple-Buffer Pipeline 设计/审计阶段结果

时间：2026-06-07 18:17 +0800

## 1. 当前状态

已按 Pro 的选择 A 执行：

```text
停止 CTA-local core two-step route
进入 CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE design/audit 阶段
不直接大规模修改 CUDA 源码
```

当前分支：

```text
exp/pressure-triple-buffer-pipeline
```

当前提交：

```text
11c46a4 docs(pressure): design triple-buffer pipeline
```

本阶段只做了文档、审计和内存估算，没有修改 CUDA kernel 行为。

## 2. 已完成产物

新增/更新文件：

```text
docs/architecture_decision_log.md
docs/pressure_pointer_swap_audit.md
docs/pressure_triple_buffer_pipeline_design.md
docs/pressure_triple_buffer_memory.md
docs/post_triple_buffer_temporal_plan.md
tools/pressure_buffer_memory_estimate.py
AGENTS.md
AGENT_LOG.md
```

其中：

```text
architecture_decision_log:
    记录 zmem accepted、PML z-slab stopped、p_core z-pencil stopped、
    core two-step correctness pass、CTA-local fused commit stopped、
    triple-buffer 新方向。

pressure_pointer_swap_audit:
    审计 d_p0/d_p1 allocation、swap、p_core、p_pml、v_pml、
    source injection、receiver extraction、ZMEM_IN_P swap、debug dump。

pressure_triple_buffer_pipeline_design:
    给出 p_prev / p_curr / p_next 时间步顺序、kernel signature、
    source/receiver 语义、debug fill、ZMEM_IN_P 相对时序和验收 gate。

pressure_triple_buffer_memory:
    估算主要 benchmark case 的 2-buffer vs 3-buffer pressure memory。

post_triple_buffer_temporal_plan:
    记录 triple-buffer 之后如何重新考虑 temporal blocking。
```

## 3. 最关键审计结论

结论：

```text
triple buffer 不能只靠 host-side pointer remap。
```

原因是当前压力 kernel 明确把 `p0` 同时当成：

```text
old p(t-1) input
new p(t+1) output
```

代码证据：

```cpp
p0[base] = 2.0f * center - p0[base] + cw2[base] * dt * lap;
```

和 PML pressure 路径：

```cpp
p0[outIndex] = 2 * p1[outIndex] - p0[outIndex] + ...
```

因此第一版 triple-buffer implementation 必须显式拆分 kernel 参数：

```cpp
cuda_fd3d_p_core_triple_ns(
    float *p_next,
    const float *p_curr,
    const float *p_prev,
    ...
)
```

PML pressure kernel 也同理：

```cpp
cuda_fd3d_p_pml_triple_ns(
    float *p_next,
    const float *p_curr,
    const float *p_prev,
    ...
)
```

如果只在 launch site 把 `p0` 改名成 `p_next`，但 kernel 内仍然读写同一个 pointer，就没有真正解决 old/new alias 问题。

## 4. 当前设计的 timestep 语义

Triple-buffer 目标：

```text
p_prev = p(t-1), read-only
p_curr = p(t), read-only
p_next = p(t+1), write target
```

每步顺序：

```text
1. v_pml reads p_curr
2. p_core writes p_next from p_curr and p_prev
3. p_pml writes p_next from p_curr, p_prev, velocity/memory
4. ZMEM_IN_P swaps d_memory_dz / d_memory_dz_next in the same relative position
5. source injection writes p_next
6. receiver extraction reads p_next
7. rotate:
       tmp    = p_prev
       p_prev = p_curr
       p_curr = p_next
       p_next = tmp
```

关键保持不变：

```text
pressure update -> source injection -> receiver extraction -> pressure rotation
```

Receiver 不能读 `p_curr`，必须读 post-injection `p_next`。

## 5. PML / ZMEM_IN_P 审计结论

`CUDA3D_PML_ZMEM_IN_P` 仍然使用自己的 double buffer：

```text
d_memory_dz
d_memory_dz_next
```

Triple-buffer 不应该改变这个逻辑。建议保持现有相对顺序：

```text
p_pml writes pressure and d_memory_dz_next
optional ZMEM coverage check
swap d_memory_dz / d_memory_dz_next
source/receiver on p_next
pressure buffer rotation
```

也就是说：

```text
ZMEM_IN_P swap 仍然发生在 source/receiver 之前；
pressure buffer rotation 仍然发生在 source/receiver 之后。
```

## 6. 内存估算结果

本地和 RTX 5090 服务器输出一致。

估算脚本：

```text
tools/pressure_buffer_memory_estimate.py
```

结论：

```text
额外 pressure buffer 本身不是 RTX 5090 显存瓶颈。
```

主要 case：

```text
correctness:
    triple pressure buffers = 0.0142 GiB
    additional pressure buffer = 0.0047 GiB

perf_1gpu:
    triple pressure buffers = 0.2456 GiB
    additional pressure buffer = 0.0819 GiB

perf_1gpu_6shots:
    triple pressure buffers = 0.2456 GiB
    additional pressure buffer = 0.0819 GiB

profile_1gpu:
    triple pressure buffers = 0.2456 GiB
    additional pressure buffer = 0.0819 GiB
```

注意：

```text
这个估算只覆盖 pressure buffers，
不包含 velocity fields、PML memory arrays、receiver output、velocity model 或 runtime overhead。
```

但至少说明：增加第三个 pressure buffer 的显存成本很小，不是阻止该路线的主要因素。

## 7. 建议的最小实现范围

如果 Pro 同意进入 implementation，我建议只做最小宏控版本：

```text
CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE
CUDA3D_PRESSURE_TRIPLE_BUFFER_DEBUG
CUDA3D_PRESSURE_TRIPLE_BUFFER_DISABLE_MPI
CUDA3D_PRESSURE_TRIPLE_BUFFER_DEBUG_FILL
```

第一版范围：

```text
1. single GPU only
2. 不改数学
3. 不做 temporal blocking
4. 只拆 pressure buffer ownership / kernel signatures
5. 默认关闭
```

必须覆盖：

```text
p_core triple variant
p_pml generic triple variant
p_pml tile-list triple variant
ZMEM_IN_P path
source injection on p_next
receiver extraction from p_next
pressure triple rotation
debug dump compatibility mapping
```

如果 `CUDA3D_PML_ZFACE_P_SPECIALIZE` 当前不在稳定 baseline 中启用，可以先在 triple-buffer 宏下禁止或不实现 zface triple variant，避免扩大第一版范围。

## 8. Debug / Correctness gate

第一版实现后建议按顺序跑：

```text
1. smoke_1gpu
2. debug dump step 0/1/2 vs zmem_reference
3. correctness
4. perf_1gpu
5. perf_1gpu_6shots
6. perf_1gpu_6shots repeat
```

Correctness gate：

```text
rel_l2 <= 1e-5
finite pass
output file count/size identical
debug dump pressure mapping correct
```

Performance gate：

```text
perf_1gpu_6shots repeat slowdown <= 2%:
    accept as disabled dataflow-clean candidate

slowdown 2%~5%:
    keep branch, diagnose overhead

slowdown >5%:
    stop and report memory/dataflow overhead
```

## 9. 主要风险

### Risk 1: p_next 未覆盖区域

如果 `p_core` 和 `p_pml` 没有覆盖所有后续会被读取/插值/输出的 cells，那么 rotated old buffer 作为 `p_next` 时可能携带陈旧值。

建议第一版加：

```text
CUDA3D_PRESSURE_TRIPLE_BUFFER_DEBUG_FILL
```

每步 pressure update 前填充 `p_next`，用于暴露未写区域。

### Risk 2: debug dump 名称语义变化

旧 dump 名称：

```text
p0
p1
```

triple-buffer 语义：

```text
p_next
p_curr
p_prev
```

兼容映射应是：

```text
baseline p0 at dump point <-> triple p_next
baseline p1 at dump point <-> triple p_curr
```

### Risk 3: PML pressure variants

稳定 baseline 使用：

```text
CUDA3D_PML_TILE_LIST
CUDA3D_PML_ZMEM_IN_P
```

因此 first implementation 必须覆盖 tile-list PML pressure path。不要只实现 generic `p_pml` 后就宣称通过。

## 10. 给 Pro 的决策请求

请 Pro 确认是否进入最小实现阶段：

```text
A. 同意进入 CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE minimal implementation。
B. 只保留 design branch，暂不实现，先要求进一步审计 debug dump/compare 工具。
C. 放弃 triple-buffer，转向其他结构路线。
```

Codex 建议选择 A，但保持严格边界：

```text
只做 macro-gated correctness-first implementation；
不在同一轮做 temporal blocking；
不声明 speedup；
以 perf_1gpu_6shots repeat slowdown <=2% 作为是否保留的性能 gate。
```

一句话总结：

> 当前 design/audit 已确认 triple-buffer 路线在显存上可行，且是解决 in-place `p0` old/new alias 的正确系统级方向。下一步若 Pro 同意，应实现最小宏控版本，显式拆 `p_next/p_curr/p_prev` kernel signature，先证明 correctness 和低 overhead，再考虑后续 temporal blocking。
