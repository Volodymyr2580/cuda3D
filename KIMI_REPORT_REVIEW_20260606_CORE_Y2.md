# Review of Kimi Feedback Report: core_y2

审阅对象：

```text
/work/wenzhe/cuda3D/feedback/kimi_report_20260606_172800_core_y2.md
```

审阅时间：2026-06-06 CST。

## 总体判断

`revert` 决策是正确的。

但这次实验暴露出三个问题：

1. 选题优先级不对：`p_core_ns` 只占约 20.9% GPU kernel 时间，不应该作为当前主攻方向。
2. 失败归因不够干净：报告把 missing-y、halved grid、sequential y processing 混在一起，没有定位到最小错误机制。
3. 调试粒度太粗：直接用完整 correctness 波形判断新 kernel，错误经过多 time step 放大后难以定位。

这不是“CUDA 核心结构化重写不可行”的证据，只说明 `core_y2` 这个局部方案不值得继续。

## 对 core_y2 假设的技术评价

Kimi 的原始假设：

```text
每个线程连续处理两个 y 点，可以复用 y 方向邻居，减少 global y-load。
```

这个假设不够稳。

当前 `p_core_ns` 的线程映射是：

```cpp
gtid1 = z fast axis
gtid2 = x axis
gtid3 = y slow axis
```

warp 内相邻线程主要沿 z 方向连续，因此：

- `p1[base ± k*stride2]` 的 x-neighbor 访问对 warp 来说仍是连续地址。
- `p1[base ± k*stride3]` 的 y-neighbor 访问对 warp 来说也是连续地址，只是换到了另一个 y plane。
- 这些访问不是“同一个值被很多线程重复读”，而是一段连续 z-line 被 warp 协同读取。

也就是说，y 方向 stride 很大，但 warp 访问形态仍然 coalesced。L1/L2 很可能已经捕捉了相当多的局部性。让一个线程算两个 y 点会增加寄存器、指令和控制复杂度，未必能换来真实带宽收益。

更重要的是：`p_core_ns` 当前已经不是最大热点。即使把 `p_core_ns` 提速 2x，总体最多也只有约：

```text
1 / (1 - 0.209 + 0.209 / 2) = 1.116x
```

如果无限快，理论上限也只有：

```text
1 / (1 - 0.209) = 1.264x
```

所以它不可能支撑 2x、3x，更不可能支撑 5x。

## 对失败机制的批评

报告中这段判断前后矛盾：

```text
This mapping does cover all core y-values.
...
The most likely explanation is missing-y-value bug.
```

如果 kernel 真的同时正确计算 `y0` 和 `y1`，那么 halved grid 覆盖是成立的。

如果 minimal version 只计算 `y0`，那么 odd y 未更新是必然结果，不能用它证明“halved grid 本身有神秘问题”。它只能证明：只改 grid 而不计算 `y1` 会漏点。

更可能的具体 bug 是：

1. `p_core_ns` 的 shared z-tile 只为当前 `gtid3` 的 y plane 加载。
2. 如果同一个线程继续计算 `y1 = y0 + 1`，它不能复用同一个 `z_tile` 作为 y1 的 z-stencil。
3. 若 y1 的 `center` 或 z-neighbor 仍来自 y0 的 shared tile，就会产生系统性错误。
4. 若 y1 改用 global load，又会失去大部分设计收益。

因此，`core_y2` 的失败不应该被总结成“grid halving 导致未知缺陷”，而应该被总结成：

```text
原 p_core 的 z-shared tile 与 one-thread-two-y 的计算模型不兼容。
要正确计算 y1，必须为 y1 重新加载 z-stencil 或扩展 shared tile 到 y 维。
扩展 shared tile 又会回到已失败的 2D/3D shared tile 成本问题。
```

## 调试流程问题

这次调试直接使用完整 `correctness` case，rel L2 约 `1e-1`。

这能判断 fail，但不能定位 bug。

以后修改单个 kernel 时，必须先建立更小的定位流程：

1. 单 time-step debug case。
2. 只运行到目标 kernel 后 dump 或 compare `p0`。
3. 对 core 区每个点检查是否 exactly once update。
4. 对比第一个错误点的 `(z, x, y)`、baseline 值、candidate 值。
5. 再进入完整 correctness。

否则错误会在多步波场传播中扩散，后续分析基本靠猜。

## 对 Kimi 后续工作的约束意见

Kimi 不应继续做 `p_core` 微优化，除非满足以下条件：

- 有 profiler 证明 `p_core_ns` 重新成为主要热点。
- 有单 kernel debug harness。
- 有明确的 memory traffic 或 instruction count 下降模型。
- 有小样本逐点对比，不只是完整 correctness 失败/通过。

当前主攻方向应转回 PML：

```text
v_pml_ns + p_pml_ns ≈ 78% GPU kernel time
```

更合理的下一步不是再调 block size 或 shared tile，而是：

1. 重新审视 PML 状态数据布局。
2. 把 PML face / edge / corner 的数据访问从 full-domain 思维改为 boundary-domain 思维。
3. 避免 compact shell 的整数除法/取模映射。
4. 避免简单拆成过多 kernel launch。
5. 在实现前先写一个 PML indexing contract，明确每个 face 的连续内存布局和邻居依赖。

## 给 Kimi 的下一条任务建议

不要继续 `core_y2`。

下一份报告应先写设计，不要先写代码：

```text
feedback/kimi_report_<time>_pml_layout_design.md
```

必须回答：

- 当前 PML shell 中哪些 vx/vy/vz 元素真的会被 `p_pml_ns` 读取？
- 这些元素是否能用 face-major contiguous buffer 表示？
- face / edge / corner 如何避免重复写或漏写？
- 新 layout 是否会引入更多 integer mapping？
- 新 layout 会增加几个 kernel launch？
- 每个 time step 预计减少多少 global load/store？
- 如何建立 one-step PML debug comparison？

通过这份设计审阅后，才能允许实现 `pml_opt` prototype。

## 结论

`core_y2` 应该撤回。

但是这次失败不应该被解释为“结构化重写失败”。它只是说明：

- `p_core` 不是主战场。
- one-thread-two-y 与现有 z-shared tile 不匹配。
- Kimi 当前还需要更严格的 kernel-level debug 和设计审查。

后续应把 Kimi 限定为实现工程师，让它先提交 PML layout design 和 one-step debug plan，再动代码。
