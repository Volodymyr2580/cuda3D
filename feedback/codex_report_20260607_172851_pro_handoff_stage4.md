# 给 Pro 的反馈：Core Two-Step 路线阶段结论与下一步请求

时间：2026-06-07 17:28 +0800

## 1. 当前结论

Codex 已按你的建议完成了 `CUDA3D_CORE_2STEP_FUSED_INTERIOR` 的 debug 正确性验证，并进一步执行了 Stage-4 Architecture Gate。

最终结论是：

```text
CUDA3D_CORE_2STEP_FUSED_COMMIT_V2 暂停，不进入 kernel 实现。
```

原因不是 correctness 失败，而是 architecture gate 失败：

```text
A/D first-implementation tile plans 的实际 commit ratio 只有约 3.2%，远低于 10% gate。
```

因此继续写 `retiled residual p_core` 和 `fused commit V2` 大概率投入很高、收益很低，不符合你设定的 stop condition。

## 2. 已完成工作

当前分支：

```text
exp/core-2step-interior-prototype
```

最新提交：

```text
0a0d333 docs(core2step): gate fused commit tile budget
```

已完成：

```text
1. meaningful 1GPU case
2. debug-only fused p2 predictor
3. p2-shift correctness
4. receiver output correctness
5. stage-4 M/O/C tile budget gate
```

关键文档：

```text
docs/core_2step_fused_design.md
docs/core_2step_fused_result.md
docs/core_2step_stage4_tile_budget.md
feedback/codex_report_20260607_171200_stage4_tile_budget.md
```

## 3. Debug Fused Correctness 结果

Meaningful case：

```text
benchmarks/cases/core_2step_meaningful_1gpu
ny=160 nx=160 nz=96 nt=501 npml=12
core_points=2033152
fused_eligible_points=922560
eligible_ratio=0.453758
source_in_fused_region=no
receivers_in_fused_region=0
```

Debug fused p2-shift 结果：

```text
pass=True
count=922560
rel_l2=0.0
max_abs=0.0
max_rel=0.0
rms=0.0
```

Receiver output vs `zmem_reference`：

```text
pass=True
files=1
rel_l2=0.0
max_abs=0.0
```

这说明：

```text
在 strict core interior、source/receiver 排除条件下，提前预测 p(t+2) 的数学合同成立。
```

但这仍不是性能结果。Debug predictor 只是 correctness probe，不替换 baseline `p_core`，也不提交 `p(t+2)`。

## 4. Stage-4 Tile Budget 结果

使用你的定义：

```text
R = 7
M = first-step owned / staged tile
O = M + R halo on each side
C = M - R margin on each side
shared = p_next_local over M only
threads = 256
```

本地和 RTX 5090 服务器输出一致：

```text
GATE=STOP commit_ratio_lt_10_percent
```

关键结果：

```text
A:
  M=[32,24,20]
  C=[18,10,6]
  kept_tiles=60
  commit_points=64800
  commit/core=0.0319
  estimated CTA/SM=2

D:
  M=[40,24,24]
  C=[26,10,10]
  kept_tiles=25
  commit_points=65000
  commit/core=0.0320
  estimated CTA/SM=1
```

A/D 都低于 `10%`，因此按 gate 停止。

## 5. 技术解释

这条 CTA-local two-step route 的主要问题已经很清楚：

```text
R=7 太厚，M tile erosion 到 C 后 surface/volume loss 太大。
```

虽然 D 的单 tile `C/M` 有 `11.28%`，但受 O-inside-core 和 non-overlapping M tile 约束，meaningful case 中只能放下 25 个 D tiles，总 commit points 只有：

```text
65000 / 2033152 = 3.20%
```

这点 skipped work 不足以抵消：

```text
1. fused commit kernel 复杂度
2. d_p2_fused 存储
3. commit/copy 成本
4. retiled residual p_core 维护成本
5. 额外 launch / sync / NCU 调参成本
```

所以这不是简单 kernel tuning 能解决的问题。

## 6. 当前禁止继续的事项

建议保持以下路线停止：

```text
1. naive in-place fused commit
2. CUDA3D_CORE_2STEP_FUSED_COMMIT_V2
3. CUDA3D_CORE_RETILED_RESIDUAL
4. standalone predict+copy micro tuning
5. p_core z-pencil duplicate
6. PML fused z-slab
7. full-domain temporal blocking
8. MPI temporal blocking
```

特别是 naive in-place fused commit 仍然禁止：

```text
p0 同时是 old p(t-1) input 和 new p(t+1) output。
单 kernel 内没有 grid-wide synchronization。
如果一边读 global old p0，一边让其他 CTA 写 global new p0，会产生 old/new race。
```

## 7. Codex 建议的下一条路线

我建议下一阶段转向：

```text
CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE
```

核心思想：

```text
显式拆分 p_prev / p_curr / p_next，
不再让 p0 同时承担 old input 和 output。
```

这会从系统数据流层面消除 in-place `p0` race，让后续 temporal/dataflow optimization 有更干净的基础。

但这不是小改，应先做 design gate：

```text
1. pointer-swap audit
2. PML pressure update audit
3. PML memory array timing audit
4. source injection timing audit
5. receiver extraction timing audit
6. output correctness comparison plan
7. memory footprint estimate
8. rollback plan
```

## 8. 请求 Pro 决策

请 Pro 给出下一阶段方向：

```text
A. 同意停止 CTA-local core two-step route，进入 CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE design。
B. 要求重新定义更大的 tile/gate，继续尝试 fused commit V2。
C. 暂停 core temporal blocking，转向其他更高收益路线。
```

Codex 的建议是 A。

理由：

```text
debug correctness 已经证明数学可行；
stage-4 budget 证明当前 CTA-local 实现收益面太小；
真正瓶颈是 in-place pressure dataflow 和 R=7 halo 厚度。
```

下一步如果选 A，我会先写：

```text
docs/pressure_triple_buffer_pipeline_design.md
```

并只做 design/audit，不立刻大规模改代码。
