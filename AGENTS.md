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
- `CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY` with direct p1 x/y second derivatives replacing `vx/vy`
- `CUDA3D_PML_ZFACE_SHARED_VP_DEBUG` p-only S2/S4 and S4 staged-V variants
- `RECOMPUTE_X` / `RECOMPUTE_Y` / `RECOMPUTE_XYZ`
- PML tile block shape sweep
- `p_core` simple block shape sweep
- `-maxrregcount` / register cap sweep
- inject/extract small-kernel block-size reduction (`CUDA3D_INJECT_EXTRACT_BS512`)
- velocity-PML vx/vy component-owner split under current `32x4x2` tile geometry

Profiler gate：

- 没有 Nsight Compute 或等价硬件级 profiler evidence，不启动新的大结构重写。
- PML 数学路径改动必须通过 debug dump step 0/1/2。
- 性能结论必须包含 `perf_1gpu_6shots repeat`。
- 小候选没有 `>=2%` repeat speedup，不进入主线。
- prototype 没有 `>=5%` repeat speedup，不扩展范围。

当前活动主线：

1. `CUDA3D_WAVESTEP_ENGINE_V2`
   - 目标是重构 wave-step ownership，而不是继续 micro sweep。
   - 主攻 PML velocity + pressure 数据流，辅攻 core global-region temporal pipeline。
   - Phase 0 设计文档：`docs/wavestep_engine_v2_design.md`。
   - Phase 1 已实现：`CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`。
   - Phase 1 宏默认关闭，必须显式开启：
     - `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`
     - `CUDA3D_CPML_VMEM_DISABLE_MPI`
     - `CUDA3D_CPML_VMEM_DEBUG_FILL`
   - Phase 1 只允许 single GPU / single MPI rank，当前用于清理 CPML velocity memory ownership，不做 fusion。
   - Phase 1 gate：debug dump step 0/1/2、correctness、`perf_1gpu_6shots repeat slowdown <= 2%`。
   - 当前 Phase 1 结果：gate `continue`，相对 zmem all-mean WP speedup `1.032605x`，Gradient speedup `1.028648x`。
   - Phase 1 报告：`reports/wavestep_engine_v2_phase1_cpml_vmem_20260608_003000/phase1_report.md`。

2. `CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY` direct form 已停止
   - separate fused z-face kernel：correctness pass，但 mean WP `2.660077s`，慢于 zmem mean WP `2.434461s`。
   - inline p_pml fused branch：correctness pass，但 mean WP `2.692579s`，慢于 zmem mean WP `2.434461s`。
   - 禁止继续重复 direct p1 x/y second-derivative 替代 `vx/vy` global round trip。

3. `CUDA3D_PML_ZFACE_SHARED_VP_DEBUG` 已停止
   - S2 p-only shared pressure tile：correctness pass，但 mean WP `3.007605s`，相对同机 zmem mean WP `2.448577s` 只有 `0.814129x`。
   - S4 p-only shared pressure tile：correctness pass，但 WP `3.039426s`，相对同机 zmem mean WP 只有 `0.805605x`。
   - S4 staged-V shared velocity intermediate：correctness pass，但 mean WP `3.090552s`，相对同机 zmem mean WP 只有 `0.792278x`。
   - 禁止继续重复当前 shared-tile z-face VP 形态；问题不是数值，而是 1 CTA/SM 大 shared tile、额外同步和 shared velocity staging 开销吞掉了 global traffic savings。
   - 后续 PML z-face fusion 只有在新的 profiler/source-counter 证据证明不同 tile/dataflow 能显著降低 pressure critical path 时才允许重开。

当前下一步建议：

- 保留 `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` 作为 ownership scaffold。
- PML compact-state audit 已完成，当前 gate 失败，不进入 compact-state CUDA prototype。
- 停止 z-face fusion 局部路线，转入更大粒度的 global-region temporal pipeline 设计/原型。

### 2026-06-08 day sprint gate 结果

Phase 1 `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL` 复验通过：

- 三轮 `perf_1gpu_6shots` A/B 均在每个候选运行前重建 binary。
- all-mean WP speedup：`1.032329x`。
- all-mean Gradient speedup：`1.028370x`。
- correctness 与三轮 perf 输出对比均通过，max rel L2 `0`。
- 结论：保留为后续 wave-step ownership scaffold，但不视为大突破。

Phase 2 PML compact-state gate 已停止：

- 当前 CPML memory 已经是 axis-slab allocation，不是 full padded-domain state。
- `cpml_dbuf` 状态 footprint：`72.391 MiB`；六个 padded wavefield/cw2 array floor：`503.039 MiB`。
- safe z-face compact 只能覆盖 `memory_dz` 的 `84.93%`，剩余 edge/corner state 仍需保留。
- 静态 estimated WP speedup ceiling：`1.005x`，低于 `>=1.05x` meaningful prototype gate。
- NCU 短 profile 显示 pressure PML duration 基本不变：
  - zmem `cuda_fd3d_p_pml_tile_ns`：`189.840us`
  - CPML dbuf `cuda_fd3d_p_pml_tile_ns`：`190.293us`
  - velocity PML 改善：
    - zmem `cuda_fd3d_v_pml_tile_ns`：`71.493us`
    - CPML dbuf `cuda_fd3d_v_pml_tile_ns`：`66.000us`
- 结论：不要写 `CUDA3D_PML_COMPACT_STATE_*` prototype，除非后续 profiler 证明 CPML state layout 成为主瓶颈。

Day sprint 关键报告：

- `docs/day_20260608/cpml_vmem_dbuf_revalidation.md`
- `docs/day_20260608/pml_compact_state_audit.md`
- `docs/day_20260608/pml_state_ncu_summary.md`
- `docs/day_20260608/phase2_compact_state_gate_decision.md`
- `docs/day_20260608/global_temporal_pipeline_phase4_design.md`
- `reports/day_20260608/cpml_vmem_dbuf_summary.json`
- `reports/day_20260608/phase2_compact_state_gate_summary.json`
- `reports/day_20260608/phase4_global_temporal_pipeline_design_summary.json`

Phase 4 已开启 global temporal pipeline 设计门：

- zmem 短 NCU 中，sampled main kernels：
  - `cuda_fd3d_p_pml_tile_ns`：`189.562us`，约 `53.43%`。
  - `cuda_fd3d_p_core_ns`：`93.670us`，约 `26.40%`。
  - `cuda_fd3d_v_pml_tile_ns`：`71.610us`，约 `20.18%`。
- 单独优化 `p_core` 若要让 sampled main kernels 达到 `>=5%` speedup，至少需要 `18.04%` 的 `p_core` reduction。
- K=2 deep-core temporal geometry 约覆盖原 core `77.7%`；K=3 约 `58.1%`，当前先只考虑 K=2。
- 普通 CUDA kernel 没有 grid-wide barrier，不能写会跨 CTA 读取半更新 `p(t+1)` 的 fused two-step kernel。
- 下一步必须先做 K=2 deep-core byte/synchronization model；未证明依赖锥安全前，不写 temporal CUDA kernel。

Phase 4.1 K=2 temporal byte/synchronization model 已完成：

- 工具：`tools/temporal_pipeline_model.py`。
- 报告：`docs/day_20260608/temporal_pipeline_model.md`。
- gate：`docs/day_20260608/phase4_1_temporal_model_gate_decision.md`。
- K=2 deep-core coverage：`77.78%`。
- current p_core bytes/output estimate：`128.438`。
- ideal K=2 local-reuse upper bound：
  - p_core pair reduction：`35.25%`。
  - sampled-main speedup upper bound：`1.103x`。
- 但 direct K=2 CUDA prototype 已停止：
  - safe global-middle design 保留 global `p(t+1)` stencil traffic，无法达到 meaningful speedup。
  - cooperative grid-sync 需要 `70688` blocks 同时 resident，按保守 RTX 5090 假设超容量约 `51.98x`。
  - no-duplication CTA-local p_mid reuse 是唯一有 `>5%` 上界的路线，但 concrete CTA-local candidates 计入 halo duplication 后，local pair bytes 约为 baseline 的 `11.29x` 到 `21.30x`。
  - 因此当前 CTA-local p_mid 形态不仅属于已禁止的 CTA-local two-step 家族，在 byte model 上也失败；除非重做成 source-aware swept/wavefront ownership 设计。
- 下一步只允许做 `Phase 4.2 source-aware swept/wavefront temporal design`，必须先解决：
  - p_mid halo ownership。
  - source injection between substeps。
  - intermediate receiver extraction。
  - shell/PML reconciliation。
  - 不读取 half-updated values 的依赖锥证明。

Phase 4.2 source-aware temporal gate 已完成：

- 工具：`tools/source_aware_temporal_model.py`。
- 报告：`docs/day_20260608/source_aware_temporal_model.md`。
- gate：`docs/day_20260608/phase4_2_source_aware_temporal_gate_decision.md`。
- shot-local aggregate K=2 deep-core share：`73.22%`。
- source influence overlaps K=2 deep core：`0` shots。
- receiver footprint overlaps K=2 deep core：`0` shots。
- 结论：
  - source/receiver placement 不阻止 temporal blocking。
  - 但 `p(t+1)` ownership/synchronization 和 halo duplication 仍失败。
  - 当前停止 swept/wavefront temporal CUDA prototype。
- 后续自动推进应暂停 K=2 temporal 路线，转向 dominant `cuda_fd3d_p_pml_tile_ns` 的 pressure PML dataflow 或 wave-step scheduling。

Phase 4.3 pressure PML dataflow gate 已完成：

- 工具：`tools/pml_pressure_dataflow_audit.py`。
- 报告：`docs/day_20260608/pml_pressure_dataflow_audit.md`。
- gate：`docs/day_20260608/phase4_3_pressure_pml_dataflow_gate_decision.md`。
- pressure PML tiles：`113840 / 181232`。
- active thread efficiency：`65.60%`。
- shell active points：`4143640`，占 active points `21.67%`。
- `recompute_vz_after_update` 当前调用量：`152951552`。
- shared z-line cache 估算调用量：`29093740`，估算减少 `80.98%`。
- NCU-linked model：
  - `cuda_fd3d_p_pml_tile_ns` sampled-main share：`53.42%`。
  - modeled p_pml speedup：`1.573x`。
  - modeled sampled-main speedup：`1.242x`。
- 结论：允许打开一个 macro-default-off CUDA prototype：
  - `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
- 此路线只允许复用 pressure PML 内部 z-line 的 `vz_after_update` 中间量；不得重开：
  - `CUDA3D_PML_TILE_MASK_FASTPATH`
  - `CUDA3D_PML_ZFACE_P_SPECIALIZE`
  - `CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY`
  - `CUDA3D_PML_ZFACE_SHARED_VP_DEBUG`
  - `RECOMPUTE_X/Y/XYZ`
- prototype 必须先通过 debug dump step 0/1/2、correctness、`perf_1gpu_6shots repeat`；若 repeat 没有 `>=5%` meaningful WP speedup，立即停止。

Phase 4.4 pressure PML z-recompute cache prototype 已完成：

- 实现宏默认关闭：`CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`。
- 改动范围：仅 `cuda_fd3d_p_pml_tile_ns`。
- 数据流：CTA-local shared z-line cache 复用 `recompute_vz_after_update_from_old_mem` 中间值。
- ownership：`memory_dz_next` 仍只由 tile-owned active central z positions 写入。
- standalone z-cache：
  - debug dump step `0/1/2` 通过。
  - correctness rel L2：`0`。
  - `perf_1gpu_6shots` repeat mean WP speedup：`1.044955x`。
  - mean Gradient speedup：`1.045506x`。
  - 结论：有价值，但低于 standalone `>=5%` gate，不单独作为主线突破。
- 与 Phase 1 CPML vmem scaffold 组合：
  - flags：
    - `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`
    - `CUDA3D_CPML_VMEM_DISABLE_MPI`
    - `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
  - debug dump step `0/1/2` 通过。
  - correctness rel L2：`0`。
  - `perf_1gpu_6shots` repeat 3 轮输出对比全部通过。
  - mean WP speedup：`1.083390x`。
  - mean Gradient speedup：`1.080857x`。
  - 结论：组合候选通过 meaningful `>=5%` gate，但已被 direct-fill 版本取代。
- Direct-fill 组合候选：
  - 将 z-cache fill 从 linear loop + division/modulo 改为 direct fill：
    - 每个 thread 填自己的 central z cache entry。
    - `threadIdx.x < 4` 填 left halo。
    - `threadIdx.x < 3` 填 right halo。
  - debug dump step `0/1/2` 通过。
  - correctness rel L2：`0`。
  - `perf_1gpu_6shots` repeat 3 轮输出对比全部通过。
  - mean WP speedup：`1.100929x`。
  - mean Gradient speedup：`1.097530x`。
  - 结论：这是当前 best combo candidate。
- 已失败并禁止继续的子路线：
  - pressure-PML `vx/vy` shared-neighbor cache。
  - 该路线 correctness pass，但 mean WP speedup 只有 `0.419906x`，mean Gradient speedup `0.426565x`，性能灾难性退化。
- 下一步：
  - profile 组合候选，分解剩余 `cuda_fd3d_p_pml_tile_ns` latency。
  - 不得重开 shared `vx/vy` cache、tile-mask fastpath、z-face specialize/fusion 或 `RECOMPUTE_X/Y/XYZ`，除非新 profiler evidence 推翻当前结论。

Phase 4.5 combo candidate NCU profile 已完成：

- 报告：`docs/day_20260608/zrecomp_cache_cpml_combo_ncu_summary.md`。
- NCU sections：SpeedOfLight、MemoryWorkloadAnalysis、SchedulerStats、WarpStateStats、Occupancy。
- `cuda_fd3d_p_core_ns` duration：zmem `76.061us`，combo `75.306us`，基本不变。
- `cuda_fd3d_p_pml_tile_ns` duration：zmem `158.291us`，combo `142.902us`，kernel speedup `1.108x`。
- `cuda_fd3d_v_pml_tile_ns` duration：zmem `58.320us`，combo `53.101us`，kernel speedup `1.098x`。
- combo `p_pml_tile` 剩余特征：
  - eligible warps/scheduler：`0.798`。
  - No Eligible：`60.879%`。
  - achieved occupancy：`75.965%`。
  - block limit registers：`5`，block limit shared mem：`7`。
- 结论：
  - combo 主要改善 `p_pml` 与 `v_pml`。
  - `p_core` 仍是 L2/memory-throughput limited，短期不要回到简单 block/register sweep。
  - combo 后 `p_pml` 剩余瓶颈更像 issue/latency overhead；下一步若继续 pressure PML，应优先降低 z-cache fill 的 integer/division/control overhead。
  - 继续禁止 shared `vx/vy` cache。

Phase 4.6 direct-fill z-cache 已完成：

- 结果：direct-fill combo 将 Phase 4.4 combo mean WP speedup 从 `1.083390x` 提高到 `1.100929x`。
- mean Gradient speedup：`1.097530x`。
- debug/correctness/perf repeat 均通过。
- 当前 best candidate flags 仍为：
  - `CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL`
  - `CUDA3D_CPML_VMEM_DISABLE_MPI`
  - `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
- 当前 best implementation 是 direct-fill z-cache，不是 linear-loop z-cache。

Phase 4.7 direct-fill NCU profile 已完成：

- 报告：
  - `reports/day_20260608/directfill_combo_ncu_20260608_120449_summary.md`
  - `reports/day_20260608/directfill_combo_ncu_20260608_120449_summary.json`
- short-profile kernel duration：
  - `cuda_fd3d_p_core_ns`：zmem `75.942us`，direct-fill `75.270us`，基本不变。
  - `cuda_fd3d_p_pml_tile_ns`：zmem `158.438us`，direct-fill `134.099us`，kernel speedup `1.181x`。
  - `cuda_fd3d_v_pml_tile_ns`：zmem `58.794us`，direct-fill `53.590us`，kernel speedup `1.097x`。
- direct-fill `p_pml_tile` 剩余特征：
  - No Eligible：`59.885%`。
  - eligible warps/scheduler：`0.820`。
  - achieved occupancy：`74.662%`。
- 结论：
  - direct-fill 后 `p_pml_tile` 继续显著优于 zmem 和 linear-loop combo。
  - `p_core` 仍不应作为下一步小修目标。
  - `CUDA3D_PML_PRESSURE_ZCACHE_WARP_RANGE` 已测试并拒绝：
    - correctness pass，6 个输出 rel L2 全部 `0`。
    - `perf_1gpu_6shots` repeat mean WP speedup vs direct-fill：`0.997223x`。
    - mean Gradient speedup vs direct-fill：`0.997502x`。
  - 禁止继续重复 warp-broadcast active-range caching；shuffle/control overhead 没有换来收益。
  - 当前源码已恢复到 direct-fill z-cache best，不保留 warp-range 候选代码。

Phase 4.8 direct-fill SourceCounters profile 已完成：

- 报告：`reports/day_20260608/directfill_source_profile_summary.md`。
- raw CSV：`reports/day_20260608/directfill_p_pml_source_ncu.csv`。
- direct-fill `p_pml_tile` source-level 主要信号：
  - No Eligible 约 `60%`。
  - eligible warps/scheduler 约 `0.81`。
  - L1TEX scoreboard stall 约 `14.4 cycles/warp`。
  - uncoalesced global accesses 约 `19% excessive sectors`。
  - avg active threads/warp 约 `19.84`，not-predicated-off 约 `18.69`。
- 最高采样行集中在：
  - `mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);`
  - `mem_dxx/mem_dyy` CPML memory update。
  - `p0[outIndex]=2*__ldg(p1+outIndex)-p0[outIndex]...`
  - z-cache shared loads 已可见但不再是主导。
- `pml_local_mem_accum` 已测试并拒绝：
  - correctness pass，6 个输出 rel L2 全部 `0`。
  - `perf_1gpu_6shots` repeat mean WP speedup vs direct-fill：`1.000647x`。
  - mean Gradient speedup vs direct-fill：`0.998957x`。
  - 结论：编译器已基本处理好写后使用表达式，显式 `new_mem` 不满足 `>=2%` gate。
- `pml_p0_ldg` 已测试并拒绝：
  - correctness pass，6 个输出 rel L2 全部 `0`。
  - `perf_1gpu_6shots` repeat mean WP speedup vs direct-fill：`1.000054x`。
  - mean Gradient speedup vs direct-fill：`1.000694x`。
  - 结论：把 final pressure update 中旧 `p0[outIndex]` 读取改成 `__ldg(p0+outIndex)` 只有噪声级收益，不满足 `>=2%` gate。
- `zsafe_direct_shared` 已测试并拒绝：
  - correctness pass，6 个输出 rel L2 最大约 `2.180533e-10`。
  - `perf_1gpu_6shots` repeat mean WP speedup vs direct-fill：`0.966920x`。
  - mean Gradient speedup vs direct-fill：`0.965779x`。
  - 结论：对 z-safe tile 使用 shared `p1` 线缓存直接二阶差分会慢约 `3.3%`，更宽 halo 和额外 shared/p1 访问超过了省掉 z-recompute 的收益。
- ptxas cache policy 已测试并拒绝：
  - `-Xptxas -dlcm=ca`：perf repeat compare pass，mean WP `0.999263x`，mean Gradient `0.999576x`。
  - `-Xptxas -dlcm=cg`：perf repeat compare pass，mean WP `0.859344x`，mean Gradient `0.864052x`。
  - 结论：强制全局 load cache policy 不解决 direct-fill 的 L1TEX scoreboard stall；`cg` 明显破坏当前 cache reuse。
- `p_core_readonly_ldg` 已测试并拒绝：
  - correctness pass，6 个输出 rel L2 全部 `0`。
  - `perf_1gpu_6shots` repeat mean WP speedup vs direct-fill：`0.999319x`。
  - mean Gradient speedup vs direct-fill：`0.999254x`。
  - 结论：对 `cuda_fd3d_p_core_ns` 的 `p1/cw2` 显式使用 `__ldg` 不能改善 p_core memory path。
- `CUDA3D_INJECT_EXTRACT_BS512` 已测试并拒绝：
  - NCU 显示 `lint3d_inject_bell_extract_gpu_zz` 平均 duration 约 `5.109us`，SOL compute `0.040%`，SOL memory `6.699%`。
  - Nsight Compute 规则提示 grid 太小，只有 `0.0` full waves，属于小 kernel / launch 调度问题。
  - correctness pass，6 个输出 rel L2 全部 `0`。
  - `perf_1gpu_6shots` repeat mean WP speedup vs direct-fill：`0.999684x`。
  - mean Gradient speedup vs direct-fill：`0.998963x`。
  - 结论：把 inject/extract block size 从 `1024` 改到 `512` 不能解决 launch/small-grid 开销，不满足 `>=2%` gate。
- `v_pml` SourceCounters gate 已完成：
  - 报告：`docs/day_20260608/v_pml_source_profile_gate.md`。
  - `cuda_fd3d_v_pml_tile_ns`：No Eligible `44.891%`，eligible warps/scheduler `1.629`，warp cycles/issued inst `18.456`。
  - active threads/warp `23.700`，not-predicated threads/warp `21.670`，branch efficiency `86.970%`。
  - NCU rule 显示 L1TEX scoreboard stall 约 `11.8 cycles/warp`，uncoalesced excessive sectors 约 `22%`。
  - 静态预算拒绝 vx/vy component-owner split：tile 总量会变成当前 combined kernel 的 `1.985645x`，active component work 约 `1.963726x`。
  - 结论：当前 `32x4x2` tile 下不要写 vx/vy split kernel；后续 v_pml 必须从 memory layout / coalescing 设计入手。
- Nsight Systems scheduling gate 已完成：
  - 报告：`docs/day_20260608/scheduling_nsys_cuda_graph_gate.md`。
  - `cudaLaunchKernel` CPU API total：`1.845401s`，调用 `36,024` 次，平均 `51.227us`。
  - 但 GPU kernel total：`2.232398465s`，WP computing time：`2.238769s`。
  - WP 与 GPU kernel total 只差 `0.006370535s`，visible gap fraction `0.2846%`。
  - 即使完美消除该 gap，理想 WP speedup 也只有 `1.002854x`。
  - 结论：当前 single-GPU / single-MPI-rank `perf_1gpu_6shots` 不允许写 CUDA Graph / launch aggregation prototype，除非未来 Nsight Systems 或多 rank wall-clock 证据显示 `>2%` visible scheduling gap 或 GPU idle。
- PML active segment compaction model 已完成：
  - 工具：`tools/pml_active_segment_compaction_model.py`。
  - 报告：`docs/day_20260608/pml_active_segment_compaction_model.md`。
  - 当前 pressure-PML launched lanes：`29,143,040`；active lanes：`19,118,944`；lane efficiency：`65.60%`。
  - active z-line length histogram：
    - len `16`：`542,100` lines，`8,673,600` active lanes。
    - len `23`：`87,776` lines，`2,018,848` active lanes。
    - len `32`：`263,328` lines，`8,426,496` active lanes。
  - 普通 active-line list 只减少 `1.92%` launched lanes，sampled-main ceiling `1.011x`，拒绝。
  - exact active-point list ceiling：p_pml lane speedup `1.524x`，sampled-main ceiling `1.228x`，但 descriptor traffic 约 `72.933 MiB/step aggregate-shots`，只能作为设计路线。
  - len-16 half-warp packing ceiling：p_pml lane speedup `1.464x`，sampled-main ceiling `1.207x`，作为下一步设计门。
  - 关键边界：该路线是 lane-utilization / active segment ownership，不是重开已失败的 z-face direct derivative、z-face fusion 或 shared-VP 路线；必须保留 direct-fill pressure z-cache 的数值路径。

Phase 4.9 length-16 half-warp pressure-PML prototype 已完成并接受：

- 实现宏默认关闭：`CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK`。
- 报告：`docs/day_20260608/len16_halfwarp_pressure_pml_prototype.md`。
- 依赖 flags：
  - `CUDA3D_PML_RECOMPUTE_Z`
  - `CUDA3D_PML_ZMEM_IN_P`
  - `CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE`
  - `PmlTileBlockSize=32x4x2`
- 数据流：
  - host 将 pressure-PML tile 拆分为 residual tiles 与 whole length-16 active-z tiles。
  - residual tiles 继续使用 `cuda_fd3d_p_pml_tile_ns`。
  - length-16 tiles 使用新 kernel `cuda_fd3d_p_pml_len16_halfwarp_ns`。
  - 一个 warp 处理两条 length-16 z-line，lane `0..15` 与 `16..31` 分别处理一条线。
  - 保留 direct-fill pressure z-cache 数值路径和既有 `vx/vy` divergence path，不重开 z-face direct/fusion/shared-VP。
- debug dump gate：
  - `profile_1gpu` step `0/1/2` 全部通过。
  - step 2 仅 `p0` 出现 rel L2 `7.852061e-09`，低于容差；其他数组 rel L2 为 `0`。
- correctness：
  - 6 个输出对比通过。
  - correctness case 的 len16 tile 数为 `0`，因此主要验证 macro wiring/residual path；packed kernel 由 `profile_1gpu` debug dump 和 `perf_1gpu_6shots` 覆盖。
- `perf_1gpu_6shots` repeat：
  - 3 轮输出对比全部通过，max rel L2 `6.384336e-07`。
  - mean base WP：`2.207751s`。
  - mean candidate WP：`2.039080s`。
  - mean WP speedup vs direct-fill：`1.082719x`。
  - mean Gradient speedup vs direct-fill：`1.072448x`。
- 当前结论：
  - 这是 direct-fill 之后第一个通过 `>=5%` meaningful repeat gate 的 CUDA prototype。
  - 作为当前 RTX 5090 single-GPU best candidate 保留。
  - 估算相对旧 `zmem_reference` 的累计 WP speedup 约 `1.191983x`，但该数是 `direct-fill vs zmem` 与 `len16 vs direct-fill` 的乘积，只能作方向参考；正式对外表格需要同机同 session 重跑 zmem/direct-fill/len16。
- 下一步：
  - 对 len16 candidate 做 Nsight Compute source/profile，确认剩余瓶颈是否转向 memory coalescing、shared pressure、final `p0/mem_dzz` update 或 length-23 active segment。
  - 不要继续抠 z-cache fill、`new_mem` 表达式、final `p0` read-only load、z-safe shared `p1` direct second derivative、ptxas `dlcm` cache-policy sweep、p_core 显式 `__ldg`、inject/extract block-size 微调、当前 tile 下的 vx/vy split，或当前 single-GPU launch aggregation/CUDA Graph，除非有新的 profiler evidence。

Phase 4.10 len16 NCU profile 已完成：

- 报告：`docs/day_20260608/len16_halfwarp_ncu_profile.md`。
- NCU summary：
  - `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.md`
  - `reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json`
- 同口径 `profile_1gpu` NCU details：
  - direct-fill pressure-PML：`164.328us`。
  - len16 residual pressure-PML：`72.683us`。
  - len16 packed pressure-PML：`65.771us`。
  - len16 pressure-PML total：`138.453us`。
  - pressure-PML kernel-path speedup：`1.187x`。
  - sampled main-kernel total：direct-fill `323.608us`，len16 `297.248us`，speedup `1.0887x`。
- profile 解释：
  - `p_core` 与 `v_pml` 基本不变，len16 收益来自 pressure-PML ownership split。
  - residual pressure-PML branch efficiency 从 `75.530%` 提高到 `83.320%`。
  - 新 packed len16 kernel `No Eligible` 为 `73.827%`，eligible warps/scheduler 仅 `0.433`，属于 latency/issue-limited kernel。
- length-23 gate：
  - 不允许直接写简单 `CUDA3D_PML_PRESSURE_LEN23_*` prototype。
  - 原因：length-23 单独只能移除约 `0.790M` inactive lanes，且不能像 length-16 一样两条线塞进一个 warp；额外 launch/tile-list/control overhead 很可能吞掉收益。
  - 只有在 exact active-point / compact descriptor 预算证明 `>=5%` repeat speedup ceiling 时，才允许进入 CUDA prototype。
- 当前下一步：
  - Phase 4.11：exact active-point / compact descriptor budget。
  - 若 gate 失败，转向 `cuda_fd3d_p_pml_len16_halfwarp_ns` source-level drill-down 或 v-PML memory layout/coalescing 设计。

Phase 4.11 exact active-point / compact descriptor budget 已完成并拒绝 CUDA prototype：

- 工具：`tools/pml_compact_descriptor_budget.py`。
- 报告：`docs/day_20260608/pml_compact_descriptor_budget.md`。
- JSON：`reports/day_20260608/pml_compact_descriptor_budget.json`。
- post-len16 lane shape：
  - accepted len16 lanes：`19,908,928`。
  - active lanes：`19,118,944`。
  - length-23 remaining inactive lanes：`789,984`。
  - post-len16 pressure-PML sampled-main share：`46.58%`。
  - direct-fill -> len16 observed pressure-PML speedup：`1.1869x`。
  - direct-fill -> len16 lane ceiling：`1.4638x`。
  - observed lane-to-time efficiency factor：`0.811`。
- compact descriptor candidates vs accepted len16：
  - `exact_length23_points_only`：
    - lane reduction：`3.97%`。
    - p-PML lane ceiling：`1.0413x`。
    - sampled-main ceiling：`1.0188x`。
    - calibrated sampled-main estimate：`1.0153x`。
    - descriptor traffic：`7.701 MiB/step aggregate-shots`。
  - `exact_all_active_points`：
    - lane reduction：`3.97%`。
    - sampled-main ceiling：`1.0188x`。
    - calibrated sampled-main estimate：`1.0153x`。
    - descriptor traffic：`72.933 MiB/step aggregate-shots`。
- 决策：
  - 不写 exact active-point / compact descriptor CUDA prototype。
  - 不写简单 length-23 active-point prototype。
  - compact descriptor 只有在新设计证明扣除 descriptor/control overhead 后仍有 `>=5%` `perf_1gpu_6shots` repeat speedup ceiling 时才允许重开。
  - 当前下一步：
  - 优先做 `cuda_fd3d_p_pml_len16_halfwarp_ns` source-level drill-down，或转向 v-PML memory layout/coalescing 设计。

Phase 4.12 len16 source-level NCU profile 已完成：

- 报告：`docs/day_20260608/len16_halfwarp_source_profile.md`。
- artifacts：
  - `reports/day_20260608/len16_source_profile_20260608_1646/details.csv`
  - `reports/day_20260608/len16_source_profile_20260608_1646/details_summary.md`
  - `reports/day_20260608/len16_source_profile_20260608_1646/details_summary.json`
  - `reports/day_20260608/len16_source_profile_20260608_1646/source_hotlines.md`
  - `reports/day_20260608/len16_source_profile_20260608_1646/source_hotlines.json`
- kernel-level signal:
  - No Eligible：`73.545%`。
  - eligible warps/scheduler：`0.427`。
  - warp cycles/issued instruction：`33.970`。
  - Branch efficiency：`65.220%`。
  - NCU CPI stall：L1TEX scoreboard dependency 约 `24.6 cycles/warp`。
- source hot lines：
  - final `p0[base]` update + `cw2` line：约 `60.78%` parsed samples。
  - z-CPML `mem_dzz` update：约 `26.82%` parsed samples。
  - z-cache shared loads 已不是主导。
- 决策：
  - 不写 len16-only `p0 __ldg`、local `new_mem`、branch-only lower/upper/margin specialization、或 z-cache/shared-memory 小修 prototype。
  - 原因：直接 fill 路线已拒绝过 `p0 __ldg` 和 `new_mem`，当前 source profile 也显示是 final writeback / CPML state dependency，不是语法写法问题。
- 当前下一步：
  - 转向 v-PML memory layout / coalescing design。
  - 或提出更大粒度 pressure-PML memory-ownership design，但必须先证明 `>=5%` repeat speedup ceiling。

Phase 4.13 v-PML coalescing/layout gate 已完成并拒绝 CUDA prototype：

- 工具：`tools/v_pml_coalescing_layout_budget.py`。
- 报告：`docs/day_20260608/v_pml_coalescing_layout_budget.md`。
- JSON：`reports/day_20260608/v_pml_coalescing_layout_budget.json`。
- NCU anchor：
  - accepted len16 sampled main：`297.248us`。
  - `cuda_fd3d_v_pml_tile_ns`：`65.248us`，sampled-main share `21.95%`。
  - 若只优化 v-PML，要让 sampled-main 达到 `>=5%`，v kernel 需要约 `1.2770x` speedup。
- 当前 `32x4x2` velocity-PML 映射：
  - `threadIdx.x` 是 z 方向。
  - 一个 warp 对应固定 x/y 的连续 32 个 z-lane。
  - 对 `p1`、`mem_dx`、`mem_dy` 主路径已经是最有利的 coalescing 形态。
- reasoned v-only tile shape gate：
  - `current_32x4x2` launched lanes ratio：`1.0000`。
  - `z8_x8_y4` launched lanes ratio：`0.8830`，但 warp 被拆成 `4` 个 z segment。
  - `z8_x8_y4` optimistic v-kernel ceiling：`1.1325x`。
  - `z8_x8_y4` optimistic sampled-main ceiling：`1.0264x`，低于 `>=5%` prototype gate。
- 决策：
  - 不写 v-only tile-layout CUDA prototype。
  - 不做随机 `PmlTileBlockSize` sweep。
  - 不重开 current-geometry vx/vy component split。
  - 只有新模型证明扣除 tile-list/control overhead 后仍有 `>=5%` `perf_1gpu_6shots` repeat speedup ceiling，才允许重开 v-PML tile/layout 路线。
- 当前下一步：
  - 从更大粒度的 memory ownership / wave-step scheduling 入手，重点是减少 `vx/vy` global round trip 或 pressure final writeback/CPML state dependency。

Phase 4.14 wave-step async streams prototype 已完成并拒绝：

- 工具：`tools/wavestep_stream_overlap_model.py`。
- 模型报告：`docs/day_20260608/wavestep_stream_overlap_model.md`。
- prototype 报告：`docs/day_20260608/wavestep_async_streams_prototype.md`。
- perf repeat：
  - `reports/day_20260608/wavestep_async_perf6_repeat_20260608_175407/summary.md`
  - `reports/day_20260608/wavestep_async_perf6_repeat_20260608_175407/summary.json`
- 模型结论：
  - accepted len16 sampled main：`297.248us`。
  - `p_core`：`93.547us`。
  - `v_pml + pressure-PML serial path`：`203.701us`。
  - two-stream conservative schedule 的 sampled-main ceiling：`1.4592x`。
  - 只需约 `15.13%` realized overlap 即可达到 `1.05x` sampled-main，因此允许 prototype。
- prototype：
  - 临时宏：`CUDA3D_WAVESTEP_ASYNC_STREAMS`。
  - 调度：`p_core` 独立 stream，`v_pml -> p_pml_len16 -> p_pml_residual` 独立 PML stream，default stream 在 source injection/extraction 前等待二者。
  - 只改 host launch scheduling，不改 CUDA math kernel。
- 测试：
  - smoke pass：`smoke_1gpu_async_streams_smoke_datafixed_flags_20260608_174937`。
  - correctness pass：`wavestep_async_correctness_compare_20260608_175029`，6 个输出 rel L2 全部 `0`。
  - `perf_1gpu_6shots` repeat 3 轮输出对比全部 pass。
- 性能：
  - mean WP speedup：`1.005183x`。
  - mean Gradient speedup：`1.002855x`。
- 决策：
  - 拒绝 `CUDA3D_WAVESTEP_ASYNC_STREAMS` prototype，不进入主线。
  - 本地源码已移除临时代码，只保留模型、报告和测试摘要。
  - 不要重复 two-stream `p_core` vs PML overlap。
  - 不要基于本结果写 single-GPU CUDA Graph / launch aggregation。
  - 三 stream pressure residual/len16 fanout 只有在 Nsight Systems 证明有真实 concurrent execution headroom 且新模型证明扣除 contention 后仍有 `>=5%` repeat speedup ceiling 时才允许重开。
- 当前下一步：
  - 回到实际减少 global memory work 的 ownership 设计，重点关注 pressure-PML final `p0/cw2` writeback 与 CPML z-state dependency。

Phase 4.15 pressure-PML writeback / CPML state gate 已完成并拒绝 micro CUDA prototype：

- 工具：`tools/pressure_pml_writeback_state_model.py`。
- 报告：`docs/day_20260608/pressure_pml_writeback_state_model.md`。
- JSON：`reports/day_20260608/pressure_pml_writeback_state_model.json`。
- NCU/source anchor：
  - accepted len16 sampled main：`297.248us`。
  - len16 packed pressure-PML：`65.771us`，sampled-main share `22.13%`。
  - total pressure-PML：`138.453us`，sampled-main share `46.58%`。
  - parsed source samples：`15,712`。
- source group：
  - final `p0/p1/cw2` update：`60.78%` len16 source samples。
  - CPML `mem_dzz` update：`26.82%`。
  - z-cache shared loads：`1.92%`，已不再主导。
  - address/control visible lines：`4.31%`。
- speedup requirement：
  - 若只靠 packed len16 kernel 达到 `1.05x` sampled-main，需要 packed kernel speedup `1.2742x`。
  - 若只改 final `p0/p1/cw2` group，需要 local speedup `1.5482x`。
  - 若只改 CPML `mem_dzz` group，需要 local speedup `5.0614x`。
  - 若 final + `mem_dzz` 一起改，需要 group speedup `1.3257x`。
- 决策：
  - 拒绝 writeback/state micro CUDA prototype。
  - 不重试 len16 `p0 __ldg`、local `new_mem`、ptxas cache policy、branch-only lower/upper specialization、或 z-cache fill/shared-cache 小修。
  - 原因：热点是数学必需的二阶时间推进写回和递归 CPML z-state，不是语法或 cache policy 问题；已有同类候选均为噪声级收益。
- 允许重开条件：
  - 只有状态表示或时间推进设计证明能真正减少 old-`p0`/`cw2` 或 `mem_dzz` traffic，并且扣除额外 storage/control 后仍有 `>=5%` `perf_1gpu_6shots` repeat speedup ceiling，才允许写 CUDA prototype。
- 当前下一步：
  - 若继续底层核心重写，进入 math-level pressure state representation / PML ownership design gate。
  - 或先做 zmem/direct-fill/len16/current-best 同 session 正式总提速表，固化当前 best。

Phase 4.16 formal same-session speed table 已完成：

- 报告目录：`reports/day_20260608/formal_current_best_table_20260608_182525/`。
- 远端隔离 worktree：
  - `/work/wenzhe/cuda3D/.codex_worktrees/formal_table_20260608_182525`
- case：`perf_1gpu_6shots`。
- rounds：`3`。
- 每轮顺序：
  - 重建 `zmem` binary 并运行。
  - 重建 `directfill` binary 并运行，对比同轮 `zmem` 输出。
  - 重建 `len16_current_best` binary 并运行，对比同轮 `zmem` 输出。
- `directfill` vs `zmem`：
  - mean WP speedup：`1.099957x`。
  - mean Gradient speedup：`1.097977x`。
  - mean elapsed speedup：`1.105408x`。
  - all compare pass：`True`。
  - max rel L2：`0`。
- `len16_current_best` vs `zmem`：
  - mean WP speedup：`1.192835x`。
  - mean Gradient speedup：`1.179213x`。
  - mean elapsed speedup：`1.156108x`。
  - mean candidate WP：`2.031753s`。
  - all compare pass：`True`。
  - max rel L2：`6.384336e-07`。
  - max abs：`4.768372e-06`。
- 决策：
  - `CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK` + direct-fill z-cache + CPML vmem double-buffer scaffold 是当前 RTX 5090 single-GPU formal best。
  - 当前正式累计 WP speedup vs `zmem_reference`：`1.192835x`。
  - 正式结果仍未达到 `1.5x` 存档阈值，因此不创建 `archives/speedups/1.5x_*`。
- 当前下一步：
  - 若继续 CUDA 核心结构重写，应从 math-level pressure state representation / PML ownership 设计 gate 开始，而不是继续微调已拒绝路线。

Phase 4.17 pressure state representation gate 已完成并拒绝 CUDA prototype：

- 工具：`tools/pressure_state_representation_model.py`。
- 报告：`docs/day_20260608/pressure_state_representation_model.md`。
- JSON：`reports/day_20260608/pressure_state_representation_model.json`。
- NCU/formal anchor：
  - sampled main：`297.248us`。
  - `p_core` share：`31.47%`。
  - `v_pml` share：`21.95%`。
  - pressure-PML share：`46.58%`。
  - len16 packed pressure-PML share：`22.13%`。
  - formal current-best WP speedup vs zmem：`1.192835x`。
- 当前二阶 pressure update 每点最小 state traffic：
  - `p_prev_read`：`4B`。
  - `p_cur_read`：`4B`。
  - `cw2_read`：`4B`。
  - `p_next_write`：`4B`。
  - total：`16B`。
- 已审查并拒绝：
  - `delta_pressure_state`：
    - 精确代数可行，但每点最小 traffic 从 `16B` 增加到 `20B`。
    - sampled-main effect 若所有 pressure updates 都这样做约 `0.8957x`，变慢。
  - `scaled_pressure_q_only` (`q=p/cw2`)：
    - 时间更新代数看似可去掉 final `cw2` load。
    - 但所有 pressure stencil 都必须重建 `p=cw2*q`。
    - `p_core` 单点至少从 `1` 次 `cw2` 读取变成 `>=29` 个 pressure value reconstruction；`p_core+v_pml` 合计 `53.42%` sampled-main 处于风险区。
  - `scaled_pressure_dual_p_and_q`：
    - 保留 `p` 和 `q` 两套表示，避免 stencil 重建，但每点最小 traffic 从 `16B` 增加到 `32B`。
  - `first_order_full_domain_velocity_pressure`：
    - 不是 bitwise 等价替换，会改变当前 mixed second-order core。
    - 每 core 点最多省 `4B` old-p read，但至少新增 `24B` velocity read/write state traffic。
  - `precomputed_cw2dt`：
    - 只减少乘法，不减少 `cw2` 的 4B global load。
  - `half_or_compressed_cw2`：
    - 当前精度契约下拒绝；即使理想 `2x` 加速 len16 `cw2` source line，sampled-main ceiling 也只有 `1.0282x`。
  - `cpml_mem_dzz_rescaled_state`：
    - 代数缩放不能消除 recursive state 的每步 read/write；`mem_dzz` alone 需要 `5.0614x` local speedup 才能触及 gate。
- 决策：
  - 拒绝 pressure state representation CUDA prototype。
  - 不写 `q=p/cw2`、delta pressure state、dual `p/q`、full-domain first-order velocity-pressure、precomputed `cw2dt`、compressed `cw2` 或 `mem_dzz` rescale prototype。
- 当前下一步：
  - 转向 PML `vx/vy` round-trip ownership design，必须先有 `>=5%` model 再写 CUDA。
  - source-aware multi-step/wavefront 只有解决 synchronization/halo ownership 后才允许重开。
  - precision-relaxation 只有用户明确给出新 tolerance policy 才允许研究。

Phase 4.18 PML `vx/vy` round-trip ownership gate 已完成并拒绝 CUDA prototype：

- 工具：`tools/pml_vxvy_roundtrip_ownership_model.py`。
- 报告：`docs/day_20260608/pml_vxvy_roundtrip_ownership_model.md`。
- JSON：`reports/day_20260608/pml_vxvy_roundtrip_ownership_model.json`。
- timing anchor：
  - sampled main：`297.248us`。
  - `p_core`：`93.547us`。
  - `v_pml`：`65.248us`。
  - pressure-PML total：`138.453us`。
  - formal current-best WP speedup vs zmem：`1.192835x`。
- generous savable-time model：
  - len16 unknown/unparsed source time 全部算给 `vx/vy`：`4.056us`。
  - residual pressure-PML 额外慷慨假设 `20%` 可省：`14.537us`。
  - total generous `vx/vy` round-trip savable time：`18.593us`。
  - 在该慷慨预算下，为达到 `1.05x` sampled-main speedup，velocity/CPML work duplicate factor 必须 `<=1.068`。
- CTA-local macro tile 结果：
  - current pressure tile `4x2`：
    - duplicate v work：`4.085x`。
    - sampled-main speedup：`0.6193x`。
  - macro `8x4`：
    - duplicate v work：`2.606x`。
    - sampled-main speedup：`0.7752x`。
  - macro `16x8`：
    - duplicate v work：`1.866x`。
    - sampled-main speedup：`0.8868x`。
  - macro `16x16`：
    - duplicate v work：`1.620x`。
    - shared velocity cache：`94,208B`，仍在 conservative `96KiB` limit 内。
    - sampled-main speedup：`0.9315x`。
  - macro `32x8` / `32x16`：
    - duplicate v work 分别 `1.743x` / `1.497x`，但 shared cache 分别 `101,376B` / `174,080B`，超出 conservative limit。
- ideal no-duplicate cross-CTA owner：
  - duplicate v work：`1.000x`。
  - sampled-main ceiling：`1.0667x`。
  - 决策：拒绝为 ordinary CUDA prototype，因为需要 velocity values 计算一次后被 neighboring pressure CTAs 直接消费；普通 CUDA 没有跨 CTA register/shared exchange 或适合当前网格的 grid-wide barrier。
- 决策：
  - 拒绝 PML `vx/vy` round-trip ownership CUDA prototype。
  - 不写 CTA-local `vx/vy` shared-cache fusion。
  - 不重开 `RECOMPUTE_X/Y/XYZ` 或 direct p1 x/y derivative replacement。
  - 不重开 current-geometry `vx/vy` component-owner split。
  - 不写依赖 cross-CTA shared values 的 ordinary CUDA producer-consumer fusion。
- 当前下一步：
  - source-aware multi-step / wavefront design 只有在先解决 synchronization 和 halo ownership proof 后才允许重开。
  - precision-relaxation 只有用户明确给出新 tolerance policy 才允许研究。
  - 如果 exact CUDA-core 路线继续被 gate 掐掉，可以转向 application-level multi-shot batching。

Phase 4.19 source-aware wavefront synchronization gate 已完成并拒绝 CUDA prototype：

- 工具：`tools/source_aware_wavefront_sync_model.py`。
- 报告：`docs/day_20260608/source_aware_wavefront_sync_model.md`。
- JSON：`reports/day_20260608/source_aware_wavefront_sync_model.json`。
- current-best rebase：
  - sampled main：`297.248us`。
  - `p_core`：`93.547us`，sampled-main share `31.47%`。
  - formal current-best WP speedup vs zmem：`1.192835x`。
  - ideal K=2 p_core pair reduction：`35.25%`。
  - ideal K=2 sampled-main speedup on current best：`1.1248x`。
  - 要达到 `1.05x` sampled-main speedup，需要 `15.13%` p_core reduction，相当于 ideal saving 的 `42.92%`。
- source / receiver gate：
  - aggregate K=2 deep-core share：`73.22%`。
  - source overlap shots：`0`。
  - receiver overlap shots：`0`。
  - 结论：source / receiver placement 不阻止 K=2 temporal blocking。
- synchronization / ownership gate：
  - p_core grid blocks：`70688`。
  - conservative resident block capacity：`1360`。
  - cooperative-grid over-capacity factor：`51.98x`。
- ordinary CUDA candidate 结果：
  - `safe_global_middle_two_kernel`：safe，但保留 global `p(t+1)` materialization 和 reload，speedup ceiling `1.0000x`。
  - `cooperative_grid_full_core_k2`：ideal ceiling `1.1248x`，但 grid 超 resident capacity 约 `52x`。
  - `cta_local_diamond_k2`：ordinary CUDA 可写，但 halo duplication 后 concrete candidates 需要 `11.29x` 到 `21.30x` baseline pair bytes。
  - `multi_kernel_global_wavefront`：safe，但仍 materialize global `p_mid`，且增加许多 small launches，speedup ceiling `1.0000x`。
  - `persistent_wavefront_without_global_barrier`：需要普通 CUDA 不具备的跨 CTA register/shared ownership。
  - `ideal_no_dup_source_aware_wavefront`：唯一 meaningful ceiling，但不是 ordinary CUDA implementation。
- 决策：
  - 拒绝 source-aware K=2 wavefront CUDA prototype。
  - 不写 ordinary CUDA K=2 source-aware wavefront prototype。
  - 不写 multi-kernel global-middle wavefront prototype。
  - 不写 CTA-local diamond temporal prototype。
  - 不写依赖 cross-CTA shared/register values 的 persistent-kernel wavefront。
- 当前下一步：
  - 在 exact CUDA-core 路线下，今日已知结构性方向基本被 gate 收口。
  - 后续若继续提速，优先转向 application-level multi-shot batching / scheduling。
  - precision relaxation 只有用户明确放宽 tolerance policy 后才允许研究。
  - 未来只有在发现具体 hardware/runtime cross-CTA ownership primitive 后，才允许重开 no-duplicate wavefront temporal blocking。

Phase 4.20 same-GPU multi-rank scheduling probe 已完成并拒绝：

- 报告：`reports/day_20260608/multirank_samegpu_sched_20260608_193042/summary.md`。
- JSON：`reports/day_20260608/multirank_samegpu_sched_20260608_193042/summary.json`。
- 远端 worktree：`/work/wenzhe/cuda3D/.codex_worktrees/multirank_samegpu_20260608_193042`。
- 测试目标：
  - 使用当前 best binary。
  - 同一张 RTX 5090 上设置 `CUDA_VISIBLE_DEVICES=0`。
  - 对 `perf_1gpu_6shots` 分别运行 `np=1/2/3`，验证多 MPI rank 共享同一 GPU 分炮是否能提高 wall-clock throughput。
- 判据：
  - 多 rank 时程序打印的 `WP computing time` 是 root-rank local 口径，不能作为正式 wall-clock speedup。
  - 调度层结论必须优先看 `/usr/bin/time` elapsed 与 `Gradient TIME all`。
- 单轮结果：
  - `np=1`：elapsed `2.990s`，Gradient `2.165543s`，输出 `6` 个文件。
  - `np=2`：elapsed `3.370s`，Gradient `2.311468s`，elapsed speedup `0.8872x`，Gradient speedup `0.9369x`。
  - `np=3`：elapsed `3.250s`，Gradient `2.328266s`，elapsed speedup `0.9200x`，Gradient speedup `0.9301x`。
  - `np=2` vs `np=1` correctness pass，max rel L2 `0`。
  - `np=3` vs `np=1` correctness pass，max rel L2 `0`。
- 决策：
  - 拒绝 same-GPU multi-rank oversubscription。
  - 不进行 same-GPU `np=2/3` repeat benchmark。
  - 不把 root-rank printed WP 作为 multi-rank 调度层 speedup 证据。
- 当前下一步：
  - 若继续 application-level scheduling，必须转向 true multi-GPU / multi-job batching，而不是同卡多 rank 抢占。
  - true multi-GPU 调度必须比较 elapsed、`Gradient TIME all`、输出 correctness，并明确 GPU 数、rank 数、shot 分配方式。

Phase 4.21 true multi-GPU / multi-job batching protocol 已完成，当前平台 defer：

- 工具：`tools/multigpu_batching_protocol.py`。
- 报告：`docs/day_20260608/true_multigpu_batching_protocol.md`。
- JSON：`reports/day_20260608/true_multigpu_batching_protocol.json`。
- 当前 RTX 5090 服务器只暴露 `1` 张 GPU：
  - `nvidia-smi -L` 仅显示 `GPU 0: NVIDIA GeForce RTX 5090`。
  - 因此当前平台不能验收 true multi-GPU batching。
- 现有代码要求：
  - `src/main.cu` 从输入文件读取 `gpus_p_node`。
  - CUDA 设备映射为 `cudaSetDevice(mytid % gpus_p_node)`。
  - shot 分配为 `sht_num[is * ntids + mytid]`。
  - true one-rank-per-GPU run 必须让三者一致：
    - `mpirun -np N`
    - `CUDA_VISIBLE_DEVICES` 暴露 `N` 张卡
    - 输入文件最后一行 `gpus_p_node=N`
- `perf_1gpu_6shots` shot-balance 上限：
  - `1` GPU：`[6]`，ideal `1.0000x`。
  - `2` GPUs：`[3,3]`，ideal `2.0000x`。
  - `3` GPUs：`[2,2,2]`，ideal `3.0000x`。
  - `4` GPUs：`[2,2,1,1]`，ideal `3.0000x`，受 6 炮数量限制。
  - `6` GPUs：`[1,1,1,1,1,1]`，ideal `6.0000x`。
- 决策：
  - 当前平台 defer true multi-GPU validation。
  - 不再做 same-GPU oversubscription。
  - 不把 `run_benchmark.py --gpus` 当作完整 true multi-GPU 配置；它只设置 `CUDA_VISIBLE_DEVICES`，还必须配套 input override。
- 未来验收 gate：
  - 至少 `2` 张 visible GPUs。
  - current-best binary，同一 case，同一 session。
  - 为 `N` GPU 创建 input copy，最后一行 `gpus_p_node=N`。
  - `np=N`、`CUDA_VISIBLE_DEVICES` 数量为 `N`。
  - 3 轮 repeat，输出对比全部 pass。
  - 使用 elapsed 和 `Gradient TIME all` 报告 speedup；root-rank printed WP 仅作诊断。

Phase 4.22 host / setup overhead gate 已完成：

- 工具：`tools/host_setup_overhead_gate.py`。
- 报告：`docs/day_20260608/host_setup_overhead_gate.md`。
- JSON：`reports/day_20260608/host_setup_overhead_gate.json`。
- current-best timing anchor：
  - mean elapsed：`2.970s`。
  - mean `Gradient TIME all`：`2.155902s`。
  - mean WP：`2.031753s`。
  - elapsed - Gradient：`0.814098s`，占 elapsed `27.41%`。
  - elapsed - WP：`0.938247s`，占 elapsed `31.59%`。
- current-best speedup vs zmem：
  - elapsed：`1.1560x`。
  - Gradient：`1.1792x`。
  - WP：`1.1928x`。
- 5% elapsed gate：
  - 要让 current-best elapsed 再提升 `1.05x`，需要节省 `0.141429s`。
  - 这相当于 `elapsed - Gradient` 的 `17.37%`。
  - 若能移除 `25%` 的 `elapsed - Gradient`，理论 elapsed speedup vs current best 为 `1.0736x`。
  - 若能移除 `50%`，理论 elapsed speedup vs current best 为 `1.1588x`。
- 决策：
  - 不盲目修改 host/setup 路径。
  - host/setup 路线只有在 Nsight Systems、CPU sampling 或 targeted timers 证明某个具体热点有 `>=5%` elapsed-speedup ceiling 后才允许写 prototype。
- 禁止：
  - 不移动计时点后声称加速。
  - 不跳过输出生成或 correctness 工作。
  - 不优化 `run_benchmark.py` output copy 来解释该 elapsed 指标，因为 output copy 在 `/usr/bin/time` 之后。
- 当前下一步：
  - 做 current-best host/setup profiling 或 targeted timer，分解 `0.814s` 的来源。

Phase 4.23 host/setup targeted timer probe 已完成：

- 实现宏默认关闭：`CUDA3D_HOST_SETUP_TIMERS`。
- 修改范围：
  - `src/main.cu`：main setup phase timers。
  - `src/optimization_cuda.cu`：`cal_fwi_grad_3d` pre-Gradient init timer。
  - `tools/host_setup_timer_summary.py`：解析 timer run log。
- 报告：`reports/day_20260608/host_setup_timer_probe_20260608_203508/summary.md`。
- JSON：`reports/day_20260608/host_setup_timer_probe_20260608_203508/summary.json`。
- 远端 worktree：`/work/wenzhe/cuda3D/.codex_worktrees/host_setup_timers_20260608_203508`。
- 构建：
  - timer binary 使用 current-best flags 追加 `-DCUDA3D_HOST_SETUP_TIMERS`。
  - default-off current-best build 通过，确认默认行为不依赖 timer macro。
- 正确性：
  - timer binary vs formal len16 current-best r1 输出对比 pass。
  - 6 个输出 max rel L2 `0`，max abs `0`。
- timer probe 结果：
  - elapsed：`2.980s`。
  - `Gradient TIME all`：`2.162907s`。
  - WP：`2.046621s`。
  - elapsed - Gradient：`0.817093s`。
  - measured pre-Gradient setup：`0.238399s`。
  - unaccounted elapsed-minus-Gradient：`0.578694s`。
  - 主要 measured stage：
    - `gpu_setup`：`0.174303s`。
    - `cal pre_gradient_init`：`0.022553s`。
    - `shot_list`：`0.022419s`。
    - `root_model_read`：`0.018118s`。
- 决策：
  - 不写 blind host/setup optimization prototype。
  - `gpu_setup` 主要是 CUDA device/context setup，一次性启动成本，不能伪装成 CUDA kernel speedup。
  - 当前 `0.578694s` unaccounted gap 主要在 after-MPI timer 外部，可能包含 bash/oneAPI source、mpirun 启动、`MPI_Init` 和 finalization。
- 当前下一步：
  - 若继续 wall-clock 路线，应增加 process-level timer 或 Nsight Systems OS/runtime profile，拆 `MPI_Init`/mpirun/source/finalize。
  - CUDA-core 优化结论仍以 `Gradient TIME all` 与 WP 为主。

Phase 4.24 process-level timer probe 已完成：

- 扩展 `CUDA3D_HOST_SETUP_TIMERS`：
  - `src/main.cu` 新增 `gettimeofday` wall-clock process timer。
  - 计时 `MPI_Init`、main after-MPI to pre-finalize、`MPI_Finalize`、process total。
  - `tools/host_setup_timer_summary.py` 新增 process accounting。
- 报告：`reports/day_20260608/process_timer_probe_20260608_205311/summary.md`。
- JSON：`reports/day_20260608/process_timer_probe_20260608_205311/summary.json`。
- 远端 worktree：`/work/wenzhe/cuda3D/.codex_worktrees/process_timers_20260608_205311`。
- 正确性：
  - process timer binary vs formal len16 current-best r1 输出对比 pass。
  - 6 个输出 max rel L2 `0`，max abs `0`。
  - default-off current-best build 通过。
- process timer 结果：
  - elapsed：`3.220s`。
  - `Gradient TIME all`：`2.161705s`。
  - elapsed - Gradient：`1.058295s`。
  - `MPI_Init`：`0.254292s`。
  - main after-MPI to pre-finalize：`2.418194s`。
  - `MPI_Finalize`：`0.000283s`。
  - process total：`2.672769s`。
  - elapsed - process total：`0.547231s`。
  - measured pre-Gradient setup：`0.250119s`。
  - known non-Gradient time including process shell/MPI/finalize/post-free：`1.053080s`。
  - residual after known non-Gradient timers：`0.005215s`。
- 决策：
  - 当前 elapsed-vs-Gradient gap 已基本闭合，不再作为 CUDA-core 优化路线。
  - 最大 wall-clock 非计算项来自 process 外壳和 MPI/context 初始化：
    - `/usr/bin/time` command shell / `source setvars` / `mpirun` wrapper 约 `0.547s`。
    - `MPI_Init` 约 `0.254s`。
    - CUDA `gpu_setup` / context setup 约 `0.186s`。
  - 这些只能作为 benchmarking/driver/startup policy 或 long-running service/multi-shot batching 议题，不能作为 CUDA kernel speedup。
- 当前下一步：
  - 停止 host/setup wall-clock 小修。
  - 若继续提速，应回到 compute metric：`Gradient TIME all` / WP，或等待 true multi-GPU 平台做 batching。

Phase 4.25 cal-loop internal timer probe 已完成：

- 扩展 `CUDA3D_HOST_SETUP_TIMERS`：
  - `src/optimization_cuda.cu` 新增 `cal_loop` timers。
  - 解析器 `tools/host_setup_timer_summary.py` 新增 `Cal Loop Timers` 表。
- 报告：`reports/day_20260608/cal_loop_timer_probe_20260608_212019/summary.md`。
- JSON：`reports/day_20260608/cal_loop_timer_probe_20260608_212019/summary.json`。
- 远端 worktree：`/work/wenzhe/cuda3D/.codex_worktrees/cal_loop_timers_20260608_212019`。
- 正确性：
  - cal-loop timer binary vs formal len16 current-best r1 输出对比 pass。
  - 6 个输出 max rel L2 `0`，max abs `0`。
  - default-off current-best build 通过。
- timer 结果：
  - elapsed：`2.990s`。
  - `Gradient TIME all`：`2.164033s`。
  - WP：`2.044622s`。
  - cal `pre_gradient_init`：`0.023527s`。
  - `cal_loop` across 6 shots：
    - `obs_setup`：`0.002679s`。
    - `domain_setup`：`0.000010s`。
    - `wavefield_prep`：`0.049816s`。
    - `fd_call`：`2.089376s`。
    - `output_write`：`0.004491s`。
    - `cleanup`：`0.002593s`。
    - `copy_reduce`：`0.015053s`。
- 决策：
  - 不写 host-side `vc/vc_pad` wavefield prep optimization prototype。
  - 不写 output write / cleanup / copy-reduce micro prototype。
  - 原因：最大非-FD项 `wavefield_prep` 即使理想消除，也只有约 `2.4%` Gradient speedup ceiling；不足 `>=5%` prototype gate。
  - `fd_call` 仍占 Gradient 主体，后续 compute route 仍应看 CUDA kernels 本身。
- 当前下一步：
  - 停止 host/pre-FD loop 小修。
  - 计算核心提速只能继续从 `fd_3d_f` 内部 kernel/dataflow 入手，或等待 true multi-GPU batching 平台。

Phase 4.26 pressure-PML len32 full-warp specialization budget 已完成并拒绝 CUDA prototype：

- 工具：`tools/pml_len32_fullwarp_specialization_budget.py`。
- 报告：`docs/day_20260608/pml_len32_fullwarp_specialization_budget.md`。
- JSON：`reports/day_20260608/pml_len32_fullwarp_specialization_budget.json`。
- 背景：
  - Phase 4.9 已接受 len16 half-warp packing。
  - len23 exact/descriptor compaction 已在 Phase 4.11 拒绝。
  - 本 gate 检查 residual pressure-PML 中 length-32 full-active z-line 是否值得单独 full-warp specialization。
- 结果：
  - sampled main：`297.248us`。
  - residual pressure-PML：`72.683us`。
  - length-32 line share：`75.00%`。
  - length-32 active-lane share：`80.67%`。
  - 要让 sampled-main 达到 `>=1.05x`，length-32 本地需要约 `1.3182x` 到 `1.3507x` speedup。
  - 对应本地时间减少约 `24.14%` 到 `25.97%`。
- 上界场景：
  - direct-fill source-visible address/control proxy 作用在 length-32：sampled-main `1.0080x`。
  - address/control proxy 作用在整个 residual kernel：sampled-main `1.0107x`。
  - perfect branch-efficiency 作用在 length-32：sampled-main `1.0340x`。
  - perfect branch-efficiency 作用在整个 residual kernel：sampled-main `1.0425x`。
  - 20% length-32 local speedup：sampled-main `1.0411x`。
- 决策：
  - 不写 `CUDA3D_PML_PRESSURE_LEN32_FULL_WARP_SPECIALIZE`。
  - 不写 branch/control-only length-32 residual split。
  - 原因：length-32 residual 没有 inactive-lane saving；单独 kernel 必须产生约 `1.32x-1.35x` 本地收益，而现有 branch/control 与 NCU proxy 都低于 `>=5%` sampled-main gate。
- 重开条件：
  - 只有 source-level profile 能单独分离 length-32 residual，并证明扣除额外 launch/tile-list/control overhead 后仍有 `>=5%` repeat speedup ceiling，才允许重开。

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
