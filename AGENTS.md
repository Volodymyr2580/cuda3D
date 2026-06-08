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
- 下一步不要继续抠 z-cache fill、`new_mem` 表达式、final `p0` read-only load、z-safe shared `p1` direct second derivative、ptxas `dlcm` cache-policy sweep、p_core 显式 `__ldg` 或 inject/extract block-size 微调，应转向更大粒度的 pressure-PML divergence / CPML memory traffic 结构，或单独设计 CUDA Graph / wave-step scheduling 级优化。

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
