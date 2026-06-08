# Architecture Decision Log

This file records CUDA3D optimization route decisions so future agents do not reopen failed paths without new evidence.

## 2026-06-08 - Start CUDA3D_WAVESTEP_ENGINE_V2

Decision:

```text
Start CUDA3D_WAVESTEP_ENGINE_V2 on branch exp/wavestep-engine-v2-pml-vp-fusion.
```

Reason:

```text
The zmem_reference baseline is stable. Further block-size, register-cap, face-split, and pressure-only split routes have not produced meaningful repeat speedup.
The remaining high-leverage path is ownership/dataflow rewrite, especially PML velocity -> pressure fusion.
```

Current stable baseline flags:

```text
-O3 -arch=sm_120 --use_fast_math
-DCUDA3D_PML_RECOMPUTE_Z
-DCUDA3D_PML_TILE_LIST
-DCUDA3D_PML_ZMEM_IN_P
-DPmlTileBlockSize1=32
-DPmlTileBlockSize2=4
-DPmlTileBlockSize3=2
```

## 2026-06-08 - Accepted Phase 1 CPML VMEM Double Buffer

Decision:

```text
Keep CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL macro-gated and default-off.
Use it as the ownership-clean scaffold for PML fused VP.
Proceed to PML_REGION_FUSED_VP_ZFACE_ONLY design/prototype.
```

Implemented macros:

```text
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_CPML_VMEM_DEBUG_FILL
CUDA3D_CPML_VMEM_DISABLE_MPI
```

Evidence:

```text
debug fill smoke/correctness: pass
debug dump step 0/1/2 vs zmem_reference: pass
release smoke/correctness/perf6/perf6_repeat comparisons: pass
all output rel_l2: <= 1e-5
```

Performance A/B:

```text
Phase1 perf6 mean WP:        2.365721s
ZMEM pre mean WP:            2.450038s, speedup 1.035641x
ZMEM post mean WP:           2.435677s, speedup 1.029570x
ZMEM all mean WP:            2.442857s, speedup 1.032605x
ZMEM all mean Gradient:      2.555540s
Phase1 all mean Gradient:    2.484369s, speedup 1.028648x
```

Report:

```text
reports/wavestep_engine_v2_phase1_cpml_vmem_20260608_003000/phase1_report.md
```

## Stop List Still Applies

Do not invest in:

```text
PML zface pressure-only split
PML fused z-slab pressure-only split
p_core z-pencil duplicate
CTA-local core two-step fused commit
standalone predict/copy tuning
block-size sweep
register cap sweep
RECOMPUTE_X/Y/XYZ global recompute
naive in-place pressure fusion
full MPI temporal blocking
```

Next allowed route:

```text
CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY
```

Hard gate:

```text
meaningful case repeat speedup >= 10%
perf_1gpu_6shots repeat speedup >= 5%
```

## 2026-06-08 - Stop Direct PML Fused VP Z-Face

Decision:

```text
Stop CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY in its direct p1 second-derivative form.
Do not repeat this route without new profiler evidence.
```

Implemented but rejected variants:

```text
1. Separate fused z-face kernel:
   v_pml skips fused-owned vx/vy writes; p_pml skips fused points; a separate zface kernel updates p0.

2. Inline p_pml fused branch:
   v_pml skips fused-owned vx/vy writes; p_pml_tile handles zface points directly without an extra kernel launch.
```

Correctness evidence:

```text
smoke: pass
correctness: pass
perf_1gpu_6shots repeat output compare: pass
max perf6 repeat rel L2: 6.358816e-07
```

Performance evidence on RTX 5090 same-session A/B:

```text
zmem mean WP:              2.434461s
separate zface mean WP:    2.660077s, speed ratio 0.915184x
inline zface mean WP:      2.692579s, speed ratio 0.904137x
```

Reason:

```text
The direct z-face fusion is mathematically valid but replaces vx/vy global reads with additional p1 loads, extra arithmetic, and branch pressure.
The saved vx/vy round trip is not enough to overcome this cost on the perf_1gpu_6shots gate.
```

Stop rule:

```text
Do not continue direct p1 x/y second-derivative z-face fusion.
Only reopen PML z-face fusion if the design keeps velocity intermediates CTA-local with shared-memory reuse or has Nsight Compute evidence showing lower total memory stalls.
```

Report:

```text
reports/wavestep_engine_v2_phase2_fused_zface_20260608_010000/phase2_fused_zface_report.md
```

## 2026-06-08 - Stop Shared-Tile PML Z-Face VP Prototype

Decision:

```text
Stop CUDA3D_PML_ZFACE_SHARED_VP_DEBUG in the tested S2/S4 forms.
Do not repeat p-only shared pressure tile or S4 staged-V shared velocity intermediate without a new source-level profiler reason.
```

Implemented variants:

```text
1. S2 p-only shared pressure tile
   output tile: 12x16x12
   shared p tile: 26x30x26
   shared memory: 81,120 bytes
   threads: 256

2. S4 p-only shared pressure tile
   output tile: 12x12x12
   shared p tile: 26x26x26
   shared memory: 70,304 bytes
   threads: 256

3. S4 staged-V shared velocity intermediate
   output tile: 12x12x12
   shared p + vx + vy memory: 92,192 bytes
   threads: 256
```

Correctness evidence:

```text
S2 p-only smoke/correctness/perf repeat output compare: pass
S4 p-only correctness/perf output compare: pass
S4 staged-V correctness/perf repeat output compare: pass
all compared output rel_l2: 0
```

Performance evidence on RTX 5090 same-session A/B:

```text
zmem mean WP:                  2.448577s
S2 p-only mean WP:             3.007605s, speed ratio 0.814129x
S4 p-only WP:                  3.039426s, speed ratio 0.805605x
S4 staged-V mean WP:           3.090552s, speed ratio 0.792278x

zmem mean Gradient:            2.560774s
S2 p-only mean Gradient:       3.169876s, speed ratio 0.807847x
S4 p-only Gradient:            3.188930s, speed ratio 0.803020x
S4 staged-V mean Gradient:     3.236345s, speed ratio 0.791255x
```

Reason:

```text
The shared z-face VP idea is numerically valid but not performance viable in the tested CTA shape.
The p-only version reuses global p1 through shared memory but repeatedly recomputes vx/vy per output.
The staged-V version reduces repeated vx/vy recompute, but its 92KB dynamic shared footprint, extra staging loops, synchronization, and 1 CTA/SM occupancy are still too expensive.
The pressure critical path becomes slower than the saved vx/vy global round trip.
```

Reports:

```text
docs/wavestep_v2/phase2_fused_zface_forensics.md
docs/wavestep_v2/pml_zface_shared_tile_budget.md
docs/wavestep_v2/pml_zface_shared_vp_design.md
reports/wavestep_v2_night_20260608/final_report.md
```

Next allowed route:

```text
Keep CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL as scaffold.
Move away from local z-face fusion and evaluate global-region temporal pipeline or PML compact-state audit.
```

## 2026-06-08 - Stop PML Compact-State Prototype

Decision:

```text
Do not implement CUDA3D_PML_COMPACT_STATE_DEBUG_MIRROR,
CUDA3D_PML_COMPACT_ZFACE_STATE, or related compact-state CUDA prototypes
from the current evidence.
```

Evidence:

```text
CPML double-buffer revalidation:
  perf_1gpu_6shots all-mean WP speedup       1.032329x
  perf_1gpu_6shots all-mean Gradient speedup 1.028370x
  correctness/perf output rel_l2             0

PML compact-state static audit:
  cpml_dbuf state footprint                  72.391 MiB
  six padded wavefield/cw2 array floor       503.039 MiB
  state footprint share vs six arrays        14.39%
  safe zface share of memory_dz              84.93%
  residual z edge/corner elements            602112
  estimated compact-state WP ceiling         1.005x

NCU short profile:
  zmem cuda_fd3d_p_pml_tile_ns               189.840 us
  cpml cuda_fd3d_p_pml_tile_ns               190.293 us
  zmem cuda_fd3d_v_pml_tile_ns                71.493 us
  cpml cuda_fd3d_v_pml_tile_ns                66.000 us
```

Reason:

```text
The current implementation already stores CPML state as axis slabs:
memory_dy, memory_dx, memory_dz, memory_dyy, memory_dxx, memory_dzz.
Therefore the obvious full-domain-to-PML compaction is already present.

A safe z-face compact layout can cover the pure/safe z-face region, but
edge and corner state still exists and must be updated for correctness.
The static upper bound is far below the >=5% meaningful prototype gate.

The measured CPML double-buffer gain comes from velocity PML ownership;
pressure PML timing is essentially unchanged, so pressure-side compact
state is not the active bottleneck.
```

Stop rule:

```text
Do not reopen compact-state storage unless a new profiler run shows CPML
state layout or state-sector traffic is the dominant pressure/velocity PML
bottleneck and the byte model predicts >=5% WP speedup.
```

Next allowed route:

```text
Keep CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL as ownership scaffold.
Move to global-region temporal pipeline design/prototype with a >=5%
meaningful-case gate.
```

## 2026-06-08 - Stop Direct K=2 Temporal CUDA Prototype

Decision:

```text
Do not implement a direct CUDA3D_WAVESTEP_ENGINE_V2_TEMPORAL_PIPELINE
K=2 fused kernel from the current model.
```

Evidence:

```text
K=2 deep-core share of pressure core              77.78%
current p_core bytes/output estimate             128.438
ideal K=2 local-reuse p_core pair reduction       35.25%
ideal K=2 sampled-main speedup upper bound        1.103x

safe global-middle design speedup estimate        ~1.0x
cooperative grid p_core blocks                    70688
conservative resident block capacity              1360
cooperative over-capacity factor                  51.98x
CTA-local candidate pair-byte ratio vs baseline   11.29x - 21.30x
```

Reason:

```text
The byte model confirms there is real upside if p(t+1) stencil values can
be reused locally for the second step.  However, the implementable safe
versions do not capture that upside:

1. Global p(t+1) middle state preserves correctness but keeps the second
   step global stencil traffic.
2. Simple cooperative grid-wide sync cannot cover the current p_core grid.
3. CTA-local p_mid reuse is the only no-duplication route with >5%
   upper-bound upside, but concrete local tiles fail after p_mid halo
   duplication is included.  The modeled candidates require 11.29x to
   21.30x baseline pair bytes per final output.
4. CTA-local p_mid reuse is also the already forbidden CTA-local two-step
   family unless it is redesigned as a source-aware swept/wavefront
   ownership algorithm.

Unresolved correctness hazards are source injection between substeps,
intermediate receiver extraction, p_mid halo ownership, shell/PML
reconciliation, and avoiding reads of half-updated values.
```

Stop rule:

```text
Do not write a direct K=2 temporal CUDA kernel until Phase 4.2 provides
a source-aware swept/wavefront design with an ownership proof and a byte
model predicting >=5% WP speedup after halo duplication.
```

Reports:

```text
docs/day_20260608/temporal_pipeline_model.md
docs/day_20260608/phase4_1_temporal_model_gate_decision.md
reports/day_20260608/phase4_1_temporal_model_gate_summary.json
```

## 2026-06-08 - Stop Source-Aware Swept/Wavefront Temporal Prototype

Decision:

```text
Do not implement a swept/wavefront K=2 temporal CUDA prototype from the
current source-aware model.
```

Evidence:

```text
aggregate shot-local K=2 deep-core share          73.22%
source influence overlaps K=2 deep core          0 shots
receiver footprint overlaps K=2 deep core        0 shots

Phase 4.1 still applies:
  safe global-middle design                       no meaningful byte saving
  cooperative grid over-capacity                  51.98x
  CTA-local pair-byte ratio vs baseline           11.29x - 21.30x
```

Reason:

```text
The benchmark source and receivers are shallow and do not overlap the K=2
deep-core temporal region.  This removes source/receiver placement as the
blocker, but it does not solve p(t+1) ownership.

No current schedule provides non-duplicating p_mid reuse without a grid-wide
sync or unsafe half-updated reads.  Direct CUDA code would therefore either
be equivalent to the safe global-middle path or fall back into the failed
CTA-local halo-duplication path.
```

Stop rule:

```text
Pause K=2 temporal work until a new p_mid ownership mechanism is proposed
with >=5% predicted WP speedup after halo duplication and an explicit
source/extract/PML schedule.
```

Next allowed route:

```text
Keep CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL as scaffold.
Move attention to pressure PML dataflow or wave-step scheduling around
cuda_fd3d_p_pml_tile_ns, currently the largest sampled kernel.
```

## 2026-06-08 - Open Pressure PML Z-Recompute Shared-Line Prototype

Decision:

```text
Open exactly one pressure PML CUDA prototype:
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE.
```

Evidence:

```text
Pressure PML dataflow audit:
  kept pressure-PML tiles                         113840 / 181232
  active thread efficiency                        65.60%
  shell active points                             4143640
  shell share of active points                    21.67%
  true-PML share of active points                 78.33%

Z recompute reuse budget:
  current recompute_vz_after_update calls         152951552
  shared z-line cache call estimate               29093740
  estimated call reduction                        80.98%
  current p1 loads inside z recompute             4667.711 MiB/step aggregate-shots
  shared-cache p1 load estimate                   887.870 MiB/step aggregate-shots

NCU-linked model:
  p_pml sampled-main share                        53.42%
  modeled p_pml speedup                           1.573x
  modeled sampled-main speedup                    1.242x
```

Reason:

```text
The dominant pressure PML kernel repeatedly computes the same
vz-after-update intermediate values along each CTA z-line.  A CTA-local
z-line cache can compute each needed z intermediate once per x/y line, then
let pressure threads consume cached neighbor values.

This route attacks repeated arithmetic/load work inside p_pml.  It does not
repeat the forbidden tile-mask fastpath, z-face specialization/fusion, or
RECOMPUTE_X/Y/XYZ routes.
```

Prototype constraints:

```text
The macro must default off.  memory_dz_next ownership must remain identical:
only tile-owned central z positions may write next z CPML memory.  x/y
velocity paths stay global-vx/vy based.

Before any performance claim, the candidate must pass debug dump step 0/1/2,
correctness, and perf_1gpu_6shots repeat against zmem_reference.
```

Stop rule:

```text
Stop this prototype if debug/correctness fails or if perf_1gpu_6shots repeat
does not show >=5% meaningful WP speedup.  Do not fall back to the forbidden
z-face/shared-tile/tile-mask fastpath families.
```

## 2026-06-08 - Accept CPML VMem + Pressure Z-Recompute Cache Combo

Decision:

```text
Keep CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE as a macro-default-off
pressure-PML prototype, and carry the combined candidate with Phase 1 CPML
velocity-memory double buffering as the current meaningful >=5% result.
```

Accepted candidate flags:

```text
CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
CUDA3D_CPML_VMEM_DISABLE_MPI
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

Evidence:

```text
Standalone z-cache:
  correctness rel L2                         0
  perf6 repeat mean WP speedup               1.044955x
  perf6 repeat mean Gradient speedup         1.045506x
  verdict                                    useful, but below standalone >=5% gate

Combined with CPML vmem scaffold:
  debug dump step 0/1/2                      pass
  correctness rel L2                         0
  perf6 repeat all output compares           pass
  perf6 repeat mean WP speedup               1.083390x
  perf6 repeat mean Gradient speedup         1.080857x
```

Reason:

```text
The z-cache prototype reduces repeated pressure-PML z intermediate
computation but is just under the standalone 5% gate.  It composes cleanly
with the already accepted Phase 1 velocity-memory ownership scaffold, and
the combination crosses the meaningful gate with zero output difference.
```

Rejected sub-route:

```text
Do not continue the pressure-PML vx/vy shared-neighbor cache attempted in
this sprint.  It passed correctness but slowed perf6 repeat to mean WP
speedup 0.419906x and mean Gradient speedup 0.426565x.
```

Next allowed route:

```text
Profile the combined candidate and look for the next dominant source of
pressure-PML latency.  Do not reopen shared vx/vy cache, tile-mask fastpath,
z-face specialize/fusion, or RECOMPUTE_X/Y/XYZ without new profiler evidence.
```

NCU follow-up:

```text
combo cuda_fd3d_p_core_ns duration          75.306us vs zmem 76.061us
combo cuda_fd3d_p_pml_tile_ns duration      142.902us vs zmem 158.291us
combo cuda_fd3d_v_pml_tile_ns duration       53.101us vs zmem 58.320us

combo p_pml eligible warps/scheduler          0.798
combo p_pml No Eligible                       60.879%
combo p_pml achieved occupancy                75.965%
```

Implication:

```text
The combined candidate's remaining pressure-PML bottleneck is issue/latency
overhead, not a simple raw-DRAM bandwidth limit.  The next pressure-PML
experiment may target z-cache fill integer/control overhead, but shared
vx/vy neighbor caching remains stopped.
```

## 2026-06-08 - Accept Direct-Fill Pressure Z-Cache

Decision:

```text
Replace the first linear-loop z-cache fill with direct fill inside
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE.
```

Evidence:

```text
linear-loop combo mean WP speedup              1.083390x
linear-loop combo mean Gradient speedup        1.080857x

direct-fill combo debug dump step 0/1/2        pass
direct-fill combo correctness rel L2           0
direct-fill combo perf6 output compares        pass
direct-fill combo mean WP speedup              1.100929x
direct-fill combo mean Gradient speedup        1.097530x
```

Reason:

```text
NCU showed the accepted combo was increasingly issue/latency limited in
p_pml_tile.  Direct fill removes the z-cache linear fill loop's division and
modulo indexing.  The algorithm and memory_dz_next ownership are unchanged.
```

Rejected boundary:

```text
Do not use shared vx/vy pressure-neighbor cache.  It remained removed after
direct-fill testing.
```

## 2026-06-08 - Reject Warp-Range Pressure Z-Cache

Decision:

```text
Reject CUDA3D_PML_PRESSURE_ZCACHE_WARP_RANGE and restore source to the
accepted direct-fill pressure z-cache implementation.
```

Evidence:

```text
correctness rel L2                         0 for 6 outputs
perf6 output compares                      pass
mean WP speedup vs direct-fill             0.997223x
mean Gradient speedup vs direct-fill       0.997502x
```

Reason:

```text
The candidate computed active z range once per 32-thread z-line and used
warp shuffle broadcast.  The reduced repeated branch work did not offset
shuffle/control overhead.  It is correctness-safe but performance-neutral
to slightly slower, so it fails the >=2% small-candidate gate.
```

Rejected boundary:

```text
Do not retry warp-broadcast active-range caching for pressure z-cache
without new profiler evidence.  Keep the direct-fill z-cache implementation
as current best.
```

## 2026-06-08 - Reject Local CPML New-Mem Accumulation

Decision:

```text
Reject the pml_local_mem_accum candidate that rewrites CPML memory updates
as explicit local new_mem values inside cuda_fd3d_p_pml_tile_ns.
```

Evidence:

```text
direct-fill SourceCounters:
  p_pml_tile No Eligible                 about 60%
  eligible warps/scheduler               about 0.81
  L1TEX scoreboard stall                 about 14.4 cycles/warp
  excessive global sectors               about 19%

pml_local_mem_accum:
  correctness rel L2                     0 for 6 outputs
  perf6 output compares                  pass
  mean WP speedup vs direct-fill         1.000647x
  mean Gradient speedup vs direct-fill   0.998957x
```

Reason:

```text
The source profile shows CPML memory update and final p0 writeback dominate,
but a syntactic local new_mem rewrite does not materially reduce the memory
dependency chain.  The compiler already preserves the value well enough, and
the candidate fails the >=2% small-candidate gate.
```

Rejected boundary:

```text
Do not retry plain local new_mem accumulation for p_pml_tile CPML updates.
Future pressure-PML work should target larger divergence or CPML traffic
structure, not this expression-level rewrite.
```

## 2026-06-08 - Reject Final P0 LDG Read

Decision:

```text
Reject pml_p0_ldg, which replaces the old p0[outIndex] read in the
cuda_fd3d_p_pml_tile_ns final pressure update with __ldg(p0+outIndex).
```

Evidence:

```text
pml_p0_ldg:
  correctness rel L2                     0 for 6 outputs
  perf6 output compares                  pass
  mean WP speedup vs direct-fill         1.000054x
  mean Gradient speedup vs direct-fill   1.000694x
```

Reason:

```text
SourceCounters marked the final p0 writeback/update line as hot, but making
the old p0 operand a read-only-cache load does not change the dominant memory
dependency enough to matter.  The result is correctness-safe but inside
measurement noise, so it fails the >=2% small-candidate gate.
```

Rejected boundary:

```text
Do not retry __ldg(p0+outIndex) for the pressure-PML final update without
new profiler evidence.  Future pressure-PML work should move to larger
region/dataflow restructuring.
```

## 2026-06-08 - Reject Z-Safe Direct Shared P1 Pressure-Z Path

Decision:

```text
Reject zsafe_direct_shared, a structural candidate that handles pressure-PML
tiles whose central z range is safely outside z-PML by loading a shared p1
z-line with +/-7 halo and computing the z second derivative directly.
```

Evidence:

```text
correctness rel L2 maximum               about 2.180533e-10
perf6 output compares                    pass
mean WP speedup vs direct-fill           0.966920x
mean Gradient speedup vs direct-fill     0.965779x
```

Reason:

```text
The design was attractive because middle-z x/y-PML and shell tiles do not
need z-CPML memory, so they can theoretically bypass recompute_vz.  In
practice the wider shared p1 halo, extra p1/shared loads, and changed
instruction mix are slower than the accepted direct-fill vz-line cache.
The candidate is correctness-safe but regresses by roughly 3.3%.
```

Rejected boundary:

```text
Do not retry z-safe shared-p1 direct second-derivative tiles in the current
32x4x2 pressure-PML shape.  Any future direct-z path must first show profiler
evidence that it reduces the pressure critical path without increasing shared
traffic and load latency.
```

## 2026-06-08 - Reject PTXAS DLCM Cache-Policy Sweep

Decision:

```text
Reject forcing direct-fill builds with -Xptxas -dlcm=ca or -Xptxas -dlcm=cg.
```

Evidence:

```text
-dlcm=ca:
  perf6 output compares                    pass
  mean WP speedup vs direct-fill           0.999263x
  mean Gradient speedup vs direct-fill     0.999576x

-dlcm=cg:
  perf6 output compares                    pass
  mean WP speedup vs direct-fill           0.859344x
  mean Gradient speedup vs direct-fill     0.864052x
```

Reason:

```text
SourceCounters showed L1TEX scoreboard stalls, but a global cache-policy
override is too blunt.  Cache-all is measurement-neutral to slightly slower,
while cache-global/bypass-L1 destroys useful locality in the accepted
direct-fill pressure-PML path.
```

Rejected boundary:

```text
Do not repeat ptxas dlcm cache-policy sweeps for the accepted direct-fill
candidate.  Future memory work needs source/dataflow changes or profiler
evidence for a narrower per-load policy, not a whole-binary cache override.
```

## 2026-06-08 - Reject P-Core Explicit Readonly LDG

Decision:

```text
Reject CUDA3D_P_CORE_READONLY_LDG, which explicitly changes p_core p1/cw2
loads to __ldg in cuda_fd3d_p_core_ns.
```

Evidence:

```text
correctness rel L2                         0 for 6 outputs
perf6 output compares                      pass
mean WP speedup vs direct-fill             0.999319x
mean Gradient speedup vs direct-fill       0.999254x
```

Reason:

```text
p_core is memory-throughput heavy, but explicit read-only loads do not
improve the current compiled path on RTX 5090.  The compiler/hardware cache
path is already adequate, and the candidate is measurement-neutral to
slightly slower.
```

Rejected boundary:

```text
Do not retry explicit __ldg wrapping for p_core p1/cw2 loads.  Future p_core
work needs a real data-reuse or temporal-ownership change, not only load
syntax changes.
```

## 2026-06-08 - Reject Inject/Extract BS512 Small-Kernel Candidate

Decision:

```text
Reject CUDA3D_INJECT_EXTRACT_BS512, which changes the
lint3d_inject_bell_extract_gpu_zz block size from 1024 to 512.
```

Evidence:

```text
NCU inject/extract duration                about 5.109us
NCU SOL compute                            0.040%
NCU SOL memory                             6.699%
NCU rule                                   grid too small, 0.0 full waves
correctness rel L2                         0 for 6 outputs
perf6 output compares                      pass
mean WP speedup vs direct-fill             0.999684x
mean Gradient speedup vs direct-fill       0.998963x
```

Reason:

```text
The inject/extract kernel is visibly small and launch/scheduling dominated,
but changing only its CUDA block size does not improve end-to-end repeat
performance.  This is not a math-kernel throughput problem in the current
form; it would require a broader CUDA Graph or wave-step scheduling design.
```

Rejected boundary:

```text
Do not retry inject/extract block-size-only changes.  Future scheduling work
must be framed as CUDA Graph / launch aggregation / wave-step orchestration
and must show a >=2% repeat gain before entering the main line.
```

## 2026-06-08 - Reject V-PML VX/VY Component Split Gate

Decision:

```text
Reject implementing separate vx and vy velocity-PML component kernels under
the current 32x4x2 PML tile geometry.
```

Evidence:

```text
v_pml SourceCounters:
  No Eligible                              44.891%
  Eligible warps/scheduler                 1.629
  Warp cycles/issued instruction           18.456
  Avg active threads/warp                  23.700
  Avg not-predicated threads/warp          21.670
  Branch efficiency                        86.970%
  L1TEX scoreboard stall                   about 11.8 cycles/warp
  Uncoalesced excessive sectors            about 22%

component split static budget:
  current combined vx/vy tiles             41,100
  vx-only tiles                            40,848
  vy-only tiles                            40,762
  split tile sum / combined tiles          1.985645x
  split active work sum / combined active  1.963726x
  overlap tiles                            40,510
```

Reason:

```text
The profiler shows real memory-latency and coalescing pressure in v_pml_tile,
but vx/vy ownership overlaps too heavily in the current tile geometry.  A
component-owner split would nearly double launches and active component work,
so it is rejected before writing CUDA code.
```

Rejected boundary:

```text
Do not implement vx/vy split kernels with the current 32x4x2 PML tile shape.
Future v_pml work must first change the memory layout/coalescing strategy or
show a new budget that avoids the near-2x component overlap.
```

## 2026-06-08 - Reject Current Single-GPU CUDA Graph Launch Gate

Decision:

```text
Reject a CUDA Graph / launch aggregation CUDA prototype for the current
single-GPU perf_1gpu_6shots loop.
```

Evidence:

```text
Nsight Systems run                         scheduling_nsys_20260608_142948
Gradient TIME all                          2.349826s
WP computing time                          2.238769s
GPU kernel total                           2.232398465s
WP minus GPU kernel total                  0.006370535s
visible non-kernel gap fraction            0.2846%
ideal speedup if gap vanished              1.002854x
cudaLaunchKernel CPU API total             1.845401s
cudaLaunchKernel calls                     36,024
```

Reason:

```text
The CPU API launch total is large, but it is mostly overlapped with GPU
kernel execution.  The WP timer almost equals the Nsight Systems GPU kernel
total, so a CUDA Graph implementation cannot satisfy the >=2% small-candidate
gate for the current single-rank loop.
```

Rejected boundary:

```text
Do not implement CUDA Graph / launch aggregation for the current single-GPU
loop unless future Nsight Systems or wall-clock multi-rank evidence shows
>2% visible scheduling gap or GPU idle time.
```

## 2026-06-08 - Gate PML Active Segment Compaction

Decision:

```text
Reject simple active-line list compaction, but keep length-16 half-warp
pressure-PML segment packing as the next design-only route.
```

Evidence:

```text
current pressure-PML launched lanes          29,143,040
active lanes after core return              19,118,944
current lane efficiency                     65.60%
active line slots                           893,204

active z-line length histogram:
  length 16                                 542,100 lines
  length 23                                  87,776 lines
  length 32                                 263,328 lines

simple active-line list sampled ceiling      1.011x
exact active-point list sampled ceiling      1.228x
length-16 half-warp sampled ceiling          1.207x
```

Reason:

```text
The ordinary line-list shape only removes empty lines, so it does not meet
the >=2% small-candidate gate.  The real lane-utilization signal is the large
length-16 z-face/margin population.  Packing two length-16 lines into one
warp has a meaningful model ceiling while preserving z-contiguous work.
```

Boundary:

```text
This does not reopen the rejected z-face direct-derivative/fusion/shared-VP
routes.  A CUDA prototype is allowed only if it preserves the accepted
direct-fill pressure z-cache math path and targets lane utilization/ownership,
not p1 x/y direct derivative substitution.
```

## 2026-06-08 - Accept Length-16 Half-Warp Pressure-PML Packing

Decision:

```text
Accept CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK as the current macro-gated
RTX 5090 single-GPU best candidate on top of the direct-fill pressure z-cache
path.
```

Implemented shape:

```text
Host splits pressure-PML tiles into residual tiles and whole length-16 active-z
tiles.

Residual tiles still run cuda_fd3d_p_pml_tile_ns.
Length-16 tiles run cuda_fd3d_p_pml_len16_halfwarp_ns.

One warp handles two length-16 z-lines:
  lanes 0..15   line A
  lanes 16..31  line B
CTA shape: 32x4x1
Required PML tile shape: 32x4x2
```

Evidence:

```text
smoke_1gpu                                 pass
debug dump profile_1gpu step 0/1/2         pass
correctness 6-output compare               pass
perf_1gpu_6shots repeat output compare     pass, max rel L2 6.384336e-07

mean base WP vs direct-fill                2.207751s
mean candidate WP                          2.039080s
mean WP speedup                            1.082719x

mean base Gradient                         2.316433s
mean candidate Gradient                     2.159948s
mean Gradient speedup                      1.072448x
```

Reason:

```text
The active-segment model correctly identified length-16 pressure-PML z-lines as
a meaningful lane-utilization problem.  Packing two such lines into one warp
reduces wasted lanes while preserving the accepted direct-fill z-cache dataflow.
The repeat gain clears the >=5% prototype gate.
```

Boundary:

```text
This acceptance does not reopen z-face direct derivative, z-face fusion, or
shared-VP routes.  Future work must keep the direct-fill pressure z-cache math
path unless new profiler evidence justifies a different dataflow.

The estimated product speedup versus zmem_reference is about 1.191983x, but this
is not a formal cumulative table until zmem/direct-fill/len16 are rerun in one
same-session comparison.
```

Report:

```text
docs/day_20260608/len16_halfwarp_pressure_pml_prototype.md
reports/day_20260608/len16_halfwarp_perf6_repeat_20260608_152944/summary.md
```

## 2026-06-08 - Confirm Len16 Profile and Reject Simple Len23 Prototype

Decision:

```text
Confirm CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK with same-worktree Nsight
Compute evidence.  Reject a simple length-23-only pressure-PML prototype.
```

Evidence:

```text
direct-fill pressure-PML duration              164.328us
len16 residual pressure-PML duration            72.683us
len16 packed pressure-PML duration              65.771us
len16 pressure-PML total                       138.453us
pressure-PML kernel-path speedup                 1.187x

sampled main-kernel total direct-fill          323.608us
sampled main-kernel total len16                297.248us
sampled main-kernel speedup                     1.0887x
```

Reason:

```text
The NCU result matches the perf repeat result and shows that len16 improves
pressure-PML ownership rather than merely moving time between kernels.

The remaining length-23 opportunity is much smaller: about 0.790M inactive
lanes, and it cannot pack two lines into one warp.  A simple length-23 kernel
would add a launch and split logic while still leaving one warp per line.
```

Boundary:

```text
Do not implement a simple CUDA3D_PML_PRESSURE_LEN23_* prototype.

Length-23 may be reopened only as part of an exact active-point / compact
descriptor design that demonstrates a >=5% perf_1gpu_6shots repeat speedup
ceiling before CUDA code.
```

Report:

```text
docs/day_20260608/len16_halfwarp_ncu_profile.md
reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.md
```

## 2026-06-08 - Reject Post-Len16 Compact Descriptor Prototype

Decision:

```text
Reject exact active-point / compact descriptor CUDA prototype after the accepted
len16 pressure-PML packing.
```

Evidence:

```text
accepted len16 lanes                         19,908,928
active lanes                                 19,118,944
remaining length-23 inactive lanes              789,984
post-len16 pressure-PML sampled-main share       46.58%

exact_length23_points_only sampled ceiling        1.0188x
exact_length23_points_only calibrated estimate    1.0153x
exact_length23_points_only descriptor traffic     7.701 MiB/step aggregate-shots

exact_all_active_points sampled ceiling           1.0188x
exact_all_active_points calibrated estimate       1.0153x
exact_all_active_points descriptor traffic        72.933 MiB/step aggregate-shots
```

Reason:

```text
The accepted len16 kernel already removes the large lane-waste population.
Only length-23 inactive lanes remain, and the optimistic sampled-main ceiling is
below the >=5% prototype gate before descriptor/control overhead.  The exact
all-point descriptor stream is especially unattractive because it adds about
72.933 MiB/step aggregate-shots of descriptor reads.
```

Boundary:

```text
Do not implement exact active-point, simple length-23, or compact descriptor
pressure-PML CUDA prototypes from the current post-len16 state.

This route may reopen only if a new descriptor/ownership design proves >=5%
perf_1gpu_6shots repeat speedup ceiling after descriptor/control overhead.
```

Next allowed routes:

```text
1. Source-level drill-down of cuda_fd3d_p_pml_len16_halfwarp_ns.
2. v-PML memory layout/coalescing design.
```

Report:

```text
docs/day_20260608/pml_compact_descriptor_budget.md
reports/day_20260608/pml_compact_descriptor_budget.json
```

## 2026-06-08 - Reject Len16 Source-Syntax Micro Prototypes

Decision:

```text
Do not implement len16-only p0 __ldg, explicit local new_mem, branch-only
lower/upper specialization, or z-cache/shared-memory micro prototypes.
```

Evidence:

```text
cuda_fd3d_p_pml_len16_halfwarp_ns SourceCounters:
  No Eligible                         73.545%
  Eligible warps/scheduler             0.427
  Warp cycles/issued instruction      33.970
  Branch efficiency                   65.220%
  L1TEX scoreboard stall              about 24.6 cycles/warp

source hot lines:
  final p0 update and cw2 load         about 60.78% parsed samples
  z-CPML mem_dzz update                about 26.82% parsed samples
  z-cache shared loads                 not dominant
```

Reason:

```text
The accepted len16 packed kernel is limited by final pressure writeback and
z-CPML memory dependency, not by z-cache fill.  The previously tested p0 __ldg
and local new_mem variants were noise-level on the direct-fill path, and this
profile does not provide a new reason to repeat them only inside len16.
Branch-only specialization also lacks a >=5% ceiling and would add tile-list and
launch overhead.
```

Boundary:

```text
Do not repeat len16 source-syntax micro tuning without new profiler evidence and
a modeled >=5% perf_1gpu_6shots repeat speedup ceiling.
```

Next allowed routes:

```text
1. v-PML memory layout / coalescing design.
2. A broader pressure-PML memory-ownership design that reduces final p0/cw2
   traffic or CPML z-state dependency with a proven >=5% ceiling.
```

Report:

```text
docs/day_20260608/len16_halfwarp_source_profile.md
reports/day_20260608/len16_source_profile_20260608_1646/source_hotlines.md
```

## 2026-06-08 - Reject V-PML Tile/Layout Prototype

Decision:

```text
Reject v-PML-only tile-layout CUDA prototype and do not run another random
PmlTileBlockSize sweep.
```

Evidence:

```text
accepted len16 sampled main                         297.248us
cuda_fd3d_v_pml_tile_ns                              65.248us
v-PML sampled-main share                              21.95%
v-kernel speedup required for 5% sampled-main gain     1.2770x

best reasoned shape: z8_x8_y4
launched lanes ratio vs current                        0.8830
warp z segments per warp                               4
optimistic v-kernel ceiling                             1.1325x
optimistic sampled-main ceiling                         1.0264x
```

Reason:

```text
The current 32x4x2 v-PML mapping uses threadIdx.x as z and gives each warp one
contiguous z-line at fixed x/y.  That is already the favorable coalescing shape
for p1, mem_dx, and mem_dy.

The only reasoned shape that reduces launched lanes enough to look interesting
is z8_x8_y4, but it splits each warp into four discontiguous x/y rows and still
only reaches a 2.64% optimistic sampled-main ceiling before separate velocity
tile-list plumbing, control overhead, and pressure-path compatibility costs.
```

Boundary:

```text
Do not implement v-only tile-layout CUDA prototypes below the 5% gate.
Do not repeat random PmlTileBlockSize sweeps.
Do not reopen current-geometry vx/vy component split.

V-PML layout may reopen only if a new model shows >=5% perf_1gpu_6shots repeat
speedup ceiling after tile-list/control overhead.
```

Next allowed routes:

```text
1. Broader memory-ownership design that reduces vx/vy global round trip without
   doubling component work.
2. Pressure-PML ownership/writeback design that targets final p0/cw2 traffic or
   CPML z-state dependency with a proven >=5% ceiling.
```

Report:

```text
docs/day_20260608/v_pml_coalescing_layout_budget.md
reports/day_20260608/v_pml_coalescing_layout_budget.json
```

## 2026-06-08 - Reject Two-Stream Wave-Step Overlap

Decision:

```text
Reject CUDA3D_WAVESTEP_ASYNC_STREAMS after smoke, correctness, and
perf_1gpu_6shots repeat.  Do not keep the prototype source in mainline.
```

Why it was opened:

```text
accepted len16 sampled main                      297.248us
p_core                                             93.547us
v_pml                                              65.248us
pressure residual                                  72.683us
pressure len16                                     65.771us

overlap p_core with serial PML path ceiling         1.4592x
required realized overlap for 1.05x sampled-main   15.13%
```

Tested prototype:

```text
stream_core: p_core
stream_pml:  v_pml -> p_pml_len16 -> p_pml_residual
default:     wait core+pml -> source injection/extraction
```

Evidence:

```text
smoke_1gpu pass
correctness pass, 6 outputs, max rel L2 0
perf_1gpu_6shots repeat compare pass in all 3 rounds

mean WP speedup        1.005183x
mean Gradient speedup  1.002855x
```

Reason:

```text
The dependency model was valid as an upper bound, but RTX 5090 resource
contention left almost no useful overlap between p_core and the PML path.
The measured gain is noise-level and far below the >=5% meaningful prototype
gate.
```

Boundary:

```text
Do not repeat the same two-stream p_core-vs-PML overlap prototype.
Do not use this result to justify CUDA Graph / launch aggregation in the
current single-GPU case.
Do not open three-stream pressure residual/len16 fanout unless Nsight Systems
shows real concurrent execution headroom and a new contention-aware model shows
>=5% perf_1gpu_6shots repeat speedup ceiling.
```

Report:

```text
docs/day_20260608/wavestep_stream_overlap_model.md
docs/day_20260608/wavestep_async_streams_prototype.md
reports/day_20260608/wavestep_async_perf6_repeat_20260608_175407/summary.md
```

## 2026-06-08 - Reject Pressure-PML Writeback/State Micro Prototype

Decision:

```text
Reject pressure-PML final writeback / CPML z-state micro CUDA prototypes after
the accepted len16 half-warp path.  Do not retry syntax/cache-policy tweaks on
the same hot lines.
```

Evidence:

```text
accepted len16 sampled main                         297.248us
len16 packed pressure-PML                            65.771us
len16 packed sampled-main share                       22.13%
total pressure-PML sampled-main share                 46.58%

parsed len16 source samples                           15,712
final p0/p1/cw2 update share                           60.78%
CPML mem_dzz update share                              26.82%
z-cache shared-load share                               1.92%
address/control visible-line share                      4.31%

packed-kernel speedup required for 1.05x sampled-main   1.2742x
final p0/p1/cw2 group speedup required if alone         1.5482x
CPML mem_dzz group speedup required if alone            5.0614x
final+mem_dzz group speedup required                    1.3257x
```

Reason:

```text
The source-hot lines are not a clear code-generation miss.  They are the
mathematically required second-order pressure update and the recursive CPML
z-state update.  Earlier concrete variants already failed:

p0 __ldg read syntax                 1.000054x WP vs direct-fill
local new_mem CPML accumulation      1.000647x WP vs direct-fill
ptxas -dlcm=ca                       0.999263x WP vs direct-fill
ptxas -dlcm=cg                       0.859344x WP vs direct-fill

A >=5% sampled-main gain from this area requires changing what traffic exists,
not changing how the same traffic is spelled.
```

Boundary:

```text
Do not retry len16 p0 __ldg, old-p0 read syntax, explicit local new_mem,
ptxas cache-policy, branch-only lower/upper specialization, or accepted
len16 z-cache fill/shared-cache micro-tuning without new profiler evidence and
a modeled >=5% perf_1gpu_6shots repeat ceiling.

Reopen only for a state-representation or time-integration design that proves
old-p0/cw2 or mem_dzz traffic is actually removed, not merely moved, and
accounts for extra storage/control costs.
```

Next allowed routes:

```text
1. Math-level pressure state representation design with equivalence proof.
2. PML vx/vy round-trip ownership redesign that reduces global traffic without
   doubling component work.
3. Formal same-session zmem/direct-fill/len16/current-best benchmark table
   before a larger phase switch.
```

Report:

```text
docs/day_20260608/pressure_pml_writeback_state_model.md
reports/day_20260608/pressure_pml_writeback_state_model.json
```

## 2026-06-08 - Formal Current-Best Same-Session Table

Decision:

```text
Freeze the current RTX 5090 single-GPU formal best as the accepted len16
half-warp pressure-PML path on top of direct-fill z-cache and CPML vmem
double-buffer scaffold.
```

Run protocol:

```text
remote worktree:
/work/wenzhe/cuda3D/.codex_worktrees/formal_table_20260608_182525

case:
perf_1gpu_6shots

rounds:
3

per round:
rebuild zmem -> run zmem
rebuild directfill -> run directfill -> compare outputs vs same-round zmem
rebuild len16_current_best -> run len16 -> compare outputs vs same-round zmem
```

Results:

```text
directfill vs zmem:
  mean WP speedup              1.099957x
  mean Gradient speedup        1.097977x
  mean elapsed speedup         1.105408x
  all compare pass             True
  max rel L2                   0

len16_current_best vs zmem:
  mean WP speedup              1.192835x
  mean Gradient speedup        1.179213x
  mean elapsed speedup         1.156108x
  mean candidate WP            2.031753s
  all compare pass             True
  max rel L2                   6.384336e-07
  max abs                      4.768372e-06
```

Boundary:

```text
The formal current-best speedup is 1.192835x vs zmem_reference on RTX 5090
single-GPU perf_1gpu_6shots.  This confirms the earlier estimated product
speedup, but it does not reach the 1.5x archive threshold.

Do not create an archives/speedups/1.5x_* milestone from this result.
```

Next:

```text
If continuing CUDA-core restructuring, start with a math-level pressure state
representation or PML ownership design gate.  Do not return to the rejected
micro routes unless new profiler evidence and a >=5% repeat ceiling are both
present.
```

Report:

```text
reports/day_20260608/formal_current_best_table_20260608_182525/summary.md
reports/day_20260608/formal_current_best_table_20260608_182525/summary.json
```

## 2026-06-08 - Reject Pressure State Representation Prototype

Decision:

```text
Reject pressure state representation CUDA prototypes under the current exact
numerical contract.  The state changes do not remove old-p/cw2 or mem_dzz
traffic without moving larger traffic into other hot kernels, adding another
state write, changing the scheme, or relaxing precision.
```

Evidence:

```text
sampled main                                  297.248us
p_core share                                   31.47%
v_pml share                                    21.95%
pressure-PML share                             46.58%
len16 packed pressure-PML share                22.13%
formal current-best WP speedup vs zmem          1.192835x

current second-order pressure update:
  p_prev read                                    4B
  p_cur read                                     4B
  cw2 read                                       4B
  p_next write                                   4B
  total                                         16B
```

Candidate outcomes:

```text
delta_pressure_state:
  exact recurrence algebra, but update traffic rises 16B -> 20B per point
  sampled-main effect if applied to all pressure updates: 0.8957x

scaled_pressure_q_only (q=p/cw2):
  final update can remove one cw2 load only if pressure stencils reconstruct p
  p_core alone needs >=29 pressure-value reconstructions per output
  p_core + v_pml share at risk: 53.42%

scaled_pressure_dual_p_and_q:
  avoids stencil reconstruction but update traffic rises 16B -> 32B

first_order_full_domain_velocity_pressure:
  not bitwise equivalent; saves 4B old-p read but adds >=24B velocity
  read/write state traffic per core point

precomputed_cw2dt:
  removes a multiply, not the cw2 global load

half_or_compressed_cw2:
  outside current exactness contract; ideal len16 cw2-line ceiling only 1.0282x

cpml_mem_dzz_rescaled_state:
  algebraic rescaling does not remove recursive state read/write
  mem_dzz alone needs 5.0614x local speedup to touch the gate
```

Boundary:

```text
Do not implement q=p/cw2, delta pressure state, dual p/q state, first-order
full-domain velocity-pressure rewrite, precomputed cw2dt, compressed cw2, or
CPML mem_dzz rescaling prototypes under the current exactness gate.

These are not merely micro-tuning rejects; they are math/state representation
rejects for the current variable-cw2 stencil path.
```

Next:

```text
Move to PML vx/vy round-trip ownership design with a >=5% model before CUDA.
Source-aware multi-step/wavefront may reopen only if synchronization and halo
ownership are solved first.  Precision relaxation requires an explicit new
tolerance policy.
```

Report:

```text
docs/day_20260608/pressure_state_representation_model.md
reports/day_20260608/pressure_state_representation_model.json
```

## 2026-06-08 - Reject PML Vx/Vy Round-Trip Ownership Prototype

Decision:

```text
Reject ordinary CUDA PML vx/vy round-trip ownership prototypes under the
current CTA-local or macro-tile shared-cache designs.
```

Evidence:

```text
sampled main                         297.248us
p_core                                93.547us
v_pml                                 65.248us
pressure-PML total                   138.453us
formal current-best WP vs zmem          1.192835x

generous savable-time model:
  len16 unknown/unparsed time assigned to vx/vy      4.056us
  residual pressure-PML generous 20% savable        14.537us
  total generous vx/vy round-trip savable           18.593us

duplicate velocity/CPML work factor allowed for 1.05x sampled-main:
  <= 1.068
```

Candidate outcomes:

```text
current pressure tile 4x2:
  duplicate velocity work        4.085x
  sampled-main speedup           0.6193x

macro 8x4:
  duplicate velocity work        2.606x
  sampled-main speedup           0.7752x

macro 16x8:
  duplicate velocity work        1.866x
  sampled-main speedup           0.8868x

macro 16x16:
  duplicate velocity work        1.620x
  shared velocity cache          94,208B
  sampled-main speedup           0.9315x

macro 32x8 / 32x16:
  duplicate velocity work        1.743x / 1.497x
  shared velocity cache          101,376B / 174,080B
  exceeds conservative 96KiB shared-memory limit

ideal no-duplicate cross-CTA owner:
  duplicate velocity work        1.000x
  sampled-main ceiling           1.0667x
  ordinary CUDA implementation   not available
```

Reason:

```text
The only route that reaches a meaningful ceiling requires each velocity value
to be computed once and consumed by neighboring pressure CTAs without passing
through global memory.  Ordinary CUDA cannot exchange register/shared values
across CTAs and the earlier global synchronization / cooperative-grid path is
not viable for this grid shape.

All implementable CTA-local or macro-tile shared-cache candidates duplicate too
much velocity and CPML state work before they can recover the global vx/vy
round-trip cost.
```

Boundary:

```text
Do not implement CTA-local vx/vy shared-cache fusion under current tile or
macro-tile ownership.
Do not reopen RECOMPUTE_X/Y/XYZ or direct p1 x/y derivative replacement.
Do not reopen current-geometry vx/vy component-owner split.
Do not write ordinary CUDA producer-consumer vx/vy fusion that relies on
cross-CTA shared values.
```

Next:

```text
Source-aware multi-step/wavefront design may reopen only after synchronization
and halo ownership are proven.  Precision relaxation requires an explicit new
tolerance policy.  If exact CUDA-core routes remain gated off, consider
application-level multi-shot batching.
```

Report:

```text
docs/day_20260608/pml_vxvy_roundtrip_ownership_model.md
reports/day_20260608/pml_vxvy_roundtrip_ownership_model.json
```

## 2026-06-08 - Reject Source-Aware Wavefront Synchronization Prototype

Decision:

```text
Reject source-aware K=2 wavefront temporal CUDA prototypes under the current
exact CUDA-core contract.
```

Evidence:

```text
sampled main                                      297.248us
p_core                                             93.547us
p_core sampled-main share                          31.47%
formal current-best WP speedup vs zmem              1.192835x

ideal K=2 p_core pair reduction                    35.25%
ideal K=2 sampled-main speedup on current best       1.1248x
p_core reduction required for 1.05x sampled-main    15.13%
fraction of ideal saving required                  42.92%

aggregate K=2 deep-core share                      73.22%
source overlap shots                                   0
receiver overlap shots                                 0

p_core grid blocks                                 70688
conservative resident block capacity                1360
cooperative-grid over-capacity factor               51.98x
```

Candidate outcomes:

```text
safe_global_middle_two_kernel:
  ordinary CUDA and safe, but materializes p(t+1) globally and reloads its
  stencil for the second step; speedup ceiling 1.0000x

cooperative_grid_full_core_k2:
  ideal speedup ceiling 1.1248x, but the full p_core grid exceeds conservative
  resident capacity by about 52x

cta_local_diamond_k2:
  ordinary CUDA, but concrete candidates require 11.29x to 21.30x baseline pair
  bytes after p_mid halo duplication

multi_kernel_global_wavefront:
  safe, but still materializes global p_mid between layers and adds many small
  wavefront launches; speedup ceiling 1.0000x

persistent_wavefront_without_global_barrier:
  requires cross-CTA shared/register ownership that ordinary CUDA does not
  provide

ideal_no_dup_source_aware_wavefront:
  meaningful ceiling, but no ordinary CUDA implementation without a concrete
  cross-CTA ownership primitive
```

Reason:

```text
Source and receiver placement are compatible with K=2 deep-core temporal
blocking for the current case.  The blocker is not physics-side injection or
extraction; it is synchronization and p_mid ownership.

Every ordinary CUDA schedule either keeps the global p_mid traffic, duplicates
p_mid halos by an order of magnitude, or requires a grid-wide / cross-CTA
ownership primitive that the current launch shape cannot use.
```

Boundary:

```text
Do not implement ordinary CUDA K=2 source-aware wavefront prototypes.
Do not implement multi-kernel global-middle wavefront prototypes.
Do not implement CTA-local diamond temporal prototypes.
Do not implement persistent-kernel wavefronts that rely on cross-CTA
shared/register values.
```

Next:

```text
Exact CUDA-core structural routes tested today are now mostly gated off.  Future
speedup work should either move to application-level multi-shot batching /
scheduling, or reopen precision relaxation only after the user explicitly
changes the tolerance policy.  No-duplication wavefront temporal blocking may
reopen only if a concrete hardware/runtime cross-CTA ownership primitive is
identified first.
```

Report:

```text
docs/day_20260608/source_aware_wavefront_sync_model.md
reports/day_20260608/source_aware_wavefront_sync_model.json
```

## 2026-06-08 - Reject Same-GPU Multi-Rank Scheduling

Decision:

```text
Reject same-GPU multi-rank oversubscription for the current RTX 5090
perf_1gpu_6shots case.
```

Evidence:

```text
current binary      current-best len16 + direct-fill z-cache + CPML vmem flags
GPU                 one RTX 5090
CUDA_VISIBLE_DEVICES=0 for all runs
case                perf_1gpu_6shots

np=1:
  elapsed                         2.990s
  Gradient TIME all               2.165543s
  outputs                         6

np=2:
  elapsed                         3.370s
  Gradient TIME all               2.311468s
  elapsed speedup vs np=1         0.8872x
  Gradient speedup vs np=1        0.9369x
  correctness vs np=1             pass, max rel L2 0

np=3:
  elapsed                         3.250s
  Gradient TIME all               2.328266s
  elapsed speedup vs np=1         0.9200x
  Gradient speedup vs np=1        0.9301x
  correctness vs np=1             pass, max rel L2 0
```

Reason:

```text
Multiple MPI ranks sharing one GPU correctly split shots, but they oversubscribe
the same device and slow down elapsed / Gradient TIME by roughly 7-11% in the
single-round probe.

For multi-rank scheduling, the printed WP computing time is root-rank local and
cannot be used as the formal wall-clock speed metric.  Use elapsed and Gradient
TIME all instead.
```

Boundary:

```text
Do not pursue same-GPU np=2/3 oversubscription repeat benchmarks for this case.
Do not claim multi-rank scheduling speedup from root-rank printed WP time.
Do not archive same-GPU multi-rank runs as speed threshold versions.
```

Next:

```text
If application-level scheduling continues, move to true multi-GPU / multi-job
batching where each rank or job owns a different GPU.  Any such result must
report elapsed, Gradient TIME all, correctness, GPU count, rank count, and shot
assignment.
```

Report:

```text
reports/day_20260608/multirank_samegpu_sched_20260608_193042/summary.md
reports/day_20260608/multirank_samegpu_sched_20260608_193042/summary.json
```

## 2026-06-08 - Defer True Multi-GPU Batching on Single-GPU Platform

Decision:

```text
Defer true multi-GPU / multi-job batching validation on the current RTX 5090
server because it exposes only one GPU.
```

Evidence:

```text
nvidia-smi -L:
  GPU 0: NVIDIA GeForce RTX 5090

current-best anchor:
  mean elapsed                  2.970s
  mean Gradient TIME all        2.155902s
  mean WP                       2.031753s
  WP speedup vs zmem            1.192835x

same-GPU oversubscription:
  np=2 elapsed speedup          0.8872x
  np=3 elapsed speedup          0.9200x
  decision                      rejected
```

Existing code requirements:

```text
src/main.cu reads gpus_p_node from the input file.
GPU mapping is cudaSetDevice(mytid % gpus_p_node).
Shot assignment is sht_num[is * ntids + mytid].

Therefore, true one-rank-per-GPU runs require all three to agree:
  mpirun -np N
  CUDA_VISIBLE_DEVICES exposes N devices
  input file last line gpus_p_node=N
```

Shot-balance upper bound for perf_1gpu_6shots:

```text
1 GPU   [6]             ideal 1.0000x
2 GPUs  [3, 3]          ideal 2.0000x
3 GPUs  [2, 2, 2]       ideal 3.0000x
4 GPUs  [2, 2, 1, 1]    ideal 3.0000x
6 GPUs  [1, 1, 1, 1, 1, 1] ideal 6.0000x
```

Boundary:

```text
Do not treat run_benchmark.py --gpus as a complete true multi-GPU setup.  It
sets CUDA_VISIBLE_DEVICES, but the program's device mapping still uses the
input file's gpus_p_node.

Do not run more same-GPU multi-rank oversubscription probes for this case.
Do not claim multi-GPU speedup from root-rank printed WP time.
```

Next:

```text
When a >=2 GPU platform is available, create input variants with last line
gpus_p_node=N, run np=N with N visible GPUs, compare outputs to the np=1
current-best baseline, and report elapsed plus Gradient TIME all over a
3-round repeat.
```

Report:

```text
docs/day_20260608/true_multigpu_batching_protocol.md
reports/day_20260608/true_multigpu_batching_protocol.json
```

## 2026-06-08 - Gate Host / Setup Overhead Optimization

Decision:

```text
Do not make blind host/setup optimization prototypes.  Profile or add targeted
timers first.
```

Evidence:

```text
current best                      len16_current_best
mean elapsed                      2.970s
mean Gradient TIME all            2.155902s
mean WP                           2.031753s

elapsed - Gradient                0.814098s / 27.41%
elapsed - WP                      0.938247s / 31.59%

current-best speedup vs zmem:
  elapsed                         1.1560x
  Gradient                        1.1792x
  WP                              1.1928x

time required for 1.05x elapsed speedup vs current best:
  saved time                      0.141429s
  fraction of elapsed-Gradient    17.37%
```

Scenario model:

```text
remove 10% of elapsed-Gradient:
  elapsed speedup vs current best 1.0282x

remove 25% of elapsed-Gradient:
  elapsed speedup vs current best 1.0736x

remove 50% of elapsed-Gradient:
  elapsed speedup vs current best 1.1588x

remove 100% of elapsed-Gradient:
  elapsed speedup vs current best 1.3776x
```

Reason:

```text
The elapsed-vs-Gradient gap is large enough to be worth profiling, but it is
not yet localized.  It may include MPI process startup, input parsing, velocity
model read, acquisition setup, broadcasts, allocations, CUDA context setup, and
finalization.  Those require different fixes and some are outside the CUDA
solver.
```

Boundary:

```text
Do not move timing markers and call it a speedup.
Do not skip output generation or correctness work.
Do not optimize run_benchmark.py output copying for this elapsed metric, because
copy_outputs runs after the timed command.
```

Next:

```text
Reopen host/setup optimization only after Nsight Systems, CPU sampling, or
targeted timers identify a concrete hotspot with >=5% elapsed-speedup ceiling.
```

Report:

```text
docs/day_20260608/host_setup_overhead_gate.md
reports/day_20260608/host_setup_overhead_gate.json
```

## 2026-06-08 - Profile Host Setup With Default-Off Timers

Decision:

```text
Keep CUDA3D_HOST_SETUP_TIMERS as a default-off diagnostic path.  Do not optimize
host/setup yet; the largest unexplained wall-clock gap is outside the current
in-program timer window.
```

Evidence:

```text
timer binary flags:
  current-best flags + -DCUDA3D_HOST_SETUP_TIMERS

correctness:
  timer binary vs formal len16 current-best r1
  pass, max rel L2 0, max abs 0

program timing:
  elapsed                         2.980s
  Gradient TIME all               2.162907s
  WP computing time               2.046621s
  elapsed - Gradient              0.817093s

measured in-program pre-Gradient setup:
  total                           0.238399s
  main total_pre_gradient         0.215846s
  cal pre_gradient_init           0.022553s

largest measured stages:
  gpu_setup                       0.174303s
  cal pre_gradient_init           0.022553s
  shot_list                       0.022419s
  root_model_read                 0.018118s

unaccounted elapsed-minus-Gradient:
  0.578694s
```

Reason:

```text
The measured in-program setup is real, but the dominant remaining gap is outside
the after-MPI timer window.  It likely includes bash / oneAPI source overhead,
mpirun process launch, MPI_Init, and finalization.  Optimizing CUDA solver code
cannot honestly claim that entire wall-clock gap.

The largest measured in-program item, gpu_setup, is likely CUDA device/context
initialization.  Moving or warming it outside the measured region would be a
benchmarking policy change, not a CUDA kernel speedup.
```

Boundary:

```text
CUDA3D_HOST_SETUP_TIMERS must remain default-off.
Do not use timer-marker movement or context warmup outside the measured command
as an optimization result.
Do not count bash/mpirun/MPI_Init savings as CUDA-core speedup.
```

Next:

```text
If wall-clock route continues, add a process-level timer around MPI_Init and/or
run Nsight Systems OS/runtime profiling to split mpirun/source/MPI_Init/finalize.
CUDA-core optimization should continue to use Gradient TIME all and WP as the
primary metrics.
```

Report:

```text
reports/day_20260608/host_setup_timer_probe_20260608_203508/summary.md
reports/day_20260608/host_setup_timer_probe_20260608_203508/summary.json
```

## 2026-06-08 - Close Process-Level Wall-Clock Gap

Decision:

```text
Stop host/setup wall-clock micro-optimization for CUDA-core speedup accounting.
The elapsed-vs-Gradient gap is now explained by process wrapper, MPI/context
startup, and measured pre-Gradient setup.
```

Evidence:

```text
timer binary flags:
  current-best flags + -DCUDA3D_HOST_SETUP_TIMERS

correctness:
  process timer binary vs formal len16 current-best r1
  pass, max rel L2 0, max abs 0

program timing:
  elapsed                         3.220s
  Gradient TIME all               2.161705s
  WP computing time               2.045140s
  elapsed - Gradient              1.058295s

process timers:
  MPI_Init                        0.254292s
  main after MPI to pre-finalize  2.418194s
  MPI_Finalize                    0.000283s
  process total                   2.672769s
  elapsed - process total         0.547231s

in-program timers:
  measured pre-Gradient setup     0.250119s
  gpu_setup                       0.186226s
  root_model_read                 0.018050s
  shot_list                       0.022546s
  cal pre_gradient_init           0.022299s

accounting:
  known non-Gradient time         1.053080s
  residual after known timers     0.005215s
```

Reason:

```text
The wall-clock overhead is no longer a mysterious CUDA solver hotspot.  The
largest pieces are external command / source / mpirun wrapper time, MPI_Init,
and CUDA device/context setup.  These are valid deployment or benchmarking
policy topics, but they are not CUDA kernel speedups.
```

Boundary:

```text
Do not pursue host/setup micro-optimizations as CUDA-core performance wins.
Do not use context warmup, moved timers, skipped setup, or skipped output as a
speedup claim.
Do not compare wall-clock runs unless command wrapper, environment source, MPI
rank count, GPU count, and input gpus_p_node are fixed.
```

Next:

```text
Use Gradient TIME all and WP for CUDA-core optimization.  Reopen wall-clock
work only for true multi-GPU/multi-job batching, long-running service mode, or a
clearly defined deployment benchmark where process startup is part of the
objective.
```

Report:

```text
reports/day_20260608/process_timer_probe_20260608_205311/summary.md
reports/day_20260608/process_timer_probe_20260608_205311/summary.json
```

## 2026-06-08 - Reject Cal-Loop Host Micro Optimizations

Decision:

```text
Do not implement host-side cal-loop micro prototypes for vc/vc_pad preparation,
output writing, cleanup, or copy/reduce under the current perf_1gpu_6shots gate.
```

Evidence:

```text
timer binary flags:
  current-best flags + -DCUDA3D_HOST_SETUP_TIMERS

correctness:
  cal-loop timer binary vs formal len16 current-best r1
  pass, max rel L2 0, max abs 0

program timing:
  elapsed                         2.990s
  Gradient TIME all               2.164033s
  WP computing time               2.044622s

cal pre_gradient_init             0.023527s

cal_loop across 6 shots:
  obs_setup                       0.002679s
  domain_setup                    0.000010s
  wavefield_prep                  0.049816s
  fd_call                         2.089376s
  output_write                    0.004491s
  cleanup                         0.002593s
  copy_reduce                     0.015053s
```

Reason:

```text
The largest non-FD cal-loop item is wavefield_prep at about 0.049816s.  Even an
unrealistic perfect removal gives only about 2.4% Gradient speedup, below the
>=5% prototype gate.  Output write, cleanup, obs setup, and copy/reduce are even
smaller.
```

Boundary:

```text
Do not write host-side vc/vc_pad preparation micro prototypes.
Do not write output write / cleanup / copy-reduce micro prototypes.
Do not treat removing diagnostic or output work as CUDA-core speedup.
```

Next:

```text
If exact compute optimization continues, focus on fd_3d_f kernel/dataflow
instead of host/pre-FD loop overhead.  Otherwise wait for true multi-GPU
batching validation on a platform with >=2 GPUs.
```

Report:

```text
reports/day_20260608/cal_loop_timer_probe_20260608_212019/summary.md
reports/day_20260608/cal_loop_timer_probe_20260608_212019/summary.json
```

## 2026-06-08 - Reject Pressure-PML Len32 Full-Warp Branch-Only Split

Decision:

```text
Do not implement CUDA3D_PML_PRESSURE_LEN32_FULL_WARP_SPECIALIZE as a
branch/control-only residual pressure-PML split after the accepted len16
half-warp packing.
```

Evidence:

```text
model:
  tools/pml_len32_fullwarp_specialization_budget.py

inputs:
  reports/day_20260608/pml_compact_descriptor_budget.json
  reports/day_20260608/len16_vs_directfill_ncu_20260608_1600/summary.json

timing anchor:
  sampled main                   297.248us
  residual pressure-PML           72.683us
  packed len16 pressure-PML       65.771us

residual shape after len16:
  length-23 lines                 87,776
  length-32 lines                263,328
  length-32 line share            75.00%
  length-32 active-lane share     80.67%

required for >=1.05x sampled-main:
  full32 local speedup            1.3182x to 1.3507x
  full32 local time reduction     24.14% to 25.97%

scenario ceilings:
  control proxy on full32         1.0080x sampled-main
  control proxy on residual       1.0107x sampled-main
  perfect branch on full32        1.0340x sampled-main
  perfect branch on residual      1.0425x sampled-main
  20% full32 local speedup        1.0411x sampled-main
```

Reason:

```text
Length-32 residual work has no inactive-lane saving after len16 packing.  A
new full-warp kernel would still perform the same pressure update, vx/vy loads,
z-cache math, and CPML state work, while adding another tile list and launch.
The measurable branch/control opportunity is too small for the >=5% prototype
gate.
```

Boundary:

```text
Do not write branch/control-only length-32 residual pressure-PML kernels.
Do not treat full-active length-32 lines as another lane-compaction opportunity.
This does not reject future length-32 work that removes real memory traffic or
proves a different ownership model with source-level profiler evidence.
```

Next:

```text
Continue only with designs that remove real memory traffic / state ownership
costs in fd_3d_f, or defer to true multi-GPU batching on a multi-GPU platform.
```

Report:

```text
docs/day_20260608/pml_len32_fullwarp_specialization_budget.md
reports/day_20260608/pml_len32_fullwarp_specialization_budget.json
```

## 2026-06-08 - Reject Current P-Core ZX Shared-Plane Prototype

Decision:

```text
Reject the current CUDA3D_P_CORE_SHARED_ZX_PLANE prototype and do not keep the
slow kernel in source.
```

Budget evidence:

```text
model:
  tools/p_core_shared_plane_budget.py

anchor:
  sampled main                    297.248us
  p_core                           93.547us
  p_core sampled-main share        31.47%
  p_core L2 SOL                    96.89%

current p_core p1 global floats/output:
  29.109375

best modeled candidate:
  shape                            16x16x1
  mode                             z+x shared plane, y global
  p1 floats/output                 17.516
  p_core byte ceiling              1.5651x
  sampled-main ceiling             1.1282x

budget decision:
  allow_cuda_prototype
```

Prototype evidence:

```text
worktree:
  /work/wenzhe/cuda3D/.codex_worktrees/p_core_zx_20260608_2158

binary sha256:
  45213389d52df56c9ab433f2bb48b72517d3c301555f32a6bde7c16d172602fe

smoke:
  pass

correctness:
  pass, 6 outputs rel L2 all 0

perf_1gpu_6shots repeat:
  round 1 WP 2.589493s, Gradient 2.731454s, compare pass
  round 2 WP 2.597138s, Gradient 2.734513s, compare pass
  round 3 WP 2.583236s, Gradient 2.727773s, compare pass
  mean WP                         2.589956s
  mean Gradient                   2.731247s
  WP speedup vs current-best      0.784474x
  Gradient speedup vs current-best 0.789347x
```

Reason:

```text
The byte model correctly identified possible p1 global-load reduction, but it
underestimated shared tile fill, control overhead, and warp/coalescing changes.
The prototype is numerically exact but a large slowdown, so this p-core shared
plane shape is not a valid optimization path.
```

Boundary:

```text
Do not continue the current 16x16x1 z+x shared-plane p_core kernel.
Do not use p-core p1 byte reduction alone as sufficient evidence for a CUDA
prototype.
Reopen p-core shared-plane work only if a new warp/coalescing design proves
lower shared-fill/control overhead with profiler/source evidence before coding.
```

Report:

```text
docs/day_20260608/p_core_shared_plane_budget.md
reports/day_20260608/p_core_shared_plane_budget.json
reports/day_20260608/p_core_zx_prototype_20260608_2158/summary.md
reports/day_20260608/p_core_zx_prototype_20260608_2158/perf6_repeat_summary.json
```

## 2026-06-08 - Reject Current P-Core Shared-Plane Shape Family

Decision:

```text
Reject the current p-core shared-plane shape family after calibrating the byte
model with the failed 16x16x1 z+x prototype.
```

Evidence:

```text
calibration tool:
  tools/p_core_shared_plane_calibrated_gate.py

inputs:
  reports/day_20260608/p_core_shared_plane_budget.json
  reports/day_20260608/p_core_zx_prototype_20260608_2158/perf6_repeat_summary.json

anchor:
  tested shape                    [16,16,1]
  tested mode                     zx_shared_y_global
  modeled p_core local speedup    1.5651x
  modeled sampled-main speedup    1.1282x
  observed WP global speedup      0.7845x
  observed Gradient global speedup 0.7893x
  inferred WP-local p_core        0.5339x
  inferred Gradient-local p_core  0.5411x
  WP model-to-observed factor     0.3411x
  Gradient model-to-observed      0.3457x

calibrated candidate examples:
  [16,16,1] zx shared             0.7845x WP sampled
  [32,8,1] zx shared              0.7768x WP sampled
  [64,2,2] zx shared              0.6878x WP sampled
  [64,2,2] zy shared              0.6878x WP sampled
```

Reason:

```text
The failed prototype provides direct evidence that shared-fill, synchronization,
control, and warp/coalescing overhead dominate the theoretical p1 traffic
savings.  Other current shared-plane shapes share the same overhead class and
fall below the >=5% gate after calibration.
```

Boundary:

```text
Do not test more variants from the current p-core shared-plane shape family
([32,8,1], [16,8,2], [64,2,2], etc.) merely because the uncalibrated byte model
looks positive.

Reopen only with a materially different warp/coalescing design and a model that
separately accounts for shared fill, synchronization, and control overhead.
```

Report:

```text
docs/day_20260608/p_core_shared_plane_calibrated_gate.md
reports/day_20260608/p_core_shared_plane_calibrated_gate.json
```

## 2026-06-08 - Keep V-PML Len16 Half-Warp Minor Candidate

Decision:

```text
Keep CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK as a macro-default-off minor
candidate, but do not expand the v-PML active-segment descriptor family.
```

Model evidence:

```text
tool:
  tools/v_pml_active_segment_packing_model.py

current v-PML:
  sampled-main share                 21.95%
  speedup required for 5% sampled    1.2770x local v-kernel
  current launched lanes             30,420,992
  true vx/vy active-any lanes        20,646,925
  length-16 z-line slots             506,974
  whole length-16 tiles              62,400

whole-tile len16 candidate:
  lane reduction                     26.26%
  v lane speedup ceiling             1.3560x
  sampled-main ceiling               1.0612x
  gate                               allow_cuda_prototype
```

Prototype evidence:

```text
macro:
  CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK

worktree:
  /work/wenzhe/cuda3D/.codex_worktrees/v_pml_len16_20260608_2238

build:
  pass

smoke:
  pass, outputs=3
  len16_tiles=0, wiring-only coverage

correctness:
  pass
  len16_tiles=0, wiring-only coverage

perf_1gpu_6shots repeat:
  all three output compares pass
  max rel L2                         0
  mean base WP                       2.052228s
  mean candidate WP                  1.988482s
  WP speedup                         1.032058x
  mean base Gradient                 2.169915s
  mean candidate Gradient            2.109314s
  Gradient speedup                   1.028730x
```

Reason:

```text
The route is numerically exact and repeat-stable, and it clears the >=2% minor
candidate line.  It does not clear the strict >=5% breakthrough gate, so it is
not a new structural phase breakthrough and should not be expanded into
descriptor variants without a fresh overhead model.
```

Boundary:

```text
Do not continue from this result into v-PML line-descriptor len16 packing,
exact active-point descriptors, random velocity tile-shape sweeps, or
current-geometry vx/vy component-owner split.

Reopen descriptor packing only if descriptor traffic, control overhead, launch
overhead, and source-level memory behavior are modeled and still predict >=5%
perf_1gpu_6shots repeat speedup after overhead.
```

Report:

```text
docs/day_20260608/v_pml_active_segment_packing_model.md
docs/day_20260608/v_pml_len16_halfwarp_prototype.md
reports/day_20260608/v_pml_active_segment_packing_model.json
reports/day_20260608/v_pml_len16_prototype_20260608_2238/summary.md
reports/day_20260608/v_pml_len16_prototype_20260608_2238/perf6_retry_summary.json
```
