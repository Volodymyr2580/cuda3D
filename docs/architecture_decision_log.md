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
