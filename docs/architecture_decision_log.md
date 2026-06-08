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
