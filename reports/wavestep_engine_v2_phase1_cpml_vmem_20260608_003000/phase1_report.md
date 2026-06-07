# WAVESTEP Engine V2 Phase1 CPML VMEM Double Buffer

## Result

- Gate: `continue`
- Phase1 perf6 mean WP: `2.365721s`
- ZMEM pre mean WP: `2.450038s`, speedup `1.035641x`
- ZMEM post mean WP: `2.435677s`, speedup `1.029570x`
- ZMEM all mean WP: `2.442857s`, speedup `1.032605x`
- ZMEM all mean Gradient: `2.555540s`, phase1 `2.484369s`, speedup `1.028648x`

## Correctness

- debug fill smoke/correctness: pass
- debug dump step 0/1/2: pass
- release smoke/correctness/perf6/perf6_repeat output comparisons: pass
- rel L2 tolerance: `<= 1e-5`

## Decision

Phase1 passes the slowdown gate and is mildly faster in this A/B window. Keep it macro-gated and default-off as the ownership-clean scaffold for PML fused VP.
