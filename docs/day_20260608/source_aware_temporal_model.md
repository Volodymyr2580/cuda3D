# Source-Aware Swept/Wavefront Temporal Model

## Case

- case_dir: `/work/wenzhe/cuda3D/benchmarks/cases/perf_1gpu_6shots`
- logical ny/nx/nz: `384/384/95`
- shots/receivers_per_shot: `6/441`
- npml/radius/nbell: `12/7/1`
- xpad: `0.5` = `20` grid points
- aggregate K=2 deep-core share across shot-local subdomains: `73.22%`
- source influence overlaps K=2 deep core in `0` shots
- receiver footprint overlaps K=2 deep core in `0` shots

## Shot Table

| shot | domain y/x | K2 deep share | src influence z/x/y | src overlaps deep | rec footprint z/x/y | rec overlaps deep |
| ---: | ---: | ---: | --- | ---: | --- | ---: |
| 0 | 216x216 | 72.99% | [[5, 22], [99, 116], [99, 116]] | False | [[13, 14], [14, 208], [14, 208]] | False |
| 1 | 216x241 | 73.56% | [[5, 22], [124, 141], [99, 116]] | False | [[13, 14], [32, 233], [14, 208]] | False |
| 2 | 216x217 | 73.02% | [[5, 22], [124, 141], [99, 116]] | False | [[13, 14], [32, 228], [14, 208]] | False |
| 3 | 217x216 | 73.02% | [[5, 22], [99, 116], [124, 141]] | False | [[13, 14], [14, 208], [32, 228]] | False |
| 4 | 217x241 | 73.58% | [[5, 22], [124, 141], [124, 141]] | False | [[13, 14], [32, 233], [32, 228]] | False |
| 5 | 217x217 | 73.04% | [[5, 22], [124, 141], [124, 141]] | False | [[13, 14], [32, 228], [32, 228]] | False |

## Schedule Findings

- Source injection is shallow enough that its radius-7 influence does not overlap the K=2 deep-core region in this benchmark.
- Receiver extraction is also shallow and does not overlap K=2 deep core.
- Therefore source/receiver placement does not kill temporal blocking for this benchmark.
- The remaining blocker is still ownership/synchronization of `p(t+1)` and the byte cost of computing `p_mid` halos.

## Gate

- verdict: `stop_swept_wavefront_cuda_prototype`
- reason: source/receiver are compatible, but no implementable swept/wavefront schedule yet beats the Phase 4.1 byte/sync gate.

## Linked Phase 4.1 Byte Gate

- direct temporal verdict: `stop_cuda_prototype`
- ideal no-dup sampled-main speedup: `1.103x`
- CTA-local byte ratio range: `see summary`
