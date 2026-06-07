# RTX 5090 Formal Speed Table

Generated on the same server under `/work/wenzhe/cuda3D`. `default_no_macro` is the available original-like/no-macro source path, not a proven untouched upstream tarball.

## Build Variants

| Variant | Compile Flags | Binary SHA256 |
|---|---|---|
| `default_no_macro` | `-O3 -arch=sm_120 --use_fast_math` | `21df625f8246a9e1309b593fe63d9ca188de0eeb136ed5bde401c3b5a71d4b04` |
| `current_best_reference` | `-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2` | `ffceea2433f787334b1bcd48578f4fe7beae1e89e5836f20429b4390e27fa780` |
| `zmem_reference` | `-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2` | `b7a5ac86612ff791662f49992538ffeac14eee61d73f739a9f7fe66d6852e867` |

## Timings

| Variant | perf1 WP | perf1 Gradient | perf6 WP | perf6 repeat WP | perf6 mean WP | Speedup vs default | perf6 mean Gradient | Gradient speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `default_no_macro` | 0.547242s | 0.576376s | 2.713933s | 2.715355s | 2.714644s | 1.000000x | 2.850866s | 1.000000x |
| `current_best_reference` | 0.511066s | 0.540409s | 2.507581s | 2.509065s | 2.508323s | 1.082255x | 2.638547s | 1.080468x |
| `zmem_reference` | 0.484380s | 0.515100s | 2.416675s | 2.414738s | 2.415706s | 1.123747x | 2.538267s | 1.123155x |

## Correctness

| Variant | Case | Pass | Max rel L2 | Max abs |
|---|---|---:|---:|---:|
| `current_best_reference` | `correctness` | `true` | 0.000000000e+00 | 0.000000000e+00 |
| `current_best_reference` | `perf1` | `true` | 0.000000000e+00 | 0.000000000e+00 |
| `current_best_reference` | `perf6` | `true` | 0.000000000e+00 | 0.000000000e+00 |
| `current_best_reference` | `perf6_repeat` | `true` | 0.000000000e+00 | 0.000000000e+00 |
| `zmem_reference` | `correctness` | `true` | 0.000000000e+00 | 0.000000000e+00 |
| `zmem_reference` | `perf1` | `true` | 0.000000000e+00 | 0.000000000e+00 |
| `zmem_reference` | `perf6` | `true` | 0.000000000e+00 | 0.000000000e+00 |
| `zmem_reference` | `perf6_repeat` | `true` | 0.000000000e+00 | 0.000000000e+00 |
| `zmem_reference` | `correctness_vs_current_best` | `true` | 0.000000000e+00 | 0.000000000e+00 |
| `zmem_reference` | `perf1_vs_current_best` | `true` | 0.000000000e+00 | 0.000000000e+00 |
| `zmem_reference` | `perf6_vs_current_best` | `true` | 0.000000000e+00 | 0.000000000e+00 |
| `zmem_reference` | `perf6_repeat_vs_current_best` | `true` | 0.000000000e+00 | 0.000000000e+00 |

## Interpretation

- `current_best_reference` speedup over `default_no_macro`: WP `1.082255x`, Gradient `1.080468x`.
- `zmem_reference` speedup over `default_no_macro`: WP `1.123747x`, Gradient `1.123155x`.
- `zmem_reference` speedup over `current_best_reference`: WP `1.038339x`, Gradient `1.039507x`.
- All listed output comparisons use `rel_l2 <= 1e-5` and finite-value checks via `tools/compare_outputs.py`.
