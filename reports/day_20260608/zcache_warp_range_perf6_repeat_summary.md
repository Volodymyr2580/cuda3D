# Z-Cache Warp-Range Perf6 Repeat

- all compare pass: `True`
- mean WP speedup vs direct-fill: `0.997223x`
- mean Gradient speedup vs direct-fill: `0.997502x`

| round | direct WP | warp WP | WP speedup | direct Gradient | warp Gradient | Gradient speedup | compare |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.196203 | 2.207274 | 0.994984 | 2.308980 | 2.314959 | 0.997417 | True |
| 2 | 2.182848 | 2.187369 | 0.997933 | 2.298789 | 2.305196 | 0.997221 | True |
| 3 | 2.180586 | 2.183309 | 0.998753 | 2.298663 | 2.303571 | 0.997869 | True |
