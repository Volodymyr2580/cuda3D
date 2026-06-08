# Inject/Extract BS512 Perf6 Repeat

Candidate macro: `CUDA3D_INJECT_EXTRACT_BS512`

- all compare pass: `True`
- mean WP speedup vs direct-fill: `0.999684x`
- mean Gradient speedup vs direct-fill: `0.998963x`

| round | direct WP | candidate WP | WP speedup | direct Gradient | candidate Gradient | Gradient speedup | compare |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.210482 | 2.207253 | 1.001463 | 2.317219 | 2.319892 | 0.998848 | True |
| 2 | 2.188303 | 2.191148 | 0.998702 | 2.305490 | 2.309288 | 0.998355 | True |
| 3 | 2.189624 | 2.192060 | 0.998889 | 2.308208 | 2.308930 | 0.999687 | True |
