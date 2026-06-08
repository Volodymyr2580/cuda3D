# P-Core Readonly LDG Perf6 Repeat

Candidate macro: `CUDA3D_P_CORE_READONLY_LDG`

- all compare pass: `True`
- mean WP speedup vs direct-fill: `0.999319x`
- mean Gradient speedup vs direct-fill: `0.999254x`

| round | direct WP | candidate WP | WP speedup | direct Gradient | candidate Gradient | Gradient speedup | compare |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.214108 | 2.215781 | 0.999245 | 2.322992 | 2.322578 | 1.000178 | True |
| 2 | 2.193193 | 2.193487 | 0.999866 | 2.309177 | 2.311565 | 0.998967 | True |
| 3 | 2.193033 | 2.195564 | 0.998847 | 2.310981 | 2.314180 | 0.998618 | True |
