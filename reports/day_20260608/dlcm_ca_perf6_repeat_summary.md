# DLCM CA Perf6 Repeat

Candidate flags append: `-Xptxas -dlcm=ca`

- all compare pass: `True`
- mean WP speedup vs direct-fill: `0.999263x`
- mean Gradient speedup vs direct-fill: `0.999576x`

| round | direct WP | candidate WP | WP speedup | direct Gradient | candidate Gradient | Gradient speedup | compare |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.208100 | 2.209109 | 0.999543 | 2.317923 | 2.321739 | 0.998356 | True |
| 2 | 2.188629 | 2.189937 | 0.999403 | 2.308771 | 2.307878 | 1.000387 | True |
| 3 | 2.190207 | 2.192747 | 0.998842 | 2.307964 | 2.308002 | 0.999984 | True |
