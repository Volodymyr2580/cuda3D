# DLCM CG Perf6 Repeat

Candidate flags append: `-Xptxas -dlcm=cg`

- all compare pass: `True`
- mean WP speedup vs direct-fill: `0.859344x`
- mean Gradient speedup vs direct-fill: `0.864052x`

| round | direct WP | candidate WP | WP speedup | direct Gradient | candidate Gradient | Gradient speedup | compare |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.211902 | 2.565765 | 0.862083 | 2.320846 | 2.684849 | 0.864423 | True |
| 2 | 2.188347 | 2.559140 | 0.855110 | 2.310185 | 2.674301 | 0.863846 | True |
| 3 | 2.189511 | 2.543460 | 0.860840 | 2.308133 | 2.671802 | 0.863886 | True |
