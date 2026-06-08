# Len16 Half-Warp Perf6 Repeat

- Pass: `True`
- Rounds: `3`
- Mean base WP: `2.207751s`
- Mean candidate WP: `2.039080s`
- Mean WP speedup: `1.082719x`
- Mean base Gradient: `2.316433s`
- Mean candidate Gradient: `2.159948s`
- Mean Gradient speedup: `1.072448x`
- Max rel L2: `6.384336e-07`

| round | base WP | cand WP | WP speedup | base Grad | cand Grad | Grad speedup | max rel L2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.212397 | 2.041717 | 1.083596 | 2.322951 | 2.161666 | 1.074611 | 6.384336e-07 |
| 2 | 2.203528 | 2.037935 | 1.081255 | 2.312805 | 2.159917 | 1.070784 | 6.384336e-07 |
| 3 | 2.207329 | 2.037589 | 1.083304 | 2.313544 | 2.158260 | 1.071949 | 6.384336e-07 |

## Candidate Tile Split

The six perf shots split pressure tiles as follows:

| shot/run order | len16 tiles | residual p tiles |
| ---: | ---: | ---: |
| 1 | 10816 | 7168 |
| 2 | 12064 | 8032 |
| 3 | 10816 | 7648 |
| 4 | 10816 | 7408 |
| 5 | 12064 | 8300 |
| 6 | 10816 | 7892 |
