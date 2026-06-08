# Len16 Source Hotlines

| source line | samples | not-issued | share | source |
| ---: | ---: | ---: | ---: | --- |
| 1813 | 5660 | 4643 | 36.02% | `p0[base]=2*__ldg(p1+base)-p0[base]` |
| 1814 | 3890 | 3161 | 24.76% | `+__ldg(cw2+base)*dt*(c1+c2+c3);` |
| 1810 | 3287 | 2612 | 20.92% | `mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);` |
| 1804 | 927 | 660 | 5.90% | `mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);` |
| 1778 | 266 | 156 | 1.69% | `const size_t ts2 = (size_t)(gtid2 + radius) * stride2;` |
| 1815 | 233 | 190 | 1.48% | `}` |
| 1737 | 186 | 132 | 1.18% | `if (blockIdx.x >= ntile) return;` |
| 1738 | 180 | 109 | 1.15% | `const PmlTile tile = tiles[blockIdx.x];` |
| 1751 | 113 | 46 | 0.72% | `const int gtid1 = active_z0 + local_z;` |
| 1784 | 111 | 59 | 0.71% | `vz_line_cache[cbase-2])` |
| 1806 | 88 | 57 | 0.56% | `} else if (gtid1>=n1-npml) {` |
| 1811 | 80 | 46 | 0.51% | `c1+=mem_dzz[pind];` |
| 1748 | 69 | 26 | 0.44% | `const int core1_hi = n1 - npml - CorePmlMargin;` |
| 1770 | 64 | 31 | 0.41% | `if (local_z < 3) {` |
| 1788 | 60 | 23 | 0.38% | `vz_line_cache[cbase-4]);` |
