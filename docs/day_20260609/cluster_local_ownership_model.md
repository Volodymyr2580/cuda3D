# Cluster-Local Ownership Model

## Summary

Thread-block clusters are available on the RTX 5090, but the
cluster-local K=2 temporal route does not pass the byte model gate.

Decision:

```text
reject_cluster_local_temporal_cuda_prototype
ordinary CUDA prototype allowed = false
cluster CUDA prototype allowed = false
```

## Current-Best Anchor

```text
formal current-best WP speedup   1.2220x
sampled main                     284.010us
p_core                           93.730us
p_core share                     33.00%
```

## Cooperative / Cluster Capacity

```text
cooperative grid ceiling         2040 blocks
previous K=2 required blocks     70688 blocks
cooperative over-capacity        34.6510x
max passing cluster size         8
```

## Byte Gate

```text
baseline pair bytes/output       256.875
p_mid compute bytes/output       124.438
final step bytes/output          12.000
required p_core reduction        14.43%
required local pair byte ratio   <= 0.8557
ideal no-dup sampled speedup     1.1317x
```

The DSM tile search is optimistic: it lets a cluster spend all per-block
shared memory as one distributed tile and ignores DSM latency.  A real
implementation would be harder, so this is an upper-bound filter.

Best DSM tile found:

```text
cluster size                     8
output z/x/y                     40 / 44 / 48
p_mid z/x/y                      54 / 58 / 62
p_mid bytes                      776736
p_mid elements/output            2.2986
local pair byte ratio            1.1602
estimated sampled-main speedup   0.9498x
```

Best tile by cluster size:

| cluster size | output z/x/y | p_mid bytes | local pair byte ratio | sampled-main estimate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16x12x16 | 93600 | 3.7367x | 0.5254x |
| 2 | 24x20x24 | 196384 | 2.1113x | 0.7317x |
| 4 | 32x32x32 | 389344 | 1.4857x | 0.8619x |
| 8 | 40x44x48 | 776736 | 1.1602x | 0.9498x |

## Gate

- decision: `reject_cluster_local_temporal_cuda_prototype`
- reason: Even an optimistic 8-block cluster DSM tile search is slower than the current two-pass p_core byte model. It gives local pair byte ratio > 1.0 instead of the <= threshold needed for a 5% sampled-main win.

Do not continue:
- direct cooperative-grid K=2 temporal prototype
- cluster-local K=2 temporal CUDA prototype with DSM p_mid tile
- cluster producer-consumer fusion without a new ownership model that beats the DSM byte gate

Allowed next directions:
- precision-relaxation study only after explicit tolerance policy change
- application-level multi-shot scheduling / batching
- a fundamentally different ownership representation that proves p_mid/state traffic removal without DSM halo blow-up
