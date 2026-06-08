# Phase 4.3 Pressure PML Dataflow Gate Decision

## Decision

Open one CUDA prototype:

```text
CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
```

This is a pressure-PML z-line recompute reuse prototype.  It is not a
tile-mask fastpath, not z-face specialization/fusion, and not
`RECOMPUTE_X/Y/XYZ`.

## Evidence

Source audit:

```text
tool                                    tools/pml_pressure_dataflow_audit.py
case                                    benchmarks/cases/perf_1gpu_6shots
pressure PML tiles kept                 113840 / 181232
active thread efficiency                65.60%
valid-domain thread efficiency          87.32%
returned-core threads in kept tiles     6328998
shell active points                     4143640
shell share of active points            21.67%
true-PML share of active points         78.33%
```

Pressure z-recompute budget:

```text
current recompute_vz_after_update calls          152951552
shared z-line cache call estimate                 29093740
estimated z recompute call reduction              80.98%
current p1 loads inside z recompute               4667.711 MiB/step aggregate-shots
shared-cache p1 load estimate                      887.870 MiB/step aggregate-shots
```

NCU-linked sampled-main model:

```text
cuda_fd3d_p_pml_tile_ns duration        189.5616 us
cuda_fd3d_p_core_ns duration             93.6704 us
cuda_fd3d_v_pml_tile_ns duration         71.6096 us
p_pml sampled-main share                 53.42%
modeled p_pml speedup                    1.573x
modeled sampled-main speedup             1.242x
```

The model passes the `>=5%` meaningful prototype gate.

Remote reproduction:

```text
server path       /work/wenzhe/cuda3D
remote gate       open_p_pml_z_recompute_line_cache_prototype
remote speedup    1.2417261903808379 sampled-main model
shell checksum    4143640 == 4143640
```

## Prototype Boundary

Allowed:

- Add a macro-default-off pressure-PML tile kernel variant that caches the
  `recompute_vz_after_update_from_old_mem` values for each CTA z-line.
- Keep ownership of `memory_dz_next` identical: only the tile-owned central
  z range may write the next z CPML memory.
- Keep x/y velocity paths unchanged; no `RECOMPUTE_X`, `RECOMPUTE_Y`, or
  `RECOMPUTE_XYZ`.
- Validate with debug dump step 0/1/2 before performance claims.

Still forbidden:

- `CUDA3D_PML_TILE_MASK_FASTPATH`
- `CUDA3D_PML_ZFACE_P_SPECIALIZE`
- `CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY`
- `CUDA3D_PML_ZFACE_SHARED_VP_DEBUG`
- PML tile block shape sweep
- register cap sweep

## Stop Rule

Stop this prototype if any of the following happens:

- build fails in a way that requires broad unrelated rewrites;
- debug dump step 0/1/2 fails;
- correctness exceeds rel L2 `1e-5` or produces NaN/Inf;
- `perf_1gpu_6shots repeat` does not show at least `>=5%` meaningful WP
  speedup relative to `zmem_reference`.

