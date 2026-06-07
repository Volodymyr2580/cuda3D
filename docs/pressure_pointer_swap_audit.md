# Pressure Pointer Swap Audit

Date: 2026-06-07

Scope: read-only audit before `CUDA3D_PRESSURE_TRIPLE_BUFFER_PIPELINE`.

## Summary

Current code uses two pressure buffers:

```text
d_p0: previous pressure input p(t-1), overwritten in-place with p(t+1)
d_p1: current pressure input p(t)
```

At the end of each timestep:

```cpp
ptr=d_p0; d_p0=d_p1; d_p1=ptr;
```

After the swap, the newly written field becomes `d_p1` for the next step.

Triple buffer must make the semantic roles explicit:

```text
p_prev = read-only p(t-1)
p_curr = read-only p(t)
p_next = write-only p(t+1), then modified by source injection and read by receiver extraction
```

## Audit Table

| Location | Current Buffer | Meaning | Read/Write | Triple Buffer Mapping |
|---|---|---|---|---|
| `src/rem_fd.cu`: device declarations | `d_p0`, `d_p1`, `ptr` | Two pressure buffers plus swap temp | allocation / pointer ownership | Add `d_p_prev`, `d_p_curr`, `d_p_next`, rotate three pointers |
| `src/rem_fd.cu`: `cudaMalloc(d_p0/d_p1)` | `d_p0`, `d_p1` | pressure wavefields with padded dimensions | allocate | Allocate three pressure buffers |
| `src/rem_fd.cu`: `cudaMemset(d_p0/d_p1)` | `d_p0`, `d_p1` | zero initial pressure states | write | Zero all three buffers; debug may fill `p_next` before each step |
| `src/rem_fd.cu`: `cuda_fd3d_v_pml_*` launch | `d_p1` | current pressure `p(t)` used to update velocity/PML memory | read | pass `p_curr` |
| `src/single_solver.cu`: `cuda_fd3d_v_pml_ns` | `p1` | current pressure | read-only | rename/alias as `p_curr` |
| `src/rem_fd.cu`: `cuda_fd3d_p_core_ns(d_p0,d_p1,...)` | `d_p0`, `d_p1` | `d_p0` is previous and output; `d_p1` is current | read/write `d_p0`, read `d_p1` | signature must become `(p_next, p_curr, p_prev, ...)` |
| `src/single_solver.cu`: `cuda_fd3d_p_core_ns` | `p0[base] = 2*p1 - p0 + ...` | second-order pressure update | read old `p0`, write new `p0`, read `p1` | write `p_next[base] = 2*p_curr - p_prev + ...` |
| `src/rem_fd.cu`: `cuda_fd3d_p_pml_*` launch | `d_p0`, `d_p1` | pressure PML update | read/write `d_p0`, read `d_p1` | signatures must become `(p_next, p_curr, p_prev, ...)` |
| `src/single_solver.cu`: `cuda_fd3d_p_pml_ns` | `p0[outIndex]=2*p1-p0+...` | second-order PML pressure update | read old `p0`, write new `p0`, read `p1` | write `p_next[outIndex] = 2*p_curr - p_prev + ...` |
| `src/single_solver.cu`: `cuda_fd3d_p_pml_tile_ns` | `p0[outIndex]=2*p1-p0+...` | tile-list pressure PML update | read old `p0`, write new `p0`, read `p1` | same signature change required |
| `src/single_solver.cu`: `cuda_fd3d_p_pml_zface_ns` | `p0[outIndex]=2*p1-p0+...` | z-face specialized pressure update | read old `p0`, write new `p0`, read `p1` | same signature change required if route is re-enabled |
| `src/rem_fd.cu`: `CUDA3D_PML_ZMEM_IN_P` swap | `d_memory_dz`, `d_memory_dz_next` | z-PML velocity memory double buffer | pointer swap after pressure PML | keep same relative position: after PML pressure writes, before source/receiver |
| `src/rem_fd.cu`: PML debug dump | `d_p0`, `d_p1` | post-pressure, pre-source dump of old names | read | dump `p_next` and `p_curr`; compatibility doc should map old `p0` to new `p_next` at same location |
| `src/rem_fd.cu`: `lint3d_inject_bell_extract_gpu_zz(d_p0,...)` | `d_p0` | post-pressure field for source injection and receiver extraction | write source, read receiver | pass `p_next` |
| `src/rem_fd.cu`: separate `lint3d_inject_bell_gpu(d_p0,...)` | `d_p0` | source injection for large receiver count path | write | pass `p_next` |
| `src/rem_fd.cu`: `lint3d_extract_gpu_zz(d_p0,...)` | `d_p0` | receiver extraction after source injection | read | pass `p_next` |
| `src/rem_fd.cu`: output copy to host | `d_est` | receiver traces accumulated by extract kernels | read device to host | unchanged |
| `src/rem_fd.cu`: pressure swap | `ptr=d_p0; d_p0=d_p1; d_p1=ptr;` | rotate two pressure states | pointer write | rotate `p_prev <- p_curr`, `p_curr <- p_next`, `p_next <- old p_prev` |
| MPI/halo exchange | no direct p0/p1 halo exchange found in `fd_3d_f` | MPI appears shot/rank distribution only in this path | N/A | no triple-buffer MPI work in single-GPU first version |
| Forward/backward/gradient path | `optimization_cuda.cu` calls `fd_3d_f`; no separate backward pressure kernel is active | forward modeling is current CUDA path | N/A | first version only changes `fd_3d_f` pressure ownership |

## Key Finding

Host-only pointer remapping is not sufficient.

The pressure kernels themselves read old `p0` and write new `p0` in the same expression:

```cpp
p0[...] = 2*p1[...] - p0[...] + ...
```

Triple buffer therefore requires explicit kernel signatures that separate:

```text
p_next output
p_curr input
p_prev input
```

## Open Checks Before Implementation

- Confirm all enabled pressure PML variants are covered: generic, tile-list, zface if compiled.
- Decide whether legacy unused kernels such as `cuda_fd3d_p_pml2` need macro-gated triple-buffer variants or can remain untouched.
- Define debug dump compatibility names before changing compare tools.
- Add optional debug fill for `p_next` to detect stale reads from not-written cells.
