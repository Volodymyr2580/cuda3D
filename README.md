# cuda3D

CUDA 3D wavefield forward-modeling optimization workspace.

This repository tracks source code, build scripts, benchmark utilities, and
agent reports for the RTX 5090 CUDA optimization workflow. Large velocity
models, generated benchmark outputs, binaries, object files, and profiler
artifacts are intentionally excluded from Git.

Current validated RTX 5090 build strategy:

```bash
cd /work/wenzhe/cuda3D
source ./env_5090.sh
cd src
make -B -f makefile.rtx5090 test NVFLAGS="-O3 -arch=sm_120 --use_fast_math -DCUDA3D_PML_RECOMPUTE_Z -DCUDA3D_PML_TILE_LIST -DCUDA3D_PML_ZMEM_IN_P -DPmlTileBlockSize1=32 -DPmlTileBlockSize2=4 -DPmlTileBlockSize3=2"
```

See `AGENTS.md`, `AGENT_LOG.md`, and
`overnight_20260607/reports/final_report.md` for the current benchmark
baseline and optimization notes.
