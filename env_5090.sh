#!/usr/bin/env bash
set -e

export CUDA_HOME=/usr/local/cuda-13.0
export MPI_HOME=/opt/intel/oneapi/mpi/latest
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

if [ -f /opt/intel/oneapi/setvars.sh ]; then
  source /opt/intel/oneapi/setvars.sh --force >/tmp/cuda3d_oneapi_setvars.log 2>&1
fi

if [ -f /work/wenzhe/miniforge3/etc/profile.d/conda.sh ]; then
  source /work/wenzhe/miniforge3/etc/profile.d/conda.sh
  conda activate cuda3d
fi

export CUDA3D_ROOT=/work/wenzhe/cuda3D
