#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "%s:%d: CUDA error %s: %s\n", __FILE__, __LINE__,   \
                   #call, cudaGetErrorString(err__));                         \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

static int device_attr(cudaDeviceAttr attr) {
  int value = -1;
  cudaError_t err = cudaDeviceGetAttribute(&value, attr, 0);
  if (err != cudaSuccess) {
    std::printf("attr_%d_error=%s\n", static_cast<int>(attr),
                cudaGetErrorString(err));
    return -1;
  }
  return value;
}

__global__ void plain_kernel(int *out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *out = gridDim.x;
  }
}

__global__ void cooperative_grid_kernel(int *out) {
  cg::grid_group grid = cg::this_grid();
  grid.sync();
  if (grid.thread_rank() == 0) {
    *out = grid.num_blocks();
  }
}

__global__ void cluster_sync_kernel(int *out) {
  cg::cluster_group cluster = cg::this_cluster();
  cluster.sync();
  if (cluster.block_rank() == 0 && threadIdx.x == 0) {
    atomicAdd(out, cluster.num_blocks());
  }
}

static int run_cooperative_probe(int *device_out, int blocks, int block_size) {
  CUDA_CHECK(cudaMemset(device_out, 0, sizeof(int)));
  void *args[] = {&device_out};
  cudaError_t err = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(cooperative_grid_kernel), blocks, block_size,
      args);
  if (err != cudaSuccess) {
    std::printf("cooperative_launch_blocks_%d=%s\n", blocks,
                cudaGetErrorString(err));
    return EXIT_SUCCESS;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  int host_out = -1;
  CUDA_CHECK(cudaMemcpy(&host_out, device_out, sizeof(int),
                        cudaMemcpyDeviceToHost));
  std::printf("cooperative_launch_blocks_%d=pass observed_blocks=%d\n", blocks,
              host_out);
  return EXIT_SUCCESS;
}

static int run_cluster_probe(int *device_out, int cluster_size, int block_size) {
  CUDA_CHECK(cudaMemset(device_out, 0, sizeof(int)));

  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = static_cast<unsigned int>(cluster_size);
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(cluster_size * 2, 1, 1);
  config.blockDim = dim3(block_size, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = nullptr;
  config.attrs = &attr;
  config.numAttrs = 1;

  int active_clusters = -1;
  cudaError_t occ_err =
      cudaOccupancyMaxActiveClusters(&active_clusters,
                                     reinterpret_cast<const void *>(
                                         cluster_sync_kernel),
                                     &config);
  std::printf("cluster_size_%d_active_clusters_status=%s active_clusters=%d\n",
              cluster_size, cudaGetErrorString(occ_err), active_clusters);

  cudaError_t launch_err = cudaLaunchKernelEx(&config, cluster_sync_kernel,
                                              device_out);
  if (launch_err != cudaSuccess) {
    std::printf("cluster_size_%d_launch=%s\n", cluster_size,
                cudaGetErrorString(launch_err));
    return EXIT_SUCCESS;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  int host_out = -1;
  CUDA_CHECK(cudaMemcpy(&host_out, device_out, sizeof(int),
                        cudaMemcpyDeviceToHost));
  std::printf("cluster_size_%d_launch=pass observed_cluster_block_sum=%d\n",
              cluster_size, host_out);
  return EXIT_SUCCESS;
}

int main() {
  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  std::printf("device_count=%d\n", device_count);
  if (device_count <= 0) {
    return EXIT_SUCCESS;
  }

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

  const int sm_count = device_attr(cudaDevAttrMultiProcessorCount);
  const int coop = device_attr(cudaDevAttrCooperativeLaunch);
  const int cluster = device_attr(cudaDevAttrClusterLaunch);
  const int max_blocks_per_sm = device_attr(cudaDevAttrMaxBlocksPerMultiprocessor);
  const int max_threads_per_sm =
      device_attr(cudaDevAttrMaxThreadsPerMultiProcessor);
  const int max_smem_per_sm =
      device_attr(cudaDevAttrMaxSharedMemoryPerMultiprocessor);
  const int max_regs_per_sm =
      device_attr(cudaDevAttrMaxRegistersPerMultiprocessor);

  std::printf("name=%s\n", prop.name);
  std::printf("compute_capability=%d.%d\n", prop.major, prop.minor);
  std::printf("multi_processor_count=%d\n", sm_count);
  std::printf("cooperative_launch=%d\n", coop);
  std::printf("cluster_launch=%d\n", cluster);
  std::printf("max_blocks_per_sm_attr=%d\n", max_blocks_per_sm);
  std::printf("max_threads_per_sm=%d\n", max_threads_per_sm);
  std::printf("max_shared_mem_per_sm=%d\n", max_smem_per_sm);
  std::printf("max_registers_per_sm=%d\n", max_regs_per_sm);

  const int block_size = 128;
  int active_blocks_per_sm = -1;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks_per_sm, plain_kernel, block_size, 0));
  const int cooperative_grid_block_ceiling = active_blocks_per_sm * sm_count;
  std::printf("plain_kernel_block_size=%d\n", block_size);
  std::printf("plain_kernel_active_blocks_per_sm=%d\n", active_blocks_per_sm);
  std::printf("cooperative_grid_block_ceiling=%d\n",
              cooperative_grid_block_ceiling);

  int *device_out = nullptr;
  CUDA_CHECK(cudaMalloc(&device_out, sizeof(int)));

  if (coop) {
    run_cooperative_probe(device_out, sm_count, block_size);
    run_cooperative_probe(device_out, cooperative_grid_block_ceiling, block_size);
    run_cooperative_probe(device_out, cooperative_grid_block_ceiling + 1,
                          block_size);
  }

  if (cluster) {
    for (int cluster_size : {1, 2, 4, 8, 16}) {
      run_cluster_probe(device_out, cluster_size, block_size);
    }
  }

  CUDA_CHECK(cudaFree(device_out));
  CUDA_CHECK(cudaDeviceReset());
  return EXIT_SUCCESS;
}
