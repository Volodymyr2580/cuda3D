//#include <cuda
// modified accroding to 
//https://github.com/mark-poscablo/gpu-sum-reduction/blob/master/sum_reduction/main.cu
// reduce3 &4 work
#include "reduce.h"

__global__ void reduce2(float *g_odata, float *g_idata, size_t len) {
  extern volatile __shared__ float sdata[];

  // each thread loads one element from global to shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  sdata[tid] = 0;

  if (i < len)
    {
      sdata[tid] = g_idata[i];
    }

  __syncthreads();

  // do reduction in shared mem
  // Sequential addressing. This solves the bank conflicts as
  //  the threads now access shared memory with a stride of one
  //  32-bit word (unsigned int) now, which does not cause bank 
  //  conflicts
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduce3(float *g_odata, float *g_idata, size_t len) {
  extern volatile __shared__ float sdata[];

  // each thread loads one element from global to shared mem
  // Do the first stage of the reduction on the global-to-shared load step
  // This reduces the previous inefficiency of having half of the threads being
  //  inactive on the first for-loop iteration below (previous first step of reduction)
  // Previously, only less than or equal to 512 out of 1024 threads in a block are active.
  // Now, all 512 threads in a block are active from the start
  unsigned int tid = threadIdx.x;
  size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  //  unsigned int gtid = blockIdx.x * blockDim.x  + threadIdx.x;

  sdata[tid] = 0;
  
  if (i < len)
    {
      sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    }

  __syncthreads();

  // do reduction in shared mem
  // this loop now starts with s = 512 / 2 = 256
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  
  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

__device__ void warpReduce_r(volatile float *sdata, size_t tid){ // need volatile!!!!           
  /*
  if(blockDim.x>=64)sdata[tid]+=sdata[tid+32];
  if(blockDim.x>=32)sdata[tid]+=sdata[tid+16];
  if(blockDim.x>=16)sdata[tid]+=sdata[tid+ 8];
  if(blockDim.x>= 8)sdata[tid]+=sdata[tid+ 4];
  if(blockDim.x>= 4)sdata[tid]+=sdata[tid+ 2];
  if(blockDim.x>= 2)sdata[tid]+=sdata[tid+ 1];
  */
  sdata[tid]+=sdata[tid+32];
  sdata[tid]+=sdata[tid+16];
  sdata[tid]+=sdata[tid+ 8];
  sdata[tid]+=sdata[tid+ 4];
  sdata[tid]+=sdata[tid+ 2];
  sdata[tid]+=sdata[tid+ 1];
} 

__global__ void reduce4(float *g_odata, float *g_idata, size_t len) {
  extern volatile __shared__ float sdata[];

  // each thread loads one element from global to shared mem
  // Do the first stage of the reduction on the global-to-shared load step
  // This reduces the previous inefficiency of having half of the threads being
  //  inactive on the first for-loop iteration below (previous first step of reduction)
  // Previously, only less than or equal to 512 out of 1024 threads in a block are active.
  // Now, all 512 threads in a block are active from the start
  unsigned int tid = threadIdx.x;
  size_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  
  sdata[tid] = 0;  

  if (i < len)
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];  
  __syncthreads();
 
  /*
  while(i< len){
    sdata[tid]+=g_idata[i]+g_idata[i+MAX_BLOCK_SZ];
    i+=MAX_BLOCK_SZ*2*gridDim.x;
  }
  __syncthreads();
  */
  // do reduction in shared mem
  // this loop now starts with s = 512 / 2 = 256
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if(tid<32) warpReduce_r(sdata, tid); // need call wrap as subroutine, the following will not work!
  /*    
  if (tid < 32)
    {
      sdata[tid] += sdata[tid + 32];
      sdata[tid] += sdata[tid + 16];
      sdata[tid] += sdata[tid + 8];
      sdata[tid] += sdata[tid + 4];
      sdata[tid] += sdata[tid + 2];
      sdata[tid] += sdata[tid + 1];
    }
  */
  // write result for this block to global mem
  if (tid == 0)
    g_odata[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize>
__global__ void reduce5(float *reduct, float *array_in, size_t array_len)
{
  extern volatile __shared__ float sdata[];
  size_t  tid        = threadIdx.x,
    gridSize   = blockSize * gridDim.x,
    i          = blockIdx.x * blockSize + tid;
  sdata[tid] = 0;
  while (i < array_len)
    { sdata[tid] += array_in[i];
      i += gridSize; }
  __syncthreads();
  if (blockSize >= 512)
    { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
  if (blockSize >= 256)
    { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
  if (blockSize >= 128)
    { if (tid <  64) sdata[tid] += sdata[tid + 64]; __syncthreads(); }
  if (tid < 32)
    { if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
      if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
      if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
      if (blockSize >= 8)  sdata[tid] += sdata[tid + 4];
      if (blockSize >= 4)  sdata[tid] += sdata[tid + 2];
      if (blockSize >= 2)  sdata[tid] += sdata[tid + 1]; }
  if (tid == 0) reduct[blockIdx.x] = sdata[0];
}


// not working !!!
__global__ void block_sum_reduce(float *d_block_sums, 
				 float *d_in,
				 size_t d_in_len){

  extern __shared__ float s_out[];
	
  //  unsigned int max_elems_per_block = blockDim.x * 2;
  unsigned int glbl_tid = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int tid = threadIdx.x;
	
  // Zero out shared memory
  // Especially important when padding shmem for
  //  non-power of 2 sized input
  s_out[threadIdx.x] = 0;
  s_out[threadIdx.x + blockDim.x] = 0;

  __syncthreads();

  // Copy d_in to shared memory per block
  if (glbl_tid < d_in_len)
    {
      s_out[threadIdx.x] = d_in[glbl_tid];
      if (glbl_tid + blockDim.x < d_in_len)
	s_out[threadIdx.x + blockDim.x] = d_in[glbl_tid + blockDim.x];
    }
  __syncthreads();
  
  // Actually do the reduction
  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) {
      s_out[tid] += s_out[tid + s];
      //      atomicAdd(&s_out[tid], s_out[tid + s]);
    }
    __syncthreads();
  }
  __syncthreads();
  // write result for this block to global mem
  if (tid == 0)
    d_block_sums[blockIdx.x] = s_out[0];
}

void print_d_array(float *d_array, unsigned int len)
{
  float *h_array = alloc1float(len);
  cudaMemcpy(h_array, d_array, sizeof(float) * len, cudaMemcpyDeviceToHost);
  for (unsigned int i = 0; i < len; ++i)
    {
      printf("i=%d, h_array=%f\n",i,h_array[i]);
    }

  free1float(h_array);
}

float gpu_sum_reduce(float *d_in, size_t d_in_len)
{
	float total_sum = 0;

	// Set up number of threads and blocks
	// If input size is not power of two, the remainder will still need a whole block
	// Thus, number of blocks must be the least number of 2048-blocks greater than the input size
	unsigned int block_sz = MAX_BLOCK_SZ; // Halve the block size due to reduce3() and further 
											  //  optimizations from there
	// our block_sum_reduce()
	unsigned int max_elems_per_block = block_sz * 2; // due to binary tree nature of algorithm
	// NVIDIA's reduceX()
	//unsigned int max_elems_per_block = block_sz;
	
	size_t grid_sz = 0;
	if (d_in_len <= max_elems_per_block) {
	  grid_sz = (unsigned int)(ceil(float(d_in_len) / float(max_elems_per_block)));

	}
	else{
	  
	  grid_sz = d_in_len / max_elems_per_block;
	  if (d_in_len % max_elems_per_block != 0)
	    grid_sz++;
	}

	// Allocate memory for array of total sums produced by each block
	// Array length must be the same as number of blocks / grid size
	float *d_block_sums;
	cudaMalloc(&d_block_sums, sizeof(float) * grid_sz);
	cudaMemset(d_block_sums, 0, sizeof(float) * grid_sz);

	// Sum data allocated for each block
	//	block_sum_reduce<<<grid_sz, block_sz, sizeof(float) * max_elems_per_block>>>(d_block_sums, d_in, d_in_len); // not working !!!
	//	reduce4<<<grid_sz, block_sz>>>(d_block_sums, d_in, d_in_len);
	reduce5<MAX_BLOCK_SZ><<<grid_sz, block_sz, sizeof(float) * block_sz>>>(d_block_sums, d_in, d_in_len);

	//	if(grid_sz<12){
	//	  printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@ here print d_in again@@@@@@@@@@@@@@@\n");
	  //	  print_d_array(d_in, d_in_len);
	//	}
	//	print_d_array(d_block_sums, grid_sz);
	//	sleep(2);
	//	printf("grid_sz=%d, mpb=%d\n", grid_sz, max_elems_per_block);
	//printf("mepb=%d\n", max_elems_per_block);

	// Sum each block's total sums (to get global total sum)
	// Use basic implementation if number of total sums is <= 2048
	// Else, recurse on this same function
	if (grid_sz <= max_elems_per_block){
	  //	  printf("supposed to be last\n");
	  float *d_total_sum;
	  cudaMalloc((void**)&d_total_sum, sizeof(float));
	  cudaMemset(d_total_sum, 0., sizeof(float));
	  
	  //	  block_sum_reduce<<<1, block_sz, sizeof(float) * max_elems_per_block>>>(d_total_sum, d_block_sums, grid_sz); //  working here !!!
	  // 	  reduce4<<<1, block_sz, sizeof(float) * block_sz>>>(d_total_sum, d_block_sums, grid_sz);
	  // 	  reduce4<<<1, block_sz>>>(d_total_sum, d_block_sums, grid_sz);
	  reduce5<MAX_BLOCK_SZ><<<1, block_sz, sizeof(float) * block_sz>>>(d_total_sum, d_block_sums, grid_sz);

	  cudaMemcpy(&total_sum, d_total_sum, sizeof(float), cudaMemcpyDeviceToHost);
	  cudaFree(d_total_sum);
	  //	  printf("in total last a2=%f\n", total_sum);
	  //	  sleep(10);
	}
	else{
	  //	  printf("supposed to be first\n");
	  float *d_in_block_sums;
	  cudaMalloc((void**)&d_in_block_sums, sizeof(float) * grid_sz);
	  cudaMemset(d_in_block_sums, 0., sizeof(float)*grid_sz);
	  cudaMemcpy(d_in_block_sums, d_block_sums, sizeof(float) * grid_sz, cudaMemcpyDeviceToDevice);
	  total_sum = gpu_sum_reduce(d_in_block_sums, grid_sz);
	  cudaFree(d_in_block_sums);
	  //	  printf("in total a2=%f\n", total_sum);
	  //	  sleep(2);
	}

	cudaFree(d_block_sums);
	//	printf("in total a2=%f\n", total_sum);
	return total_sum;
}



