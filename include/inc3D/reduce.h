#ifndef _reduce_h
#define _reduce_h
#include "cu_common.h"
__global__ void reduce2(float *g_odata, float *g_idata, size_t len);
__global__ void reduce3(float *g_odata, float *g_idata, size_t len);
__global__ void reduce4(float *g_odata, float *g_idata, size_t len);

template <unsigned int blockSize>
__global__ void reduce5(float *reduct, float *array_in, size_t array_len);

__global__ void block_sum_reduce(float *d_block_sums, 
				 float *d_in,
				 size_t d_in_len);

void print_d_array(float *d_array, unsigned int len);

float gpu_sum_reduce(float *d_in, size_t d_in_len);

#endif
