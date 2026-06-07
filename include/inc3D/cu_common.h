#ifndef cu_common_h
#define cu_common_h
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <errno.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
//#include <float.h>
#include <unistd.h>
//#include <complex.h>
#include <stdbool.h>
#include <mpi.h>

//#ifndef _OPENMP
//#include <omp.h>
//#endif

#include "rem_fd.h"
#include "single_solver.h"
#include "reduce.h"
#include "optimization_cuda.h"

extern "C" {
// common
#include "susgy.h"
#include "alloc.h"
#include "utility_zz.h"

// program related
#include "abc.h"
#include "acqui.h"
#include "lint.h"
}

#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))
#define	MAX(x,y) ((x) > (y) ? (x) : (y))
#define	MIN(x,y) ((x) < (y) ? (x) : (y))
#define pi (3.141592653589793)

#ifndef ABS
#define ABS(x) ((x) < 0 ? -(x) : (x))
#endif

#ifndef _gpu_par_
#define radius 4      
//#define radius 2      
#ifndef BlockSize1
#define BlockSize1 128// tile size in 1st-axis
#endif
#ifndef BlockSize2
#define BlockSize2 2// tile size in 2nd-axis
#endif
#ifndef BlockSize3
#define BlockSize3 1// tile size in 3rd-axis
#endif

#ifndef VBlockSize1
#define VBlockSize1   128
#endif
#ifndef VBlockSize2
#define VBlockSize2   2
#endif
#ifndef VBlockSize3
#define VBlockSize3   1
#endif

#ifndef PBlockSize1
#define PBlockSize1   128
#endif
#ifndef PBlockSize2
#define PBlockSize2   2
#endif
#ifndef PBlockSize3
#define PBlockSize3   1
#endif

#ifndef PCoreBlockSize1
#define PCoreBlockSize1 128
#endif
#ifndef PCoreBlockSize2
#define PCoreBlockSize2 8
#endif
#ifndef PCoreBlockSize3
#define PCoreBlockSize3 1
#endif

#ifndef PmlBlockSize1
#define PmlBlockSize1   128
#endif
#ifndef PmlBlockSize2
#define PmlBlockSize2   2
#endif
#ifndef PmlBlockSize3
#define PmlBlockSize3   1
#endif
#ifndef PmlTileBlockSize1
#define PmlTileBlockSize1   32
#endif
#ifndef PmlTileBlockSize2
#define PmlTileBlockSize2   4
#endif
#ifndef PmlTileBlockSize3
#define PmlTileBlockSize3   2
#endif
#ifndef PmlZFaceBlockSize1
#define PmlZFaceBlockSize1   8
#endif
#ifndef PmlZFaceBlockSize2
#define PmlZFaceBlockSize2   8
#endif
#ifndef PmlZFaceBlockSize3
#define PmlZFaceBlockSize3   4
#endif
#define BlockSize 256
#define MAX_BLOCK_SZ 256
#endif

#ifndef CorePmlMargin
#define CorePmlMargin 4
#endif

#ifndef CUDA3D_CORE_STENCIL_RADIUS
#define CUDA3D_CORE_STENCIL_RADIUS 7
#endif

#ifndef _con_stencil
#define _con_stencil   

__constant__ float stencil[radius+1]={105.0/945755921747804200.0,
				      1131397464981504000.0/945755921747804200.0,
				      -75426497665433580.0/945755921747804200.0,
				      9051179719852032.0/945755921747804200.0,
				      -659706976665600.0/945755921747804200.0};

//__constant__ float stencil[radius+1]={0.,
//				      1.125,
//				      -0.0417};
#endif

//#ifndef _check_error
//#define _check_error
//void check_gpu_error(const char *msg){                                                                        
//  cudaError_t err = cudaGetLastError ();                                                                      
//  if (cudaSuccess !=err){                                                                                     
//    printf("Cuda error: %s: %s", msg, cudaGetErrorString(err));                                               
//    exit(0);                                                                                                  
//  }                                                                                                           
//}  
//#endif

#endif
