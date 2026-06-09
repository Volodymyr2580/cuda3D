#define CUDA3D_DEFINE_PML_CONSTANTS
#include "single_solver.h"

#if defined(CUDA3D_PML_ZMEM_IN_P) && !defined(CUDA3D_PML_RECOMPUTE_Z)
#error "CUDA3D_PML_ZMEM_IN_P requires CUDA3D_PML_RECOMPUTE_Z"
#endif

#if defined(CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL) && !defined(CUDA3D_PML_ZMEM_IN_P)
#error "CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL currently requires the stable CUDA3D_PML_ZMEM_IN_P path"
#endif

#if defined(CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL) && !defined(CUDA3D_CPML_VMEM_DISABLE_MPI)
#error "CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL phase 1 is single-rank only; define CUDA3D_CPML_VMEM_DISABLE_MPI to acknowledge this gate"
#endif

#if defined(CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY) && !defined(CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL)
#error "CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY requires CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL"
#endif

#if defined(CUDA3D_PML_ZFACE_SHARED_VP_DEBUG) && !defined(CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL)
#error "CUDA3D_PML_ZFACE_SHARED_VP_DEBUG requires CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL"
#endif

#if defined(CUDA3D_PML_ZFACE_SHARED_VP_DEBUG) && defined(CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY)
#error "CUDA3D_PML_ZFACE_SHARED_VP_DEBUG replaces the direct fused zface prototype; do not enable both"
#endif

#if defined(CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY) && defined(CUDA3D_PML_ZFACE_P_SPECIALIZE)
#error "CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY replaces the old pressure-only zface specialize path"
#endif

#if defined(CUDA3D_PML_ZFACE_SHARED_VP_DEBUG) && defined(CUDA3D_PML_ZFACE_P_SPECIALIZE)
#error "CUDA3D_PML_ZFACE_SHARED_VP_DEBUG replaces the old pressure-only zface specialize path"
#endif

__constant__ float c_ay_pml[CUDA3D_MAX_PML];
__constant__ float c_by_pml[CUDA3D_MAX_PML];
__constant__ float c_ax_pml[CUDA3D_MAX_PML];
__constant__ float c_bx_pml[CUDA3D_MAX_PML];
__constant__ float c_az_pml[CUDA3D_MAX_PML];
__constant__ float c_bz_pml[CUDA3D_MAX_PML];
__constant__ float c_ay_h_pml[CUDA3D_MAX_PML];
__constant__ float c_by_h_pml[CUDA3D_MAX_PML];
__constant__ float c_ax_h_pml[CUDA3D_MAX_PML];
__constant__ float c_bx_h_pml[CUDA3D_MAX_PML];
__constant__ float c_az_h_pml[CUDA3D_MAX_PML];
__constant__ float c_bz_h_pml[CUDA3D_MAX_PML];

void upload_pml_constants(int nbd,
			  const float *ay, const float *by, const float *ax, const float *bx, const float *az, const float *bz,
			  const float *ay_h, const float *by_h, const float *ax_h, const float *bx_h, const float *az_h, const float *bz_h){
  cudaMemcpyToSymbol(c_ay_pml, ay, nbd*sizeof(float));
  cudaMemcpyToSymbol(c_by_pml, by, nbd*sizeof(float));
  cudaMemcpyToSymbol(c_ax_pml, ax, nbd*sizeof(float));
  cudaMemcpyToSymbol(c_bx_pml, bx, nbd*sizeof(float));
  cudaMemcpyToSymbol(c_az_pml, az, nbd*sizeof(float));
  cudaMemcpyToSymbol(c_bz_pml, bz, nbd*sizeof(float));
  cudaMemcpyToSymbol(c_ay_h_pml, ay_h, nbd*sizeof(float));
  cudaMemcpyToSymbol(c_by_h_pml, by_h, nbd*sizeof(float));
  cudaMemcpyToSymbol(c_ax_h_pml, ax_h, nbd*sizeof(float));
  cudaMemcpyToSymbol(c_bx_h_pml, bx_h, nbd*sizeof(float));
  cudaMemcpyToSymbol(c_az_h_pml, az_h, nbd*sizeof(float));
  cudaMemcpyToSymbol(c_bz_h_pml, bz_h, nbd*sizeof(float));
}

#if defined(CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY) || defined(CUDA3D_PML_ZFACE_SHARED_VP_DEBUG)
__device__ __forceinline__ bool pml_fused_zface_pressure_point(int gtid1, int gtid2, int gtid3,
							       int n3, int n2, int n1, int npml) {
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;
  return ((gtid1 < npml) || (gtid1 >= n1 - npml)) &&
    (gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
    (gtid3 >= core3_lo) && (gtid3 < core3_hi);
}

__device__ __forceinline__ bool pml_fused_zface_vx_global_point(int gtid1, int gtid2, int gtid3,
								int n3, int n2, int n1, int npml) {
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;
  return ((gtid1 < npml) || (gtid1 >= n1 - npml)) &&
    (gtid2 >= core2_lo + 3) && (gtid2 < core2_hi - 4) &&
    (gtid3 >= core3_lo) && (gtid3 < core3_hi);
}

__device__ __forceinline__ bool pml_fused_zface_vy_global_point(int gtid1, int gtid2, int gtid3,
								int n3, int n2, int n1, int npml) {
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;
  return ((gtid1 < npml) || (gtid1 >= n1 - npml)) &&
    (gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
    (gtid3 >= core3_lo + 3) && (gtid3 < core3_hi - 4);
}
#endif

//< step forward: 3-D FD, order=8 >
// mod from pengliang yang
__global__ void cuda_fd3d_v_pml(float *p1, float *vy, float *vx, float *vz,
				float _dy2, float _dx2, float _dz2, 
				int n3, int n2, int n1, int npml, float dt,
				float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
				float *mem_dy, float *mem_dx, float *mem_dz){
  float c1, c2, c3;
  bool validr = true;
  bool validw = true;
  int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;
  int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;
  const int ltid1 = threadIdx.x;
  const int ltid2 = threadIdx.y;
  const int work1 = blockDim.x;
  const int work2 = blockDim.y;
  __shared__ float tile[BlockSize2 + 2 * radius][BlockSize1 + 2 * radius];

  float infront[radius];
  float behind[radius];
  float current;

  size_t inIndex = 0;
  size_t outIndex = 0;
  size_t ic, pind;

  const int lt1 = ltid1 + radius;
  const int lt2 = ltid2 + radius;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  int i, i3;

  // Advance inputIndex to start of inner volume, the begining of the area
  inIndex += radius * stride2 + radius;

    // Advance inputIndex to target element
  //  inIndex += gtid2 * stride2 + gtid1;

  // Check in bounds
  //  while (gtid1 < n1+radius && gtid2 < n2+radius){
  if ((gtid1 >= n1 + radius) ||(gtid2 >= n2 + radius)) validr = false;
  if ((gtid1 >= n1) || (gtid2 >= n2)) validw = false;

  //    // Advance inputIndex to target element
  inIndex += gtid2 * stride2 + gtid1;

    // Preload the "infront" and "behind" data
    for (i = radius - 2 ; i >= 0 ; i--){
      if (validr) behind[i] = p1[inIndex];
      inIndex += stride3;
    }

    if (validr)	current = p1[inIndex];

    outIndex = inIndex;
    inIndex += stride3;

    for (i = 0 ; i < radius ; i++){
      if (validr) infront[i] = p1[inIndex];
      inIndex += stride3;
    }

    // Step through the zx-planes
#pragma unroll 9
    for (i3 = 0 ; i3 < n3 ; i3++){
      // Advance the slice (move the thread-front)
      for (i = radius - 1 ; i > 0 ; i--) 
	behind[i] = behind[i - 1];

      behind[0] = current;
      current = infront[0];
#pragma unroll 4
      for (i = 0 ; i < radius - 1 ; i++) 
	infront[i] = infront[i + 1];
      
      if (validr) 
	infront[radius - 1] = p1[inIndex];
    
      inIndex += stride3; // needed???
      outIndex += stride3;
      __syncthreads();
      
      // Update the data slice in the local tile
      // Halo above & below // why not 9 tile initiated???
      if (ltid2 < radius){
	tile[ltid2][lt1]                  = p1[outIndex - radius * stride2];
	tile[ltid2 + work2 + radius][lt1] = p1[outIndex + work2 * stride2]; // not might out of bound????
	//	ic=(int)(MIN(gtid2+work2+radius, n2+2*radius));
	//	tile[ltid2 + work2 + radius][lt1] = p1[ic*stride2+gtid1+radius];
      }
      // Halo left & right
      if (ltid1 < radius){
	tile[lt2][ltid1]                  = p1[outIndex - radius];
	tile[lt2][ltid1 + work1 + radius] = p1[outIndex + work1];
      }

      tile[lt2][lt1] = current;
      __syncthreads();

      // Compute the output value
      c1=c2=c3=0.0;

      c1=stencil[1]*(tile[lt2][lt1+1]-tile[lt2][lt1])
	+stencil[2]*(tile[lt2][lt1+2]-tile[lt2][lt1-1])
	+stencil[3]*(tile[lt2][lt1+3]-tile[lt2][lt1-2])
	+stencil[4]*(tile[lt2][lt1+4]-tile[lt2][lt1-3]);

      c2=stencil[1]*(tile[lt2+1][lt1]-tile[lt2][lt1])
	+stencil[2]*(tile[lt2+2][lt1]-tile[lt2-1][lt1])
	+stencil[3]*(tile[lt2+3][lt1]-tile[lt2-2][lt1])
	+stencil[4]*(tile[lt2+4][lt1]-tile[lt2-3][lt1]);

      c3=stencil[1]*(infront[0]-current)
	+stencil[2]*(infront[1]-behind[0])
	+stencil[3]*(infront[2]-behind[1])
	+stencil[4]*(infront[3]-behind[2]);
      c1*=_dz2;
      c2*=_dx2;
      c3*=_dy2;
    //if (validw) {vz[outIndex]+=(-dt*c1);vx[outIndex]+=(-dt*c2);vy[outIndex]+=(-dt*c3);}

      if (validw) {
	vz[outIndex]=c1;
	vx[outIndex]=c2;
	vy[outIndex]=c3;
		
	//PML Zone
	//Start Z-PML
	if(gtid1<npml) {
	  //Apply PML in Z-direction, pind is index inside PML zone
	  pind=i3*n2*npml + gtid2*npml + gtid1;
	  mem_dz[pind]=mem_dz[pind]*bz_h[gtid1]+c1*(bz_h[gtid1]-1.);
	  vz[outIndex]+=mem_dz[pind];
	}
	if (gtid1>=n1-npml) {
	  //Apply PML in Z-direction, pind is index inside PML zone
	  ic=gtid1-n1+npml;
	  pind=  n3*n2*npml+i3*n2*npml + gtid2*npml + ic;
	  mem_dz[pind]=mem_dz[pind]*az_h[ic]+c1*(az_h[ic]-1.);
	  vz[outIndex]+=mem_dz[pind];
	}
	//End ZPML
	
	//Start X-PML
	if (gtid2<npml){
	  //Apply PML in X-direction, pind is index inside PML zone
	  pind=i3*npml*n1 + gtid2*n1 + gtid1;
	  mem_dx[pind]=mem_dx[pind]*bx_h[gtid2]+c2*(bx_h[gtid2]-1.);
	  vx[outIndex]+=mem_dx[pind];
	}
	if (gtid2>=n2-npml){
	  //Apply PML in X-direction, pind is index inside PML zone
	  ic=gtid2-n2+npml;
	  pind=n3*npml*n1+i3*(npml*n1) + ic*n1 + gtid1;
	  mem_dx[pind]=mem_dx[pind]*ax_h[ic]+c2*(ax_h[ic]-1.);
	  vx[outIndex]+=mem_dx[pind];
	}
	//End XPML
	
	// Start Y-PML
	if(i3<npml) {
	  //Apply PML in Y-direction, pind is index inside PML zone
	  pind=i3*n2*n1 + gtid2*n1 + gtid1;
	  mem_dy[pind]=mem_dy[pind]*by_h[i3]+c3*(by_h[i3]-1.);
	  vy[outIndex]+=mem_dy[pind];
	}
	if (i3>=n3-npml){
	  //Apply PML in Y-direction, pind is index inside PML zone
	  ic=i3-n3+npml;
	  pind= npml*n2*n1+ic*n2*n1 + gtid2*(n1) + gtid1;
	  mem_dy[pind]=mem_dy[pind]*ay_h[ic]+c3*(ay_h[ic]-1.);
	  vy[outIndex]+=mem_dy[pind];
	}
	//END YPML
      } // end valid domain
    }//end step through xz plane/ y direction
    //    gtid1 =gtid1+ blockDim.x * gridDim.x;  //???
    //    gtid2 =gtid2+ blockDim.y * gridDim.y;/// ??
    //  }
}

/*< step forward: 3-D FD, order=8 >*/
__global__ void cuda_fd3d_p_pml(float *p0, float *p1, float *vy, float *vx, float *vz,
				float *vyy, float *vxx, float *vzz,
				float *cw2, float _dy2, float _dx2, float _dz2, 
				int n3, int n2, int n1, int npml, float dt, 
				float *ay, float *by, float *ax, float *bx, float *az, float *bz,
				float *mem_dyy, float *mem_dxx, float *mem_dzz){
  float c1, c2, c3;
  bool validr = true;
  bool validw = true;
  int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;
  int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;
  const int ltid1 = threadIdx.x;
  const int ltid2 = threadIdx.y;
  const int work1 = blockDim.x;
  const int work2 = blockDim.y;
  __shared__ float tile1[BlockSize2 + 2 * radius][BlockSize1 + 2 * radius];
  __shared__ float tile2[BlockSize2 + 2 * radius][BlockSize1 + 2 * radius];
  // comment by zz    __shared__ float tile3[BlockSize2 + 2 * radius][BlockSize1 + 2 * radius];

  float infront1[radius];float infront2[radius];float infront3[radius];
  float behind1[radius];float behind2[radius];float behind3[radius];
  float current1;float current2;float current3;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);

  size_t inIndex = 0;
  size_t outIndex = 0;
  size_t ic, pind;
  int i, i3;


  const int lt1 = ltid1 + radius;
  const int lt2 = ltid2 + radius;

  // Advance inputIndex to start of inner volume that skips the radius region
  inIndex += radius * stride2 + radius;

  // Check in bounds
  //  while ( gtid1 <n1+radius && gtid2 < n2+radius ){
  if ((gtid1 >= n1 + radius) ||(gtid2 >= n2 + radius)) validr = false;
  if ((gtid1 >= n1) || (gtid2 >= n2)) validw = false;

    // Advance inputIndex to target element /global target location
    inIndex += gtid2 * stride2 + gtid1;

    // Preload the "infront" and "behind" data
    for (i = radius - 2 ; i >= 0 ; i--){
      if (validr) {
	behind1[i] = vz[inIndex];
	behind2[i] = vx[inIndex];
	behind3[i] = vy[inIndex];
      }
      inIndex += stride3;
    }

    if (validr) {
      current1 = vz[inIndex];
      current2 = vx[inIndex];
      current3 = vy[inIndex];
    }
    
    outIndex = inIndex;
    inIndex += stride3;
    
    for (i = 0 ; i < radius ; i++){
      if (validr) {
	infront1[i] = vz[inIndex];
	infront2[i] = vx[inIndex];
	infront3[i] = vy[inIndex];
      }
      inIndex += stride3;
    }

    // Step through the zx-planes
#pragma unroll 9
    for (i3 = 0 ; i3 < n3 ; i3++){
      // Advance the slice (move the thread-front)
      for (i = radius - 1 ; i > 0 ; i--) {
	behind1[i] = behind1[i - 1];
	behind2[i] = behind2[i - 1];
	behind3[i] = behind3[i - 1];
      }

      behind1[0] = current1;
      behind2[0] = current2;
      behind3[0] = current3;

      current1 = infront1[0];
      current2 = infront2[0];
      current3 = infront3[0];
#pragma unroll 4
      for (i = 0 ; i < radius - 1 ; i++) {
	infront1[i] = infront1[i + 1];
	infront2[i] = infront2[i + 1];
	infront3[i] = infront3[i + 1];
      }

      if (validr) {
	infront1[radius - 1] = vz[inIndex];
	infront2[radius - 1] = vx[inIndex];
	infront3[radius - 1] = vy[inIndex];
      }

      inIndex += stride3;
      outIndex += stride3;
      __syncthreads();

      // Update the data slice in the local tile
      // Halo above & below
      if (ltid2 < radius){
	tile1[ltid2][lt1]                  = vz[outIndex - radius * stride2];
	//	ic=(int)(MIN(gtid2+work2+radius, n2+2*radius));
	tile1[ltid2 + work2 + radius][lt1] = vz[outIndex + work2 * stride2]; // outIndex might need change

	tile2[ltid2][lt1]                  = vx[outIndex - radius * stride2];
	tile2[ltid2 + work2 + radius][lt1] = vx[outIndex + work2 * stride2];
      }

      // Halo left & right
      if (ltid1 < radius){
	tile1[lt2][ltid1]                  = vz[outIndex - radius];
	tile1[lt2][ltid1 + work1 + radius] = vz[outIndex + work1];

	tile2[lt2][ltid1]                  = vx[outIndex - radius];
	tile2[lt2][ltid1 + work1 + radius] = vx[outIndex + work1];
      }

      tile1[lt2][lt1] = current1; 
      tile2[lt2][lt1] = current2; //[t2][t1] = current3;
      __syncthreads();

      // Compute the output value
  
      //c1=stencil[0]*current1;c2=stencil[0]*current2;c3=stencil[0]*current3;
      c1=c2=c3=0.0;

      c1=stencil[1]*(tile1[lt2][lt1]-tile1[lt2][lt1-1])
	+stencil[2]*(tile1[lt2][lt1+1]-tile1[lt2][lt1-2])
	+stencil[3]*(tile1[lt2][lt1+2]-tile1[lt2][lt1-3])
	+stencil[4]*(tile1[lt2][lt1+3]-tile1[lt2][lt1-4]);

      c2=stencil[1]*(tile2[lt2][lt1]-tile2[lt2-1][lt1])
	+stencil[2]*(tile2[lt2+1][lt1]-tile2[lt2-2][lt1])
	+stencil[3]*(tile2[lt2+2][lt1]-tile2[lt2-3][lt1])
	+stencil[4]*(tile2[lt2+3][lt1]-tile2[lt2-4][lt1]);

      c3=stencil[1]*(current3-behind3[0])
	+stencil[2]*(infront3[0]-behind3[1])
	+stencil[3]*(infront3[1]-behind3[2])
	+stencil[4]*(infront3[2]-behind3[3]);
      c1*=_dz2;
      c2*=_dx2;
      c3*=_dy2;


      //if (validw) {p0[outIndex]=p1[outIndex]-vel[outIndex]*(c1+c2+c3);}//{p0[outIndex]=2*p1[outIndex]-p0[outIndex]-vel[outIndex]*(c1+c2+c3);}
      //right
      if (validw) {
	vzz[outIndex]=c1;
	vxx[outIndex]=c2;
	vyy[outIndex]=c3;

	
	//Start Z-PML
	if(gtid1<npml) { 
	  //Apply PML in Z-direction  //pind is index inside PML zone
	  pind=i3*npml*n2 + gtid2*npml + gtid1;
	  mem_dzz[pind]=mem_dzz[pind]*bz[gtid1]+c1*(bz[gtid1]-1.);
	  vzz[outIndex]+=mem_dzz[pind];
	}
	if (gtid1>=n1-npml) {
	  //Apply PML in Z-direction	  //pind is index inside PML zone
	  ic=gtid1-n1+npml;
	  pind=  n3*n2*npml+i3*npml*n2 + gtid2*npml + ic;
	  mem_dzz[pind]=mem_dzz[pind]*az[ic]+c1*(az[ic]-1.);
	  vzz[outIndex]+=mem_dzz[pind];
	}
	//End ZPML

	//Start X-PML
	if (gtid2<npml){
	  //Apply PML in X-direction	  //pind is index inside PML zone
	  pind=i3*npml*n1 + gtid2*n1 + gtid1;
	  mem_dxx[pind]=mem_dxx[pind]*bx[gtid2]+c2*(bx[gtid2]-1.);
	  vxx[outIndex]+=mem_dxx[pind];
	}
	if (gtid2>=n2-npml){
	  //Apply PML in X-direction	  //pind is index inside PML zone
	  ic=gtid2-n2+npml;
	  pind=n3*npml*n1+i3*(npml*n1) + ic*n1 + gtid1;
	  mem_dxx[pind]=mem_dxx[pind]*ax[ic]+c2*(ax[ic]-1.);
	  vxx[outIndex]+=mem_dxx[pind];
	}
	//End XPML

	// Start Y-PML
	if(i3<npml) {
	  //Apply PML in Y-direction	  //pind is index inside PML zone
	  pind=i3*n2*n1 + gtid2*n1 + gtid1;
	  mem_dyy[pind]=mem_dyy[pind]*by[i3]+c3*(by[i3]-1.);
	  vyy[outIndex]+=mem_dyy[pind];
	}
	if (i3>=n3-npml){
	  //Apply PML in Y-direction	  //pind is index inside PML zone
	  ic=i3-n3+npml;
	  pind= npml*n2*n1+ic*n2*n1 + gtid2*(n1) + gtid1;
	  mem_dyy[pind]=mem_dyy[pind]*ay[ic]+c3*(ay[ic]-1.);
	  vyy[outIndex]+=mem_dyy[pind];
	}
	//END YPML
	
	p0[outIndex]=2*p1[outIndex]-p0[outIndex]
	  +cw2[outIndex]*dt*(vzz[outIndex]+vxx[outIndex]+vyy[outIndex]);

      }// validw end

    }// end through xz plane
    //    gtid1 =gtid1+ blockDim.x * gridDim.x; //???
    //    gtid2 =gtid2+ blockDim.y * gridDim.y; //???
    //  }
}

__global__ void lint3d_inject_bell_gpu(float *d_u, int nbd, int yl, int xl, int it, int snum,
				       float *src, float *d_bell, int nbell,
				       int indexy, int indexx, int indexz,
				       int ny, int nx, int nz,
				       float *d_sw000, float *d_sw001, float *d_sw010, float *d_sw011, 
				       float *d_sw100, float *d_sw101, float *d_sw110, float *d_sw111){
  int iz = threadIdx.x + blockIdx.x * blockDim.x; // 0 to 2*nbell + 1
  int ix = threadIdx.y + blockIdx.y * blockDim.y; // 0 to 2*nbell + 1   
  int iy = threadIdx.z + blockIdx.z * blockDim.z; // 0 to 2*nbell + 1   
  //  int ia = blockIdx.x; // ????
  size_t nxz;
  int indx1, indx2, indx3;
  float wa000, wa001, wa010, wa011, wa100, wa101, wa110, wa111, aaa;
  nxz=nx*nz;
  // comment by zz    int haloCorrection = 0; // GPU 0 does not have any additional halo cells in stress/acceleration arrays

    
  //  for (int iy = 0; iy < 2*nbell + 1; iy++){
    //    float wa = d_ww[it] * d_bell[(iy * (2*nbell+1) * (2*nbell+1)) + (ix * (2*nbell+1)) + iz];
    aaa=src[it]*d_bell[iy*(2*nbell+1)*(2*nbell+1)+ix*(2*nbell+1)+iz];
            
    wa000 = (aaa * d_sw000[snum]); //ia???
    wa001 = (aaa * d_sw001[snum]);
    wa010 = (aaa * d_sw010[snum]);
    wa011 = (aaa * d_sw011[snum]);

    wa100 = (aaa * d_sw100[snum]);
    wa101 = (aaa * d_sw101[snum]);
    wa110 = (aaa * d_sw110[snum]);
    wa111 = (aaa * d_sw111[snum]);

    indx1=radius+nbd+indexz+iz;
    indx2=radius+nbd+indexx+ix-xl;
    indx3=radius+nbd+indexy+iy-yl;

    atomicAdd(&d_u[(indx3  -nbell)*nxz+(indx2  -nbell)*nz+indx1  -nbell], wa000);
    atomicAdd(&d_u[(indx3  -nbell)*nxz+(indx2  -nbell)*nz+indx1+1-nbell], wa001);
    atomicAdd(&d_u[(indx3  -nbell)*nxz+(indx2+1-nbell)*nz+indx1  -nbell], wa010);
    atomicAdd(&d_u[(indx3  -nbell)*nxz+(indx2+1-nbell)*nz+indx1+1-nbell], wa011);
    atomicAdd(&d_u[(indx3+1-nbell)*nxz+(indx2  -nbell)*nz+indx1  -nbell], wa100);
    atomicAdd(&d_u[(indx3+1-nbell)*nxz+(indx2  -nbell)*nz+indx1+1-nbell], wa101);
    atomicAdd(&d_u[(indx3+1-nbell)*nxz+(indx2+1-nbell)*nz+indx1  -nbell], wa110);
    atomicAdd(&d_u[(indx3+1-nbell)*nxz+(indx2+1-nbell)*nz+indx1+1-nbell], wa111);

    //    d_u[(indx3  )*nx*nz+(indx2  )*nz+indx1  ]+=wa000;
    //    d_u[(indx3  )*nx*nz+(indx2  )*nz+indx1+1]+=wa001;
    //    d_u[(indx3  )*nx*nz+(indx2+1)*nz+indx1  ]+=wa010;
    //    d_u[(indx3  )*nx*nz+(indx2+1)*nz+indx1+1]+=wa011;
    //    d_u[(indx3+1)*nx*nz+(indx2  )*nz+indx1  ]+=wa100;
    //    d_u[(indx3+1)*nx*nz+(indx2  )*nz+indx1+1]+=wa101;
    //    d_u[(indx3+1)*nx*nz+(indx2+1)*nz+indx1  ]+=wa110;
    //    d_u[(indx3+1)*nx*nz+(indx2+1)*nz+indx1+1]+=wa111;
                                                                  
    //    atomicAdd(&d_uu[((d_jy[ia]  - nbell) + iy ) * nxpad * nzpad + ((d_jx[ia] - nbell) + ix    )*nzpad + ((d_jz[ia] - nbell) + iz    )], wa000);
    //    atomicAdd(&d_uu[((d_jy[ia]  - nbell) + iy ) * nxpad * nzpad + ((d_jx[ia] - nbell) + ix    )*nzpad + ((d_jz[ia] - nbell) + iz + 1)], wa001);
    //    atomicAdd(&d_uu[((d_jy[ia]  - nbell) + iy ) * nxpad * nzpad + ((d_jx[ia] - nbell) + ix + 1)*nzpad + ((d_jz[ia] - nbell) + iz    )], wa010);
    //    atomicAdd(&d_uu[((d_jy[ia]  - nbell) + iy ) * nxpad * nzpad + ((d_jx[ia] - nbell) + ix + 1)*nzpad + ((d_jz[ia] - nbell) + iz + 1)], wa011);
    //  }
}

__global__ void lint3d_extract_gpu_zz(float *din, int nbd, int min_all_y, int min_all_x,
				      int it, int nt, float *dout, int *rec0_indx,
				      size_t nr, int ny, int nx, int nz,
				      float *rw000, float *rw001, float *rw010, float *rw011, 
				      float *rw100, float *rw101, float *rw110, float *rw111){
  size_t ir = threadIdx.x + blockIdx.x * blockDim.x;
  size_t i1, i2, i3;
  size_t nxz;
  nxz=nx*nz;
  if (ir < nr){
    i1=nbd+radius + rec0_indx[ir*3+2];
    i2=nbd+radius + rec0_indx[ir*3+1]-min_all_x;
    i3=nbd+radius + rec0_indx[ir*3  ]-min_all_y;
    //    dout[ir*nt+it]=
    dout[it*nr+ir]= // note in 3D ir is the fast direction
      din[(i3  )*nxz+(i2  )*nz+i1  ]*rw000[ir]+
      din[(i3  )*nxz+(i2  )*nz+i1+1]*rw001[ir]+
      din[(i3  )*nxz+(i2+1)*nz+i1  ]*rw010[ir]+
      din[(i3  )*nxz+(i2+1)*nz+i1+1]*rw011[ir]+
      din[(i3+1)*nxz+(i2  )*nz+i1  ]*rw100[ir]+
      din[(i3+1)*nxz+(i2  )*nz+i1+1]*rw101[ir]+
      din[(i3+1)*nxz+(i2+1)*nz+i1  ]*rw110[ir]+
      din[(i3+1)*nxz+(i2+1)*nz+i1+1]*rw111[ir];
  }
}

__global__ void lint3d_inject_bell_extract_gpu_zz(float *d_u, int nbd, int yl, int xl,
				      int it, int nt, int snum,
				      float *src, float *d_bell, int nbell,
				      int indexy, int indexx, int indexz,
				      int ny, int nx, int nz,
				      float *d_sw000, float *d_sw001, float *d_sw010, float *d_sw011,
				      float *d_sw100, float *d_sw101, float *d_sw110, float *d_sw111,
				      float *dout, int *rec0_indx, size_t nr,
				      float *rw000, float *rw001, float *rw010, float *rw011,
				      float *rw100, float *rw101, float *rw110, float *rw111){
  size_t tid = threadIdx.x;
  size_t nxz = nx * nz;
  int side = 2 * nbell + 1;
  int bell_size = side * side * side;

  if (tid < (size_t)bell_size) {
    int iy = tid / (side * side);
    int ix = (tid / side) % side;
    int iz = tid % side;
    float aaa = src[it] * d_bell[iy * side * side + ix * side + iz];

    float wa000 = aaa * d_sw000[snum];
    float wa001 = aaa * d_sw001[snum];
    float wa010 = aaa * d_sw010[snum];
    float wa011 = aaa * d_sw011[snum];
    float wa100 = aaa * d_sw100[snum];
    float wa101 = aaa * d_sw101[snum];
    float wa110 = aaa * d_sw110[snum];
    float wa111 = aaa * d_sw111[snum];

    int indx1 = radius + nbd + indexz + iz;
    int indx2 = radius + nbd + indexx + ix - xl;
    int indx3 = radius + nbd + indexy + iy - yl;

    atomicAdd(&d_u[(indx3  -nbell)*nxz+(indx2  -nbell)*nz+indx1  -nbell], wa000);
    atomicAdd(&d_u[(indx3  -nbell)*nxz+(indx2  -nbell)*nz+indx1+1-nbell], wa001);
    atomicAdd(&d_u[(indx3  -nbell)*nxz+(indx2+1-nbell)*nz+indx1  -nbell], wa010);
    atomicAdd(&d_u[(indx3  -nbell)*nxz+(indx2+1-nbell)*nz+indx1+1-nbell], wa011);
    atomicAdd(&d_u[(indx3+1-nbell)*nxz+(indx2  -nbell)*nz+indx1  -nbell], wa100);
    atomicAdd(&d_u[(indx3+1-nbell)*nxz+(indx2  -nbell)*nz+indx1+1-nbell], wa101);
    atomicAdd(&d_u[(indx3+1-nbell)*nxz+(indx2+1-nbell)*nz+indx1  -nbell], wa110);
    atomicAdd(&d_u[(indx3+1-nbell)*nxz+(indx2+1-nbell)*nz+indx1+1-nbell], wa111);
  }

  __syncthreads();

  if (tid < nr) {
    size_t i1 = nbd + radius + rec0_indx[tid * 3 + 2];
    size_t i2 = nbd + radius + rec0_indx[tid * 3 + 1] - xl;
    size_t i3 = nbd + radius + rec0_indx[tid * 3] - yl;

    dout[it * nr + tid] =
      d_u[(i3  )*nxz+(i2  )*nz+i1  ]*rw000[tid]+
      d_u[(i3  )*nxz+(i2  )*nz+i1+1]*rw001[tid]+
      d_u[(i3  )*nxz+(i2+1)*nz+i1  ]*rw010[tid]+
      d_u[(i3  )*nxz+(i2+1)*nz+i1+1]*rw011[tid]+
      d_u[(i3+1)*nxz+(i2  )*nz+i1  ]*rw100[tid]+
      d_u[(i3+1)*nxz+(i2  )*nz+i1+1]*rw101[tid]+
      d_u[(i3+1)*nxz+(i2+1)*nz+i1  ]*rw110[tid]+
      d_u[(i3+1)*nxz+(i2+1)*nz+i1+1]*rw111[tid];
  }
}

/*
__global__ void cuda_p_extract_3d(float *p0, float *pu, int itc,
				  int n3, int n2, int n1, int npml, int yl, int xl){
  int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;
  int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;
  size_t nsize, n21, nn21, id, i3;
  const int npad=npml+radius;
  const int nn1=n1+2*npad;
  const int nn2=n2+2*npad;
  nsize=itc*n1*n2*n3;
  n21=n2*n1;
  nn21=nn2*nn1;
  //  const int stride2 = n1 + 2 * (radius+npml);
  //  const int stride3 = stride2 * (n2 + 2 * radius);

  // Advance inputIndex to start of inner volume that skip the radius
  //  id += radius * stride2 + radius;
  for (i3=0; i3<n3; i3++){
  id=0;
    while( gtid1 < n1 && gtid2 < n2 ){ 
      // Advance inputIndex to target element/ global target location
      id += i3*nn21+ (gtid2+npad) * nn1 + gtid1+npad;
      
      pu[nsize+i3*n21+gtid2*n1+gtid1]=p0[id];

      gtid1+=blockDim.x*gridDim.x;
      gtid2+=blockDim.y*gridDim.y;
    }
  }
}
*/

__global__ void cuda_p_extract_3d(float *p0, float *pu, size_t itc,
				  size_t n3, size_t n2, size_t n1, int npml, int yl, int xl){
  size_t gtid1 = blockIdx.x * blockDim.x + threadIdx.x; // nz
  size_t gtid2 = blockIdx.y * blockDim.y + threadIdx.y; // nx
  size_t gtid3 = blockIdx.z * blockDim.z + threadIdx.z; //ny
  size_t nsize, n21, nn21, id, i3;
  const int npad=npml+radius;
  const int nn1=n1+2*npad;
  const int nn2=n2+2*npad;
  nsize=itc*n1*n2*n3;
  n21=n2*n1;
  nn21=nn2*nn1;
  //  const int stride2 = n1 + 2 * (radius+npml);
  //  const int stride3 = stride2 * (n2 + 2 * radius);

  //  const int stride2 = n1 + 2 * radius;
  //  const int stride3 = stride2 * (n2 + 2 * radius);
  //  inIndex += radius * stride2 + radius + radius*stride3;

  // Advance inputIndex to start of inner volume that skip the radius
  //  id += radius * stride2 + radius;
  //  for (i3=0; i3<n3; i3++){
  id=0;
  while( gtid1 < n1 && gtid2 < n2 && gtid3 < n3){ 
    // Advance inputIndex to target element/ global target location
    id += (gtid3+npad)*nn21+ (gtid2+npad) * nn1 + gtid1+npad;
      
    pu[nsize+gtid3*n21+gtid2*n1+gtid1]=p0[id];

    gtid1+=blockDim.x*gridDim.x;
    gtid2+=blockDim.y*gridDim.y;
    gtid3+=blockDim.z*gridDim.z;
  }
  //}
}

__global__ void cuda_get_misfit_l2(float *d_obs, float *d_est, float *d_wb, float *d_adj,
				   float *tmut, float *bmut, int nr, int nt,
				   float sht_scl, int snum){
  int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;
  int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;
  //  const int npad=npml+radius;
  //  const int nn1=n1+2*npad;
  float a2, b2, c, delta, dis, tmp;
  int ic;
  delta=0.0;

  //  sum=0.0;
  while( gtid2<nr && gtid1<=(int)(bmut[gtid2]) && gtid1>=(int)(tmut[gtid2]) ){ 
    ic=gtid2*nt+gtid1;
    d_adj[ic]=d_obs[ic];
    gtid1+=blockDim.x*gridDim.x;
    gtid2+=blockDim.y*gridDim.y;
  }
}

__global__ void cuda_vector_mult(float *d1, float *d2, size_t n, float *out){
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  while (tid < n){
    out[tid]=(d1[tid]*d2[tid]);
    tid+=blockDim.x*gridDim.x;
  }
}

////
__global__ void lint3d_inject_rec_gpu(float *d_u, int nbd, int yl, int xl, int it, int nt,
				      float *d_d, int *rec0_indx, int nr,
				      int ny, int nx, int nz,
				      float *rw000, float *rw001, float *rw010, float *rw011, 
				      float *rw100, float *rw101, float *rw110, float *rw111){
  size_t ir=threadIdx.x+blockIdx.x*blockDim.x;
  //  int iz = threadIdx.x; // 0 to 2*nbell + 1
  //  int ix = threadIdx.y; // 0 to 2*nbell + 1   
  //  int ia = blockIdx.x; // ????
  size_t i3, i2, i1, ic, n21;
  int indx1, indx2, indx3, npad;
  float wa000, wa001, wa010, wa011, wa100, wa101, wa110, wa111, aaa;
  
  // comment by zz    int haloCorrection = 0; // GPU 0 does not have any additional halo cells in stress/acceleration arrays
  n21=nx*nz;
  npad=nbd+radius;
  while (ir < nr){
    ic=it*nr+ir;
    i1=npad+rec0_indx[ir*3+2];
    i2=npad+rec0_indx[ir*3+1]-xl;
    i3=npad+rec0_indx[ir*3  ]-yl;
            
    wa000 = (d_d[ic] * rw000[ir]);
    wa001 = (d_d[ic] * rw001[ir]);
    wa010 = (d_d[ic] * rw010[ir]);
    wa011 = (d_d[ic] * rw011[ir]);
    wa100 = (d_d[ic] * rw100[ir]);
    wa101 = (d_d[ic] * rw101[ir]);
    wa110 = (d_d[ic] * rw110[ir]);
    wa111 = (d_d[ic] * rw111[ir]);

    atomicAdd(&d_u[(i3  )*n21+(i2  )*nz+i1  ], wa000);
    atomicAdd(&d_u[(i3  )*n21+(i2  )*nz+i1+1], wa001);
    atomicAdd(&d_u[(i3  )*n21+(i2+1)*nz+i1  ], wa010);
    atomicAdd(&d_u[(i3  )*n21+(i2+1)*nz+i1+1], wa011);
    atomicAdd(&d_u[(i3+1)*n21+(i2  )*nz+i1  ], wa100);
    atomicAdd(&d_u[(i3+1)*n21+(i2  )*nz+i1+1], wa101);
    atomicAdd(&d_u[(i3+1)*n21+(i2+1)*nz+i1  ], wa110);
    atomicAdd(&d_u[(i3+1)*n21+(i2+1)*nz+i1+1], wa111);
    //    atomicAdd(&d_u[(indx3  -nbell)*nxz+(indx2  -nbell)*nz+indx1  -nbell], wa000);

    ir+=blockDim.x*gridDim.x;
  }
}

////
__global__ void lint3d_bell_gpu(int it, int nc, int ns, int c, 
				int nbell, int nxpad,  int nzpad, 
				float *d_uu, float *d_bell, 
				int *d_jx, int *d_jz, int *d_jy, float *d_ww, 
				float *d_Sw000, float *d_Sw001, float *d_Sw010, float *d_Sw011, 
				float *d_Sw100, float *d_Sw101, float *d_Sw110, float *d_Sw111)
{ 
  int ix = threadIdx.x; // 0 to 2*nbell + 1
  int iy = threadIdx.y; // 0 to 2*nbell + 1   
  int ia = blockIdx.x; // ????
    
  // comment by zz    int haloCorrection = 0; // GPU 0 does not have any additional halo cells in stress/acceleration arrays
    
  for (int iz = 0; iz < 2*nbell + 1; iz++){
    float wa = d_ww[it] * d_bell[(iy * (2*nbell+1) * (2*nbell+1)) + (ix * (2*nbell+1)) + iz];
            
    float wa000 = -(wa * d_Sw000[ia]);
    float wa001 = -(wa * d_Sw001[ia]);
    float wa010 = -(wa * d_Sw010[ia]);
    float wa011 = -(wa * d_Sw011[ia]);
                                                                  
    atomicAdd(&d_uu[((d_jy[ia]  - nbell) + iy ) * nxpad * nzpad + ((d_jx[ia] - nbell) + ix    )*nzpad + ((d_jz[ia] - nbell) + iz    )], wa000);
    atomicAdd(&d_uu[((d_jy[ia]  - nbell) + iy ) * nxpad * nzpad + ((d_jx[ia] - nbell) + ix    )*nzpad + ((d_jz[ia] - nbell) + iz + 1)], wa001);
    atomicAdd(&d_uu[((d_jy[ia]  - nbell) + iy ) * nxpad * nzpad + ((d_jx[ia] - nbell) + ix + 1)*nzpad + ((d_jz[ia] - nbell) + iz    )], wa010);
    atomicAdd(&d_uu[((d_jy[ia]  - nbell) + iy ) * nxpad * nzpad + ((d_jx[ia] - nbell) + ix + 1)*nzpad + ((d_jz[ia] - nbell) + iz + 1)], wa011);
  }  
}


__global__ void lint3d_extract_gpu(int gpuID, float *d_dd, 
				   int nr, int nxpad,  int nzpad, 
				   float *d_uoz, int *d_Rjz, int *d_Rjx, int *d_Rjy, 
				   float *d_Rw000, float *d_Rw001, float *d_Rw010, float *d_Rw011, 
				   float *d_Rw100, float *d_Rw101, float *d_Rw110, float *d_Rw111)
{
  int rr = threadIdx.x + blockIdx.x * blockDim.x;   // rr = the receiver this thread is extracting data for	
  // comment by zz	int haloCorrection = 0;	// GPU 0 does not have any additional halo cells in stress/acceleration arrays
	
  if (rr < nr){
    d_dd[rr] =	d_uoz[(d_Rjy[rr] + 0) * nxpad * nzpad + (d_Rjx[rr] + 0)*nzpad + (d_Rjz[rr] + 0)  ]  * d_Rw000[rr] +
      d_uoz[(d_Rjy[rr] + 0) * nxpad * nzpad + (d_Rjx[rr] + 0)* nzpad  + (d_Rjz[rr] + 1) ]  * d_Rw001[rr] +
      d_uoz[(d_Rjy[rr] + 0) * nxpad * nzpad + (d_Rjx[rr] + 1) * nzpad  + (d_Rjz[rr] + 0) ]  * d_Rw010[rr] +
      d_uoz[(d_Rjy[rr] + 0) * nxpad * nzpad + (d_Rjx[rr] + 1)* nzpad  + (d_Rjz[rr] + 1)]  * d_Rw011[rr] +
      d_uoz[(d_Rjy[rr] + 1) * nxpad * nzpad + (d_Rjx[rr] + 0)* nzpad  + (d_Rjz[rr] + 0) ]  * d_Rw100[rr] +
      d_uoz[(d_Rjy[rr] + 1) * nxpad * nzpad + (d_Rjx[rr] + 0)* nzpad  + (d_Rjz[rr] + 1)]  * d_Rw101[rr] +
      d_uoz[(d_Rjy[rr] + 1) * nxpad * nzpad + (d_Rjx[rr] + 1)* nzpad  + (d_Rjz[rr] + 0)]  * d_Rw110[rr] +
      d_uoz[(d_Rjy[rr] + 1) * nxpad * nzpad + (d_Rjx[rr] + 1)* nzpad  + (d_Rjz[rr] + 1) ]  * d_Rw111[rr];
  }
}

__global__ void cuda_fd3d_v_pml_ns(const float *__restrict__ p1, float *vy, float *vx, float *vz,
				   float _dy2, float _dx2, float _dz2,
				   int n3, int n2, int n1, int npml, float dt,
				   float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
				   const float *__restrict__ mem_dy, const float *__restrict__ mem_dx, const float *__restrict__ mem_dz,
				   float *mem_dy_next, float *mem_dx_next, float *mem_dz_next){
#else
				   float *mem_dy, float *mem_dx, float *mem_dz){
#endif
  float c1, c2, c3;
  int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;
  int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;
  int gtid3 = blockIdx.z * blockDim.z + threadIdx.z;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  size_t outIndex;
  size_t ic, pind;
  const int core1_lo = npml + CorePmlMargin;
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;

  if( gtid1 < n1 && gtid2 < n2 && gtid3 < n3){
#ifdef CUDA3D_PML_ZMEM_IN_P
    const bool need_vz = false;
#else
    const bool need_vz = !((gtid1 >= core1_lo + 3) && (gtid1 < core1_hi - 4) &&
			   (gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
			   (gtid3 >= core3_lo) && (gtid3 < core3_hi));
#endif
    bool need_vx = !((gtid1 >= core1_lo) && (gtid1 < core1_hi) &&
		     (gtid2 >= core2_lo + 3) && (gtid2 < core2_hi - 4) &&
		     (gtid3 >= core3_lo) && (gtid3 < core3_hi));
    bool need_vy = !((gtid1 >= core1_lo) && (gtid1 < core1_hi) &&
		     (gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
		     (gtid3 >= core3_lo + 3) && (gtid3 < core3_hi - 4));
#if defined(CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY) || defined(CUDA3D_PML_ZFACE_SHARED_VP_DEBUG)
    if (need_vx && pml_fused_zface_vx_global_point(gtid1, gtid2, gtid3, n3, n2, n1, npml))
      need_vx = false;
    if (need_vy && pml_fused_zface_vy_global_point(gtid1, gtid2, gtid3, n3, n2, n1, npml))
      need_vy = false;
#endif

    if (!need_vz && !need_vx && !need_vy) return;

    outIndex = (size_t)(gtid3 + radius) * stride3 + (size_t)(gtid2 + radius) * stride2 + (gtid1 + radius);
    const size_t ts3 = (size_t)(gtid3 + radius) * stride3;
    const size_t ts2 = (size_t)(gtid2 + radius) * stride2;
    const size_t base = ts3 + ts2 + gtid1 + radius;

    c1=c2=c3=0.0f;

    if (need_vz)
      c1=stencil[1]*(__ldg(p1+base+1)-__ldg(p1+base  ))
	+stencil[2]*(__ldg(p1+base+2)-__ldg(p1+base-1))
	+stencil[3]*(__ldg(p1+base+3)-__ldg(p1+base-2))
	+stencil[4]*(__ldg(p1+base+4)-__ldg(p1+base-3));

    if (need_vx)
      c2=stencil[1]*(__ldg(p1+ts3+(gtid2+radius+1)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius  )*stride2+gtid1+radius))
	+stencil[2]*(__ldg(p1+ts3+(gtid2+radius+2)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius-1)*stride2+gtid1+radius))
	+stencil[3]*(__ldg(p1+ts3+(gtid2+radius+3)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius-2)*stride2+gtid1+radius))
	+stencil[4]*(__ldg(p1+ts3+(gtid2+radius+4)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius-3)*stride2+gtid1+radius));

    if (need_vy)
      c3=stencil[1]*(__ldg(p1+(gtid3+radius+1)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius  )*stride3+ts2+gtid1+radius))
	+stencil[2]*(__ldg(p1+(gtid3+radius+2)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius-1)*stride3+ts2+gtid1+radius))
	+stencil[3]*(__ldg(p1+(gtid3+radius+3)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius-2)*stride3+ts2+gtid1+radius))
	+stencil[4]*(__ldg(p1+(gtid3+radius+4)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius-3)*stride3+ts2+gtid1+radius));
    
    c1*=_dz2;
    c2*=_dx2;
    c3*=_dy2;

#ifndef CUDA3D_PML_RECOMPUTE_Z
    if (need_vz) vz[outIndex]=c1;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_X
    if (need_vx) vx[outIndex]=c2;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_Y
    if (need_vy) vy[outIndex]=c3;
#endif
    
    if(need_vz && gtid1<npml) {
      pind=gtid3*n2*npml + gtid2*npml + gtid1;
      const float coef = c_bz_h_pml[gtid1];
      const float new_mem = __ldg(mem_dz+pind)*coef+c1*(coef-1);
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
      mem_dz_next[pind]=new_mem;
#else
      mem_dz[pind]=new_mem;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_Z
      vz[outIndex]+=new_mem;
#endif
    }
    if (need_vz && gtid1>=n1-npml) {
      ic=gtid1-n1+npml;
      pind=  n3*n2*npml+gtid3*n2*npml + gtid2*npml + ic;
      const float coef = c_az_h_pml[ic];
      const float new_mem = __ldg(mem_dz+pind)*coef+c1*(coef-1);
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
      mem_dz_next[pind]=new_mem;
#else
      mem_dz[pind]=new_mem;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_Z
      vz[outIndex]+=new_mem;
#endif
    }
    
    if (need_vx && gtid2<npml){
      pind=gtid3*npml*n1 + gtid2*n1 + gtid1;
      const float coef = c_bx_h_pml[gtid2];
      const float new_mem = __ldg(mem_dx+pind)*coef+c2*(coef-1);
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
      mem_dx_next[pind]=new_mem;
#else
      mem_dx[pind]=new_mem;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_X
      vx[outIndex]+=new_mem;
#endif
    }
    if (need_vx && gtid2>=n2-npml){
      ic=gtid2-n2+npml;
      pind=n3*npml*n1+gtid3*(npml*n1) + ic*n1 + gtid1;
      const float coef = c_ax_h_pml[ic];
      const float new_mem = __ldg(mem_dx+pind)*coef+c2*(coef-1);
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
      mem_dx_next[pind]=new_mem;
#else
      mem_dx[pind]=new_mem;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_X
      vx[outIndex]+=new_mem;
#endif
    }
    
    if(need_vy && gtid3<npml) {
      pind=gtid3*n2*n1 + gtid2*n1 + gtid1;
      const float coef = c_by_h_pml[gtid3];
      const float new_mem = __ldg(mem_dy+pind)*coef+c3*(coef-1);
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
      mem_dy_next[pind]=new_mem;
#else
      mem_dy[pind]=new_mem;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_Y
      vy[outIndex]+=new_mem;
#endif
    }
    if (need_vy && gtid3>=n3-npml){
      ic=gtid3-n3+npml;
      pind= npml*n2*n1+ic*n2*n1 + gtid2*(n1) + gtid1;
      const float coef = c_ay_h_pml[ic];
      const float new_mem = __ldg(mem_dy+pind)*coef+c3*(coef-1);
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
      mem_dy_next[pind]=new_mem;
#else
      mem_dy[pind]=new_mem;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_Y
      vy[outIndex]+=new_mem;
#endif
    }
  }
}

#ifdef CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK
#if !defined(CUDA3D_PML_ZMEM_IN_P)
#error CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK requires CUDA3D_PML_ZMEM_IN_P
#endif
#if PmlTileBlockSize1 != 32 || PmlTileBlockSize2 != 4 || PmlTileBlockSize3 != 2
#error CUDA3D_PML_VELOCITY_LEN16_HALF_WARP_PACK currently requires PmlTileBlockSize=32x4x2
#endif
__global__ void cuda_fd3d_v_pml_len16_halfwarp_ns(const float *__restrict__ p1, float *vy, float *vx, float *vz,
				   float _dy2, float _dx2, float _dz2,
				   int n3, int n2, int n1, int npml, float dt,
				   float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
				   const float *__restrict__ mem_dy, const float *__restrict__ mem_dx, const float *__restrict__ mem_dz,
				   float *mem_dy_next, float *mem_dx_next, float *mem_dz_next,
#else
				   float *mem_dy, float *mem_dx, float *mem_dz,
#endif
				   const PmlTile *__restrict__ tiles, int ntile){
  if (blockIdx.x >= ntile) return;
  const PmlTile tile = tiles[blockIdx.x];
  const int lane = threadIdx.x;
  const int pair = threadIdx.y;
  const int local_line = pair * 2 + (lane >> 4);
  const int local_z = lane & 15;
  const int local_x = local_line & (PmlTileBlockSize2 - 1);
  const int local_y = local_line >> 2;
  const int gtid2 = tile.x0 + local_x;
  const int gtid3 = tile.y0 + local_y;
  const int core1_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int active_z0 = (tile.z0 < core1_lo) ? tile.z0 : core1_hi;
  const int gtid1 = active_z0 + local_z;
  if (gtid1 < 0 || gtid1 >= n1 || gtid2 >= n2 || gtid3 >= n3)
    return;

  (void)vz;
  (void)dt;
  (void)ay_h; (void)by_h; (void)ax_h; (void)bx_h; (void)az_h; (void)bz_h;
  (void)mem_dy; (void)mem_dx; (void)mem_dz;
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
  (void)mem_dy_next; (void)mem_dx_next; (void)mem_dz_next;
#endif

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  const size_t ts3 = (size_t)(gtid3 + radius) * stride3;
  const size_t ts2 = (size_t)(gtid2 + radius) * stride2;
  const size_t base = ts3 + ts2 + gtid1 + radius;

  float c2=stencil[1]*(__ldg(p1+ts3+(gtid2+radius+1)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius  )*stride2+gtid1+radius))
    +stencil[2]*(__ldg(p1+ts3+(gtid2+radius+2)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius-1)*stride2+gtid1+radius))
    +stencil[3]*(__ldg(p1+ts3+(gtid2+radius+3)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius-2)*stride2+gtid1+radius))
    +stencil[4]*(__ldg(p1+ts3+(gtid2+radius+4)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius-3)*stride2+gtid1+radius));

  float c3=stencil[1]*(__ldg(p1+(gtid3+radius+1)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius  )*stride3+ts2+gtid1+radius))
    +stencil[2]*(__ldg(p1+(gtid3+radius+2)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius-1)*stride3+ts2+gtid1+radius))
    +stencil[3]*(__ldg(p1+(gtid3+radius+3)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius-2)*stride3+ts2+gtid1+radius))
    +stencil[4]*(__ldg(p1+(gtid3+radius+4)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius-3)*stride3+ts2+gtid1+radius));

  c2*=_dx2;
  c3*=_dy2;

#ifndef CUDA3D_PML_RECOMPUTE_X
  vx[base]=c2;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_Y
  vy[base]=c3;
#endif
}
#endif

__global__ void cuda_fd3d_v_pml_tile_ns(const float *__restrict__ p1, float *vy, float *vx, float *vz,
				   float _dy2, float _dx2, float _dz2,
				   int n3, int n2, int n1, int npml, float dt,
				   float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
				   const float *__restrict__ mem_dy, const float *__restrict__ mem_dx, const float *__restrict__ mem_dz,
				   float *mem_dy_next, float *mem_dx_next, float *mem_dz_next,
#else
				   float *mem_dy, float *mem_dx, float *mem_dz,
#endif
				   const PmlTile *__restrict__ tiles, int ntile){
  if (blockIdx.x >= ntile) return;
  const PmlTile tile = tiles[blockIdx.x];
  float c1, c2, c3;
  int gtid1 = tile.z0 + threadIdx.x;
  int gtid2 = tile.x0 + threadIdx.y;
  int gtid3 = tile.y0 + threadIdx.z;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  size_t outIndex;
  size_t ic, pind;
  const int core1_lo = npml + CorePmlMargin;
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;

  if( gtid1 < n1 && gtid2 < n2 && gtid3 < n3){
#ifdef CUDA3D_PML_ZMEM_IN_P
    const bool need_vz = false;
#else
    const bool need_vz = !((gtid1 >= core1_lo + 3) && (gtid1 < core1_hi - 4) &&
			   (gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
			   (gtid3 >= core3_lo) && (gtid3 < core3_hi));
#endif
    bool need_vx = !((gtid1 >= core1_lo) && (gtid1 < core1_hi) &&
		     (gtid2 >= core2_lo + 3) && (gtid2 < core2_hi - 4) &&
		     (gtid3 >= core3_lo) && (gtid3 < core3_hi));
    bool need_vy = !((gtid1 >= core1_lo) && (gtid1 < core1_hi) &&
		     (gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
		     (gtid3 >= core3_lo + 3) && (gtid3 < core3_hi - 4));
#if defined(CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY) || defined(CUDA3D_PML_ZFACE_SHARED_VP_DEBUG)
    if (need_vx && pml_fused_zface_vx_global_point(gtid1, gtid2, gtid3, n3, n2, n1, npml))
      need_vx = false;
    if (need_vy && pml_fused_zface_vy_global_point(gtid1, gtid2, gtid3, n3, n2, n1, npml))
      need_vy = false;
#endif

    if (!need_vz && !need_vx && !need_vy) return;

    outIndex = (size_t)(gtid3 + radius) * stride3 + (size_t)(gtid2 + radius) * stride2 + (gtid1 + radius);
    const size_t ts3 = (size_t)(gtid3 + radius) * stride3;
    const size_t ts2 = (size_t)(gtid2 + radius) * stride2;
    const size_t base = ts3 + ts2 + gtid1 + radius;

    c1=c2=c3=0.0f;

    if (need_vz)
      c1=stencil[1]*(__ldg(p1+base+1)-__ldg(p1+base  ))
	+stencil[2]*(__ldg(p1+base+2)-__ldg(p1+base-1))
	+stencil[3]*(__ldg(p1+base+3)-__ldg(p1+base-2))
	+stencil[4]*(__ldg(p1+base+4)-__ldg(p1+base-3));

    if (need_vx)
      c2=stencil[1]*(__ldg(p1+ts3+(gtid2+radius+1)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius  )*stride2+gtid1+radius))
	+stencil[2]*(__ldg(p1+ts3+(gtid2+radius+2)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius-1)*stride2+gtid1+radius))
	+stencil[3]*(__ldg(p1+ts3+(gtid2+radius+3)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius-2)*stride2+gtid1+radius))
	+stencil[4]*(__ldg(p1+ts3+(gtid2+radius+4)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius-3)*stride2+gtid1+radius));

    if (need_vy)
      c3=stencil[1]*(__ldg(p1+(gtid3+radius+1)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius  )*stride3+ts2+gtid1+radius))
	+stencil[2]*(__ldg(p1+(gtid3+radius+2)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius-1)*stride3+ts2+gtid1+radius))
	+stencil[3]*(__ldg(p1+(gtid3+radius+3)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius-2)*stride3+ts2+gtid1+radius))
	+stencil[4]*(__ldg(p1+(gtid3+radius+4)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius-3)*stride3+ts2+gtid1+radius));
    
    c1*=_dz2;
    c2*=_dx2;
    c3*=_dy2;

#ifndef CUDA3D_PML_RECOMPUTE_Z
    if (need_vz) vz[outIndex]=c1;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_X
    if (need_vx) vx[outIndex]=c2;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_Y
    if (need_vy) vy[outIndex]=c3;
#endif
    
    if(need_vz && gtid1<npml) {
      pind=gtid3*n2*npml + gtid2*npml + gtid1;
      const float coef = c_bz_h_pml[gtid1];
      const float new_mem = __ldg(mem_dz+pind)*coef+c1*(coef-1);
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
      mem_dz_next[pind]=new_mem;
#else
      mem_dz[pind]=new_mem;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_Z
      vz[outIndex]+=new_mem;
#endif
    }
    if (need_vz && gtid1>=n1-npml) {
      ic=gtid1-n1+npml;
      pind=  n3*n2*npml+gtid3*n2*npml + gtid2*npml + ic;
      const float coef = c_az_h_pml[ic];
      const float new_mem = __ldg(mem_dz+pind)*coef+c1*(coef-1);
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
      mem_dz_next[pind]=new_mem;
#else
      mem_dz[pind]=new_mem;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_Z
      vz[outIndex]+=new_mem;
#endif
    }
    
    if (need_vx && gtid2<npml){
      pind=gtid3*npml*n1 + gtid2*n1 + gtid1;
      const float coef = c_bx_h_pml[gtid2];
      const float new_mem = __ldg(mem_dx+pind)*coef+c2*(coef-1);
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
      mem_dx_next[pind]=new_mem;
#else
      mem_dx[pind]=new_mem;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_X
      vx[outIndex]+=new_mem;
#endif
    }
    if (need_vx && gtid2>=n2-npml){
      ic=gtid2-n2+npml;
      pind=n3*npml*n1+gtid3*(npml*n1) + ic*n1 + gtid1;
      const float coef = c_ax_h_pml[ic];
      const float new_mem = __ldg(mem_dx+pind)*coef+c2*(coef-1);
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
      mem_dx_next[pind]=new_mem;
#else
      mem_dx[pind]=new_mem;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_X
      vx[outIndex]+=new_mem;
#endif
    }
    
    if(need_vy && gtid3<npml) {
      pind=gtid3*n2*n1 + gtid2*n1 + gtid1;
      const float coef = c_by_h_pml[gtid3];
      const float new_mem = __ldg(mem_dy+pind)*coef+c3*(coef-1);
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
      mem_dy_next[pind]=new_mem;
#else
      mem_dy[pind]=new_mem;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_Y
      vy[outIndex]+=new_mem;
#endif
    }
    if (need_vy && gtid3>=n3-npml){
      ic=gtid3-n3+npml;
      pind= npml*n2*n1+ic*n2*n1 + gtid2*(n1) + gtid1;
      const float coef = c_ay_h_pml[ic];
      const float new_mem = __ldg(mem_dy+pind)*coef+c3*(coef-1);
#ifdef CUDA3D_CPML_VMEM_DOUBLE_BUFFER_ALL
      mem_dy_next[pind]=new_mem;
#else
      mem_dy[pind]=new_mem;
#endif
#ifndef CUDA3D_PML_RECOMPUTE_Y
      vy[outIndex]+=new_mem;
#endif
    }
  }
}




enum { CoreStencilRadius = 7 };

__global__ void cuda_fd3d_p_core_ns(float *p0, float *p1, float *cw2,
				   float _dy2, float _dx2, float _dz2,
				   int n3, int n2, int n1, int npml, float dt){
  const int core1_lo = npml + CorePmlMargin;
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;
  const int block1 = core1_lo + blockIdx.x * blockDim.x;
  const int block2 = core2_lo + blockIdx.y * blockDim.y;
  const int block3 = core3_lo + blockIdx.z * blockDim.z;
  int gtid1 = block1 + threadIdx.x;
  int gtid2 = block2 + threadIdx.y;
  int gtid3 = block3 + threadIdx.z;
  __shared__ float z_tile[PBlockSize3][PBlockSize2][PBlockSize1 + 2 * CoreStencilRadius];

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  const int t1 = gtid1 + radius;
  const int t2 = gtid2 + radius;
  const int t3 = gtid3 + radius;
  const size_t base = (size_t)t3 * stride3 + (size_t)t2 * stride2 + t1;
  const int local1 = threadIdx.x + CoreStencilRadius;

  if (gtid1 < n1 && gtid2 < n2 && gtid3 < n3) {
    z_tile[threadIdx.z][threadIdx.y][local1] = p1[base];
  }
  if (threadIdx.x < CoreStencilRadius && gtid2 < n2 && gtid3 < n3) {
    const int left1 = block1 + threadIdx.x - CoreStencilRadius;
    const int right1 = block1 + blockDim.x + threadIdx.x;
    const size_t plane = (size_t)t3 * stride3 + (size_t)t2 * stride2;
    z_tile[threadIdx.z][threadIdx.y][threadIdx.x] =
      (left1 >= 0 && left1 < n1) ? p1[plane + left1 + radius] : 0.0f;
    z_tile[threadIdx.z][threadIdx.y][threadIdx.x + blockDim.x + CoreStencilRadius] =
      (right1 >= 0 && right1 < n1) ? p1[plane + right1 + radius] : 0.0f;
  }
  __syncthreads();

  if (gtid1 < core1_lo || gtid1 >= core1_hi ||
      gtid2 < core2_lo || gtid2 >= core2_hi ||
      gtid3 < core3_lo || gtid3 >= core3_hi) return;

  const float z2 = _dz2 * _dz2;
  const float x2 = _dx2 * _dx2;
  const float y2 = _dy2 * _dy2;
  const float center = z_tile[threadIdx.z][threadIdx.y][local1];
  float lap = -2.8751201527567405f * (z2 + x2 + y2) * center;

  lap += 1.6234617233276367f *
    (z2 * (z_tile[threadIdx.z][threadIdx.y][local1 + 1] + z_tile[threadIdx.z][threadIdx.y][local1 - 1]) +
     x2 * (p1[base + stride2] + p1[base - stride2]) +
     y2 * (p1[base + stride3] + p1[base - stride3]));
  lap += -0.21382331848144528f *
    (z2 * (z_tile[threadIdx.z][threadIdx.y][local1 + 2] + z_tile[threadIdx.z][threadIdx.y][local1 - 2]) +
     x2 * (p1[base + 2 * stride2] + p1[base - 2 * stride2]) +
     y2 * (p1[base + 2 * stride3] + p1[base - 2 * stride3]));
  lap += 0.030927128261990015f *
    (z2 * (z_tile[threadIdx.z][threadIdx.y][local1 + 3] + z_tile[threadIdx.z][threadIdx.y][local1 - 3]) +
     x2 * (p1[base + 3 * stride2] + p1[base - 3 * stride2]) +
     y2 * (p1[base + 3 * stride3] + p1[base - 3 * stride3]));
  lap += -0.003195444742838541f *
    (z2 * (z_tile[threadIdx.z][threadIdx.y][local1 + 4] + z_tile[threadIdx.z][threadIdx.y][local1 - 4]) +
     x2 * (p1[base + 4 * stride2] + p1[base - 4 * stride2]) +
     y2 * (p1[base + 4 * stride3] + p1[base - 4 * stride3]));
  lap += 0.0002028528849283854f *
    (z2 * (z_tile[threadIdx.z][threadIdx.y][local1 + 5] + z_tile[threadIdx.z][threadIdx.y][local1 - 5]) +
     x2 * (p1[base + 5 * stride2] + p1[base - 5 * stride2]) +
     y2 * (p1[base + 5 * stride3] + p1[base - 5 * stride3]));
  lap += -0.000013351440429687502f *
    (z2 * (z_tile[threadIdx.z][threadIdx.y][local1 + 6] + z_tile[threadIdx.z][threadIdx.y][local1 - 6]) +
     x2 * (p1[base + 6 * stride2] + p1[base - 6 * stride2]) +
     y2 * (p1[base + 6 * stride3] + p1[base - 6 * stride3]));
  lap += 0.000000486568528778699f *
    (z2 * (z_tile[threadIdx.z][threadIdx.y][local1 + 7] + z_tile[threadIdx.z][threadIdx.y][local1 - 7]) +
     x2 * (p1[base + 7 * stride2] + p1[base - 7 * stride2]) +
     y2 * (p1[base + 7 * stride3] + p1[base - 7 * stride3]));

  p0[base] = 2.0f * center - p0[base] + cw2[base] * dt * lap;

}

#ifdef CUDA3D_PML_RECOMPUTE_Z
__device__ __forceinline__ float recompute_vz_from_p1_mem(const float *__restrict__ p1,
							  const float *__restrict__ mem_dz,
							  float _dz2,
							  int n3, int n2, int n1, int npml,
							  int gtid3, int gtid2, int gtid1) {
  if (gtid1 < 0 || gtid1 >= n1 || gtid2 < 0 || gtid2 >= n2 || gtid3 < 0 || gtid3 >= n3)
    return 0.0f;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  const size_t base = (size_t)(gtid3 + radius) * stride3 + (size_t)(gtid2 + radius) * stride2 + (gtid1 + radius);
  float value=stencil[1]*(__ldg(p1+base+1)-__ldg(p1+base  ))
    +stencil[2]*(__ldg(p1+base+2)-__ldg(p1+base-1))
    +stencil[3]*(__ldg(p1+base+3)-__ldg(p1+base-2))
    +stencil[4]*(__ldg(p1+base+4)-__ldg(p1+base-3));

  value *= _dz2;
  if (gtid1 < npml) {
    const size_t pind=(size_t)gtid3*n2*npml + (size_t)gtid2*npml + gtid1;
    value += __ldg(mem_dz+pind);
  }
  if (gtid1 >= n1-npml) {
    const size_t ic=gtid1-n1+npml;
    const size_t pind=(size_t)n3*n2*npml + (size_t)gtid3*npml*n2 + (size_t)gtid2*npml + ic;
    value += __ldg(mem_dz+pind);
  }
  return value;
}

#ifdef CUDA3D_PML_ZMEM_IN_P
__device__ __forceinline__ float recompute_vz_after_update_from_old_mem(const float *__restrict__ p1,
									const float *__restrict__ mem_dz_old,
									float *__restrict__ mem_dz_new,
									float _dz2,
									int n3, int n2, int n1, int npml,
									int gtid3, int gtid2, int gtid1,
									bool write_owned) {
  if (gtid1 < 0 || gtid1 >= n1 || gtid2 < 0 || gtid2 >= n2 || gtid3 < 0 || gtid3 >= n3)
    return 0.0f;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  const size_t base = (size_t)(gtid3 + radius) * stride3 + (size_t)(gtid2 + radius) * stride2 + (gtid1 + radius);
  float value=stencil[1]*(__ldg(p1+base+1)-__ldg(p1+base  ))
    +stencil[2]*(__ldg(p1+base+2)-__ldg(p1+base-1))
    +stencil[3]*(__ldg(p1+base+3)-__ldg(p1+base-2))
    +stencil[4]*(__ldg(p1+base+4)-__ldg(p1+base-3));

  value *= _dz2;
  if (gtid1 < npml) {
    const size_t pind=(size_t)gtid3*n2*npml + (size_t)gtid2*npml + gtid1;
    const float coef = c_bz_h_pml[gtid1];
    const float new_mem = __ldg(mem_dz_old+pind)*coef + value*(coef-1.0f);
    if (write_owned) mem_dz_new[pind] = new_mem;
    value += new_mem;
  } else if (gtid1 >= n1-npml) {
    const size_t ic=gtid1-n1+npml;
    const size_t pind=(size_t)n3*n2*npml + (size_t)gtid3*npml*n2 + (size_t)gtid2*npml + ic;
    const float coef = c_az_h_pml[ic];
    const float new_mem = __ldg(mem_dz_old+pind)*coef + value*(coef-1.0f);
    if (write_owned) mem_dz_new[pind] = new_mem;
    value += new_mem;
  }
  return value;
}
#endif
#endif

#ifdef CUDA3D_PML_RECOMPUTE_X
__device__ __forceinline__ float recompute_vx_from_p1_mem(const float *__restrict__ p1,
							  const float *__restrict__ mem_dx,
							  float _dx2,
							  int n3, int n2, int n1, int npml,
							  int gtid3, int gtid2, int gtid1) {
  if (gtid1 < 0 || gtid1 >= n1 || gtid2 < 0 || gtid2 >= n2 || gtid3 < 0 || gtid3 >= n3)
    return 0.0f;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  const size_t ts3 = (size_t)(gtid3 + radius) * stride3;
  float value=stencil[1]*(__ldg(p1+ts3+(gtid2+radius+1)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius  )*stride2+gtid1+radius))
    +stencil[2]*(__ldg(p1+ts3+(gtid2+radius+2)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius-1)*stride2+gtid1+radius))
    +stencil[3]*(__ldg(p1+ts3+(gtid2+radius+3)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius-2)*stride2+gtid1+radius))
    +stencil[4]*(__ldg(p1+ts3+(gtid2+radius+4)*stride2+gtid1+radius)-__ldg(p1+ts3+(gtid2+radius-3)*stride2+gtid1+radius));

  value *= _dx2;
  if (gtid2 < npml) {
    const size_t pind=(size_t)gtid3*npml*n1 + (size_t)gtid2*n1 + gtid1;
    value += __ldg(mem_dx+pind);
  }
  if (gtid2 >= n2-npml) {
    const size_t ic=gtid2-n2+npml;
    const size_t pind=(size_t)n3*npml*n1 + (size_t)gtid3*npml*n1 + ic*n1 + gtid1;
    value += __ldg(mem_dx+pind);
  }
  return value;
}
#endif

#ifdef CUDA3D_PML_RECOMPUTE_Y
__device__ __forceinline__ float recompute_vy_from_p1_mem(const float *__restrict__ p1,
							  const float *__restrict__ mem_dy,
							  float _dy2,
							  int n3, int n2, int n1, int npml,
							  int gtid3, int gtid2, int gtid1) {
  if (gtid1 < 0 || gtid1 >= n1 || gtid2 < 0 || gtid2 >= n2 || gtid3 < 0 || gtid3 >= n3)
    return 0.0f;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  const size_t ts2 = (size_t)(gtid2 + radius) * stride2;
  float value=stencil[1]*(__ldg(p1+(gtid3+radius+1)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius  )*stride3+ts2+gtid1+radius))
    +stencil[2]*(__ldg(p1+(gtid3+radius+2)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius-1)*stride3+ts2+gtid1+radius))
    +stencil[3]*(__ldg(p1+(gtid3+radius+3)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius-2)*stride3+ts2+gtid1+radius))
    +stencil[4]*(__ldg(p1+(gtid3+radius+4)*stride3+ts2+gtid1+radius)-__ldg(p1+(gtid3+radius-3)*stride3+ts2+gtid1+radius));

  value *= _dy2;
  if (gtid3 < npml) {
    const size_t pind=(size_t)gtid3*n2*n1 + (size_t)gtid2*n1 + gtid1;
    value += __ldg(mem_dy+pind);
  }
  if (gtid3 >= n3-npml) {
    const size_t ic=gtid3-n3+npml;
    const size_t pind=(size_t)npml*n2*n1 + ic*n2*n1 + (size_t)gtid2*n1 + gtid1;
    value += __ldg(mem_dy+pind);
  }
  return value;
}
#endif

#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
__device__ __forceinline__ bool pml_zface_p_special_point(int gtid1, int gtid2, int gtid3,
							  int n3, int n2, int n1, int npml) {
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;
  return ((gtid1 < npml) || (gtid1 >= n1 - npml)) &&
    (gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
    (gtid3 >= core3_lo) && (gtid3 < core3_hi);
}
#endif

#ifdef CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY
__device__ __forceinline__ float pml_fused_second_inline_from_p1(const float *__restrict__ p1,
								 size_t base,
								 size_t stride,
								 float inv_d) {
  const float d2 = inv_d * inv_d;
  const float center = __ldg(p1 + base);
  float value = -2.8751201527567405f * center;
  value += 1.6234617233276367f * (__ldg(p1 + base + stride) + __ldg(p1 + base - stride));
  value += -0.21382331848144528f * (__ldg(p1 + base + 2 * stride) + __ldg(p1 + base - 2 * stride));
  value += 0.030927128261990015f * (__ldg(p1 + base + 3 * stride) + __ldg(p1 + base - 3 * stride));
  value += -0.003195444742838541f * (__ldg(p1 + base + 4 * stride) + __ldg(p1 + base - 4 * stride));
  value += 0.0002028528849283854f * (__ldg(p1 + base + 5 * stride) + __ldg(p1 + base - 5 * stride));
  value += -0.000013351440429687502f * (__ldg(p1 + base + 6 * stride) + __ldg(p1 + base - 6 * stride));
  value += 0.000000486568528778699f * (__ldg(p1 + base + 7 * stride) + __ldg(p1 + base - 7 * stride));
  return d2 * value;
}

__device__ __forceinline__ void pml_fused_zface_pressure_update(float *p0,
							       const float *__restrict__ p1,
							       float *cw2,
							       float _dy2, float _dx2, float _dz2,
							       int n3, int n2, int n1, int npml, float dt,
							       float *mem_dzz,
							       const float *__restrict__ mem_dz_v,
							       float *mem_dz_next_v,
							       int gtid3, int gtid2, int gtid1) {
  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  const size_t ts3 = (size_t)(gtid3 + radius) * stride3;
  const size_t ts2 = (size_t)(gtid2 + radius) * stride2;
  const size_t base = ts3 + ts2 + gtid1 + radius;

  const float vz0 = recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2,
							   n3, n2, n1, npml, gtid3, gtid2, gtid1, true);
  float c1 = stencil[1]*(vz0-
			 recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-1, false))
    +stencil[2]*(recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+1, false)-
		 recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-2, false))
    +stencil[3]*(recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+2, false)-
		 recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-3, false))
    +stencil[4]*(recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+3, false)-
		 recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-4, false));

  c1 *= _dz2;
  const float c2 = pml_fused_second_inline_from_p1(p1, base, (size_t)stride2, _dx2);
  const float c3 = pml_fused_second_inline_from_p1(p1, base, (size_t)stride3, _dy2);

  size_t pind, ic;
  if (gtid1 < npml) {
    pind=(size_t)gtid3*npml*n2 + (size_t)gtid2*npml + gtid1;
    const float coef = c_bz_pml[gtid1];
    mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
    c1+=mem_dzz[pind];
  } else {
    ic=gtid1-n1+npml;
    pind=(size_t)n3*n2*npml+(size_t)gtid3*npml*n2 + (size_t)gtid2*npml + ic;
    const float coef = c_az_pml[ic];
    mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
    c1+=mem_dzz[pind];
  }

  p0[base]=2*__ldg(p1+base)-p0[base]
    +__ldg(cw2+base)*dt*(c1+c2+c3);
}

#ifdef CUDA3D_PML_LEN16_COMPACT_STATE_MIRROR
__device__ __forceinline__ int cuda3d_pml_z_state_index(
  int gtid3, int gtid2, int gtid1, int n3, int n2, int n1, int npml,
  size_t *pind) {
  if (gtid1 < 0 || gtid1 >= n1 || gtid2 < 0 || gtid2 >= n2 ||
      gtid3 < 0 || gtid3 >= n3)
    return 0;
  if (gtid1 < npml) {
    *pind = (size_t)gtid3 * npml * n2 + (size_t)gtid2 * npml + gtid1;
    return 1;
  }
  if (gtid1 >= n1 - npml) {
    const size_t ic = (size_t)(gtid1 - n1 + npml);
    *pind = (size_t)n3 * n2 * npml + (size_t)gtid3 * npml * n2 +
	    (size_t)gtid2 * npml + ic;
    return 1;
  }
  return 0;
}

__global__ void cuda3d_pml_len16_compact_state_gather_ns(
				   const float *__restrict__ mem_dzz,
				   const float *__restrict__ mem_dz,
				   const float *__restrict__ mem_dz_next,
				   float *__restrict__ compact_dzz16,
				   float *__restrict__ compact_dz_old23,
				   float *__restrict__ compact_dz_next23,
				   const PmlTile *__restrict__ tiles,
				   int ntile, int n3, int n2, int n1, int npml) {
  const size_t total = (size_t)ntile * PmlTileBlockSize2 * PmlTileBlockSize3 * 23u;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += (size_t)blockDim.x * gridDim.x) {
    const int window_z = (int)(idx % 23u);
    const size_t line = idx / 23u;
    const int local_line = (int)(line % (PmlTileBlockSize2 * PmlTileBlockSize3));
    const int tile_id = (int)(line / (PmlTileBlockSize2 * PmlTileBlockSize3));
    const PmlTile tile = tiles[tile_id];
    const int local_x = local_line & (PmlTileBlockSize2 - 1);
    const int local_y = local_line >> 2;
    const int gtid2 = tile.x0 + local_x;
    const int gtid3 = tile.y0 + local_y;
    const int core1_lo = npml + CorePmlMargin;
    const int core1_hi = n1 - npml - CorePmlMargin;
    const int active_z0 = (tile.z0 < core1_lo) ? tile.z0 : core1_hi;
    const int gtid1 = active_z0 + window_z - 4;
    size_t pind = 0;
    const int valid = cuda3d_pml_z_state_index(gtid3, gtid2, gtid1,
					       n3, n2, n1, npml, &pind);
    compact_dz_old23[idx] = valid ? mem_dz[pind] : 0.0f;
    compact_dz_next23[idx] = valid ? mem_dz_next[pind] : 0.0f;
    if (window_z >= 4 && window_z < 20) {
      const size_t z16 = line * 16u + (size_t)(window_z - 4);
      compact_dzz16[z16] = valid ? mem_dzz[pind] : 0.0f;
    }
  }
}

__device__ __forceinline__ void cuda3d_atomic_max_float(float *addr, float value) {
  int *addr_i = (int*)addr;
  int old = *addr_i;
  int assumed;
  const int value_i = __float_as_int(value);
  do {
    assumed = old;
    if (__int_as_float(assumed) >= value)
      break;
    old = atomicCAS(addr_i, assumed, value_i);
  } while (assumed != old);
}

__global__ void cuda3d_pml_len16_compact_state_compare_ns(
				   const float *__restrict__ mem_dzz,
				   const float *__restrict__ mem_dz,
				   const float *__restrict__ mem_dz_next,
				   const float *__restrict__ compact_dzz16,
				   const float *__restrict__ compact_dz_old23,
				   const float *__restrict__ compact_dz_next23,
				   const PmlTile *__restrict__ tiles,
				   int ntile, int n3, int n2, int n1, int npml,
				   float *__restrict__ err_sum,
				   float *__restrict__ ref_sum,
				   float *__restrict__ max_abs,
				   int *__restrict__ bad_count) {
  const size_t total = (size_t)ntile * PmlTileBlockSize2 * PmlTileBlockSize3 * 23u;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < total;
       idx += (size_t)blockDim.x * gridDim.x) {
    const int window_z = (int)(idx % 23u);
    const size_t line = idx / 23u;
    const int local_line = (int)(line % (PmlTileBlockSize2 * PmlTileBlockSize3));
    const int tile_id = (int)(line / (PmlTileBlockSize2 * PmlTileBlockSize3));
    const PmlTile tile = tiles[tile_id];
    const int local_x = local_line & (PmlTileBlockSize2 - 1);
    const int local_y = local_line >> 2;
    const int gtid2 = tile.x0 + local_x;
    const int gtid3 = tile.y0 + local_y;
    const int core1_lo = npml + CorePmlMargin;
    const int core1_hi = n1 - npml - CorePmlMargin;
    const int active_z0 = (tile.z0 < core1_lo) ? tile.z0 : core1_hi;
    const int gtid1 = active_z0 + window_z - 4;
    size_t pind = 0;
    const int valid = cuda3d_pml_z_state_index(gtid3, gtid2, gtid1,
					       n3, n2, n1, npml, &pind);
    const float full_old = valid ? mem_dz[pind] : 0.0f;
    const float full_next = valid ? mem_dz_next[pind] : 0.0f;
    const float diff_old = compact_dz_old23[idx] - full_old;
    const float diff_next = compact_dz_next23[idx] - full_next;
    float local_err = diff_old * diff_old + diff_next * diff_next;
    float local_ref = full_old * full_old + full_next * full_next;
    float local_max = fmaxf(fabsf(diff_old), fabsf(diff_next));
    if (window_z >= 4 && window_z < 20) {
      const size_t z16 = line * 16u + (size_t)(window_z - 4);
      const float full_dzz = valid ? mem_dzz[pind] : 0.0f;
      const float diff_dzz = compact_dzz16[z16] - full_dzz;
      local_err += diff_dzz * diff_dzz;
      local_ref += full_dzz * full_dzz;
      local_max = fmaxf(local_max, fabsf(diff_dzz));
    }
    if (!isfinite(local_err) || !isfinite(local_ref) || !isfinite(local_max))
      atomicAdd(bad_count, 1);
    if (local_max > 0.0f)
      atomicAdd(bad_count, 1);
    atomicAdd(err_sum, local_err);
    atomicAdd(ref_sum, local_ref);
    cuda3d_atomic_max_float(max_abs, local_max);
  }
}
#endif
#endif

__global__ void cuda_fd3d_p_pml_ns(float *p0, const float *__restrict__ p1, const float *__restrict__ vy, const float *__restrict__ vx, const float *__restrict__ vz,
				   float *cw2, float _dy2, float _dx2, float _dz2,
				   int n3, int n2, int n1, int npml, float dt,
				   float *ay, float *by, float *ax, float *bx, float *az, float *bz,
				   float *mem_dyy, float *mem_dxx, float *mem_dzz,
				   const float *__restrict__ mem_dz_v,
				   float *mem_dz_next_v,
				   const float *__restrict__ mem_dx_v,
				   const float *__restrict__ mem_dy_v){
  float c1, c2, c3;
  int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;
  int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;
  int gtid3 = blockIdx.z * blockDim.z + threadIdx.z;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  
  size_t outIndex;
  size_t ic, pind;
  const int core1_lo = npml + CorePmlMargin;
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;

  if (gtid1 < n1 && gtid2 < n2 && gtid3 < n3){
    if ((gtid1 >= core1_lo) && (gtid1 < core1_hi) &&
	(gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
	(gtid3 >= core3_lo) && (gtid3 < core3_hi)) return;
#ifdef CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY
    if (pml_fused_zface_pressure_point(gtid1, gtid2, gtid3, n3, n2, n1, npml)) {
      pml_fused_zface_pressure_update(p0, p1, cw2, _dy2, _dx2, _dz2,
				       n3, n2, n1, npml, dt,
				       mem_dzz, mem_dz_v, mem_dz_next_v,
				       gtid3, gtid2, gtid1);
      return;
    }
#elif defined(CUDA3D_PML_ZFACE_SHARED_VP_DEBUG)
    if (pml_fused_zface_pressure_point(gtid1, gtid2, gtid3, n3, n2, n1, npml)) return;
#endif
#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
    if (pml_zface_p_special_point(gtid1, gtid2, gtid3, n3, n2, n1, npml)) return;
#endif

    outIndex = (size_t)(gtid3 + radius) * stride3 + (size_t)(gtid2 + radius) * stride2 + (gtid1 + radius);

    c1=c2=c3=0.0f;
    const size_t ts3 = (size_t)(gtid3 + radius) * stride3;
    const size_t ts2 = (size_t)(gtid2 + radius) * stride2;
    const size_t base = ts3 + ts2 + gtid1 + radius;

#ifdef CUDA3D_PML_RECOMPUTE_Z
#ifdef CUDA3D_PML_ZMEM_IN_P
    const float vz0 = recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2,
							     n3, n2, n1, npml, gtid3, gtid2, gtid1, true);
    c1=stencil[1]*(vz0-
		   recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-1, false))
      +stencil[2]*(recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+1, false)-
		   recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-2, false))
      +stencil[3]*(recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+2, false)-
		   recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-3, false))
      +stencil[4]*(recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+3, false)-
		   recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-4, false));
#else
    c1=stencil[1]*(recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1  )-
		   recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-1))
      +stencil[2]*(recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+1)-
		   recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-2))
      +stencil[3]*(recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+2)-
		   recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-3))
      +stencil[4]*(recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+3)-
		   recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-4));
#endif
#else
    c1=stencil[1]*(__ldg(vz+base  )-__ldg(vz+base-1))
      +stencil[2]*(__ldg(vz+base+1)-__ldg(vz+base-2))
      +stencil[3]*(__ldg(vz+base+2)-__ldg(vz+base-3))
      +stencil[4]*(__ldg(vz+base+3)-__ldg(vz+base-4));
#endif

#ifdef CUDA3D_PML_RECOMPUTE_X
    c2=stencil[1]*(recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2  , gtid1)-
		   recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2-1, gtid1))
      +stencil[2]*(recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2+1, gtid1)-
		   recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2-2, gtid1))
      +stencil[3]*(recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2+2, gtid1)-
		   recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2-3, gtid1))
      +stencil[4]*(recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2+3, gtid1)-
		   recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2-4, gtid1));
#else
    c2=stencil[1]*(__ldg(vx+ts3+(gtid2+radius  )*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-1)*stride2+gtid1+radius))
      +stencil[2]*(__ldg(vx+ts3+(gtid2+radius+1)*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-2)*stride2+gtid1+radius))
      +stencil[3]*(__ldg(vx+ts3+(gtid2+radius+2)*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-3)*stride2+gtid1+radius))
      +stencil[4]*(__ldg(vx+ts3+(gtid2+radius+3)*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-4)*stride2+gtid1+radius));
#endif

#ifdef CUDA3D_PML_RECOMPUTE_Y
    c3=stencil[1]*(recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3  , gtid2, gtid1)-
		   recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3-1, gtid2, gtid1))
      +stencil[2]*(recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3+1, gtid2, gtid1)-
		   recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3-2, gtid2, gtid1))
      +stencil[3]*(recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3+2, gtid2, gtid1)-
		   recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3-3, gtid2, gtid1))
      +stencil[4]*(recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3+3, gtid2, gtid1)-
		   recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3-4, gtid2, gtid1));
#else
    c3=stencil[1]*(__ldg(vy+(gtid3+radius  )*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-1)*stride3+ts2+gtid1+radius))
      +stencil[2]*(__ldg(vy+(gtid3+radius+1)*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-2)*stride3+ts2+gtid1+radius))
      +stencil[3]*(__ldg(vy+(gtid3+radius+2)*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-3)*stride3+ts2+gtid1+radius))
      +stencil[4]*(__ldg(vy+(gtid3+radius+3)*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-4)*stride3+ts2+gtid1+radius));
#endif

    c1*=_dz2;
    c2*=_dx2;
    c3*=_dy2;

    //Start Z-PML
    if(gtid1<npml) {
      pind=gtid3*npml*n2 + gtid2*npml + gtid1;
      const float coef = c_bz_pml[gtid1];
      mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
      c1+=mem_dzz[pind];
    }
    if (gtid1>=n1-npml) {
      ic=gtid1-n1+npml;
      pind= n3*n2*npml+gtid3*npml*n2 + gtid2*npml + ic;
      const float coef = c_az_pml[ic];
      mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
      c1+=mem_dzz[pind];
    }
    //End ZPML

    //Start X-PML
    if (gtid2<npml){
      pind=gtid3*npml*n1 + gtid2*n1 + gtid1;
      const float coef = c_bx_pml[gtid2];
      mem_dxx[pind]=mem_dxx[pind]*coef+c2*(coef-1);
      c2+=mem_dxx[pind];
    }
    if (gtid2>=n2-npml){
      ic=gtid2-n2+npml;
      pind=n3*npml*n1+gtid3*(npml*n1) + ic*n1 + gtid1;
      const float coef = c_ax_pml[ic];
      mem_dxx[pind]=mem_dxx[pind]*coef+c2*(coef-1);
      c2+=mem_dxx[pind];
    }
    //End XPML

    // Start Y-PML
    if(gtid3<npml) {
      pind=gtid3*n2*n1 + gtid2*n1 + gtid1;
      const float coef = c_by_pml[gtid3];
      mem_dyy[pind]=mem_dyy[pind]*coef+c3*(coef-1);
      c3+=mem_dyy[pind];
    }
    if (gtid3>=n3-npml){
      ic=gtid3-n3+npml;
      pind= npml*n2*n1+ic*n2*n1 + gtid2*(n1) + gtid1;
      const float coef = c_ay_pml[ic];
      mem_dyy[pind]=mem_dyy[pind]*coef+c3*(coef-1);
      c3+=mem_dyy[pind];
    }
    //END YPML

    p0[outIndex]=2*__ldg(p1+outIndex)-p0[outIndex]
      +__ldg(cw2+outIndex)*dt*(c1+c2+c3);
  }
}

#ifdef CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
__device__ __forceinline__ void fill_pml_pressure_vz_cache_entry(
  float *vz_line_cache, int cache_idx, int local_z,
  int tile_z0, int cache_x, int cache_y,
  const float *__restrict__ p1,
  const float *__restrict__ mem_dz_v,
  float *mem_dz_next_v,
  float _dz2,
  int n3, int n2, int n1, int npml,
  int core1_lo, int core1_hi,
  int core2_lo, int core2_hi,
  int core3_lo, int core3_hi) {
  const int cache_z = tile_z0 + local_z - 4;
  const int central_z0 = tile_z0;
  const int central_z1_full = tile_z0 + PmlTileBlockSize1;
  const int central_z1 = central_z1_full < n1 ? central_z1_full : n1;
  int active_z_lo = central_z0;
  int active_z_hi = central_z1;
  const bool xy_in_domain = cache_x >= 0 && cache_x < n2 &&
    cache_y >= 0 && cache_y < n3;
  if (!xy_in_domain || central_z1 <= central_z0) {
    active_z_hi = active_z_lo;
  } else {
    const bool xy_in_core = (cache_x >= core2_lo) && (cache_x < core2_hi) &&
      (cache_y >= core3_lo) && (cache_y < core3_hi);
    if (xy_in_core && central_z0 < core1_lo) {
      active_z_hi = central_z1 < core1_lo ? central_z1 : core1_lo;
    } else if (xy_in_core && central_z1 > core1_hi) {
      active_z_lo = central_z0 > core1_hi ? central_z0 : core1_hi;
    } else if (xy_in_core) {
      active_z_hi = active_z_lo;
    }
  }
  const bool cache_needed = (active_z_hi > active_z_lo) &&
    (cache_z >= active_z_lo - 4) && (cache_z < active_z_hi + 3);
  if (!cache_needed) {
    vz_line_cache[cache_idx] = 0.0f;
    return;
  }
  const bool in_owned_z = (local_z >= 4) && (local_z < 4 + PmlTileBlockSize1);
  const bool in_domain = cache_z >= 0 && cache_z < n1 &&
    cache_x >= 0 && cache_x < n2 &&
    cache_y >= 0 && cache_y < n3;
  const bool in_core = in_domain &&
    (cache_z >= core1_lo) && (cache_z < core1_hi) &&
    (cache_x >= core2_lo) && (cache_x < core2_hi) &&
    (cache_y >= core3_lo) && (cache_y < core3_hi);
  const bool write_owned = in_owned_z && in_domain && !in_core;
  vz_line_cache[cache_idx] = recompute_vz_after_update_from_old_mem(
    p1, mem_dz_v, mem_dz_next_v, _dz2,
    n3, n2, n1, npml, cache_y, cache_x, cache_z, write_owned);
}
#endif

#ifdef CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK
#if !defined(CUDA3D_PML_RECOMPUTE_Z) || !defined(CUDA3D_PML_ZMEM_IN_P) || !defined(CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE)
#error CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK requires CUDA3D_PML_RECOMPUTE_Z, CUDA3D_PML_ZMEM_IN_P, and CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
#endif
#if PmlTileBlockSize1 != 32 || PmlTileBlockSize2 != 4 || PmlTileBlockSize3 != 2
#error CUDA3D_PML_PRESSURE_LEN16_HALF_WARP_PACK currently requires PmlTileBlockSize=32x4x2
#endif
__global__ void cuda_fd3d_p_pml_len16_halfwarp_ns(float *p0, const float *__restrict__ p1,
				   const float *__restrict__ vy, const float *__restrict__ vx,
				   float *cw2, float _dy2, float _dx2, float _dz2,
				   int n3, int n2, int n1, int npml, float dt,
				   float *mem_dzz,
				   const float *__restrict__ mem_dz_v,
				   float *mem_dz_next_v,
				   const PmlTile *__restrict__ tiles, int ntile) {
  if (blockIdx.x >= ntile) return;
  const PmlTile tile = tiles[blockIdx.x];
  const int lane = threadIdx.x;
  const int pair = threadIdx.y;
  const int local_line = pair * 2 + (lane >> 4);
  const int local_z = lane & 15;
  const int local_x = local_line & (PmlTileBlockSize2 - 1);
  const int local_y = local_line >> 2;
  const int gtid2 = tile.x0 + local_x;
  const int gtid3 = tile.y0 + local_y;
  const int core1_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int tile_z1 = min(tile.z0 + PmlTileBlockSize1, n1);
  const int active_z0 = (tile.z0 < core1_lo) ? tile.z0 : core1_hi;
  const int gtid1 = active_z0 + local_z;
  if (gtid1 < 0 || gtid1 >= n1 || gtid2 >= n2 || gtid3 >= n3 || tile_z1 <= active_z0)
    return;

  const int z_cache_len = 16 + 7;
  __shared__ float vz_line_cache[(16 + 7) * PmlTileBlockSize2 * PmlTileBlockSize3];
  const int cache_base = local_line * z_cache_len;
  const int cache_z_center = active_z0 + local_z;
  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);

  vz_line_cache[cache_base + local_z + 4] = recompute_vz_after_update_from_old_mem(
    p1, mem_dz_v, mem_dz_next_v, _dz2,
    n3, n2, n1, npml, gtid3, gtid2, cache_z_center, true);
  if (local_z < 4) {
    vz_line_cache[cache_base + local_z] = recompute_vz_after_update_from_old_mem(
      p1, mem_dz_v, mem_dz_next_v, _dz2,
      n3, n2, n1, npml, gtid3, gtid2, active_z0 + local_z - 4, false);
  }
  if (local_z < 3) {
    vz_line_cache[cache_base + 20 + local_z] = recompute_vz_after_update_from_old_mem(
      p1, mem_dz_v, mem_dz_next_v, _dz2,
      n3, n2, n1, npml, gtid3, gtid2, active_z0 + 16 + local_z, false);
  }
  __syncthreads();

  const size_t ts3 = (size_t)(gtid3 + radius) * stride3;
  const size_t ts2 = (size_t)(gtid2 + radius) * stride2;
  const size_t base = ts3 + ts2 + gtid1 + radius;
  const int cbase = cache_base + local_z + 4;
  float c1=stencil[1]*(vz_line_cache[cbase]-
		       vz_line_cache[cbase-1])
    +stencil[2]*(vz_line_cache[cbase+1]-
		 vz_line_cache[cbase-2])
    +stencil[3]*(vz_line_cache[cbase+2]-
		 vz_line_cache[cbase-3])
    +stencil[4]*(vz_line_cache[cbase+3]-
		 vz_line_cache[cbase-4]);
  float c2=stencil[1]*(__ldg(vx+ts3+(gtid2+radius  )*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-1)*stride2+gtid1+radius))
    +stencil[2]*(__ldg(vx+ts3+(gtid2+radius+1)*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-2)*stride2+gtid1+radius))
    +stencil[3]*(__ldg(vx+ts3+(gtid2+radius+2)*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-3)*stride2+gtid1+radius))
    +stencil[4]*(__ldg(vx+ts3+(gtid2+radius+3)*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-4)*stride2+gtid1+radius));
  float c3=stencil[1]*(__ldg(vy+(gtid3+radius  )*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-1)*stride3+ts2+gtid1+radius))
    +stencil[2]*(__ldg(vy+(gtid3+radius+1)*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-2)*stride3+ts2+gtid1+radius))
    +stencil[3]*(__ldg(vy+(gtid3+radius+2)*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-3)*stride3+ts2+gtid1+radius))
    +stencil[4]*(__ldg(vy+(gtid3+radius+3)*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-4)*stride3+ts2+gtid1+radius));

  c1*=_dz2;
  c2*=_dx2;
  c3*=_dy2;
  if(gtid1<npml) {
    const size_t pind=(size_t)gtid3*npml*n2 + (size_t)gtid2*npml + gtid1;
    const float coef = c_bz_pml[gtid1];
    mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
    c1+=mem_dzz[pind];
  } else if (gtid1>=n1-npml) {
    const size_t ic=gtid1-n1+npml;
    const size_t pind=(size_t)n3*n2*npml+(size_t)gtid3*npml*n2 + (size_t)gtid2*npml + ic;
    const float coef = c_az_pml[ic];
    mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
    c1+=mem_dzz[pind];
  }
  p0[base]=2*__ldg(p1+base)-p0[base]
    +__ldg(cw2+base)*dt*(c1+c2+c3);
}
#endif

__global__ void cuda_fd3d_p_pml_tile_ns(float *p0, const float *__restrict__ p1, const float *__restrict__ vy, const float *__restrict__ vx, const float *__restrict__ vz,
				   float *cw2, float _dy2, float _dx2, float _dz2,
				   int n3, int n2, int n1, int npml, float dt,
				   float *ay, float *by, float *ax, float *bx, float *az, float *bz,
				   float *mem_dyy, float *mem_dxx, float *mem_dzz,
				   const float *__restrict__ mem_dz_v,
				   float *mem_dz_next_v,
				   const float *__restrict__ mem_dx_v,
				   const float *__restrict__ mem_dy_v,
				   const PmlTile *__restrict__ tiles, int ntile){
  if (blockIdx.x >= ntile) return;
  const PmlTile tile = tiles[blockIdx.x];
#ifdef CUDA3D_PML_TILE_MASK_FASTPATH
  const unsigned int tile_mask = tile.mask & 15u;
  const unsigned int tile_axes = tile_mask & (PML_TILE_MASK_Z | PML_TILE_MASK_X | PML_TILE_MASK_Y);
#endif
  float c1, c2, c3;
  int gtid1 = tile.z0 + threadIdx.x;
  int gtid2 = tile.x0 + threadIdx.y;
  int gtid3 = tile.y0 + threadIdx.z;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  
  size_t outIndex;
  size_t ic, pind;
  const int core1_lo = npml + CorePmlMargin;
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;

#ifdef CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
#if !defined(CUDA3D_PML_RECOMPUTE_Z) || !defined(CUDA3D_PML_ZMEM_IN_P)
#error CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE requires CUDA3D_PML_RECOMPUTE_Z and CUDA3D_PML_ZMEM_IN_P
#endif
  const int z_cache_len = PmlTileBlockSize1 + 7;
  __shared__ float vz_line_cache[(PmlTileBlockSize1 + 7) * PmlTileBlockSize2 * PmlTileBlockSize3];
  const int z_cache_line_base = (threadIdx.z * PmlTileBlockSize2 + threadIdx.y) * z_cache_len;
  const int z_cache_x = gtid2;
  const int z_cache_y = gtid3;
  fill_pml_pressure_vz_cache_entry(
    vz_line_cache, z_cache_line_base + threadIdx.x + 4, threadIdx.x + 4,
    tile.z0, z_cache_x, z_cache_y,
    p1, mem_dz_v, mem_dz_next_v, _dz2,
    n3, n2, n1, npml,
    core1_lo, core1_hi, core2_lo, core2_hi, core3_lo, core3_hi);
  if (threadIdx.x < 4) {
    fill_pml_pressure_vz_cache_entry(
      vz_line_cache, z_cache_line_base + threadIdx.x, threadIdx.x,
      tile.z0, z_cache_x, z_cache_y,
      p1, mem_dz_v, mem_dz_next_v, _dz2,
      n3, n2, n1, npml,
      core1_lo, core1_hi, core2_lo, core2_hi, core3_lo, core3_hi);
  }
  if (threadIdx.x < 3) {
    const int local_z = PmlTileBlockSize1 + 4 + threadIdx.x;
    fill_pml_pressure_vz_cache_entry(
      vz_line_cache, z_cache_line_base + local_z, local_z,
      tile.z0, z_cache_x, z_cache_y,
      p1, mem_dz_v, mem_dz_next_v, _dz2,
      n3, n2, n1, npml,
      core1_lo, core1_hi, core2_lo, core2_hi, core3_lo, core3_hi);
  }
  __syncthreads();
#endif

  if (gtid1 < n1 && gtid2 < n2 && gtid3 < n3){
    if ((gtid1 >= core1_lo) && (gtid1 < core1_hi) &&
	(gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
	(gtid3 >= core3_lo) && (gtid3 < core3_hi)) return;
#ifdef CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY
    if (pml_fused_zface_pressure_point(gtid1, gtid2, gtid3, n3, n2, n1, npml)) {
      pml_fused_zface_pressure_update(p0, p1, cw2, _dy2, _dx2, _dz2,
				       n3, n2, n1, npml, dt,
				       mem_dzz, mem_dz_v, mem_dz_next_v,
				       gtid3, gtid2, gtid1);
      return;
    }
#elif defined(CUDA3D_PML_ZFACE_SHARED_VP_DEBUG)
    if (pml_fused_zface_pressure_point(gtid1, gtid2, gtid3, n3, n2, n1, npml)) return;
#endif
#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
    const bool tile_may_hit_zface = ((tile.z0 < npml) || (tile.z0 + (int)blockDim.x > n1 - npml)) &&
      (tile.x0 < core2_hi) && (tile.x0 + (int)blockDim.y > core2_lo) &&
      (tile.y0 < core3_hi) && (tile.y0 + (int)blockDim.z > core3_lo);
    if (tile_may_hit_zface && pml_zface_p_special_point(gtid1, gtid2, gtid3, n3, n2, n1, npml)) return;
#endif

    outIndex = (size_t)(gtid3 + radius) * stride3 + (size_t)(gtid2 + radius) * stride2 + (gtid1 + radius);

    c1=c2=c3=0.0f;
    const size_t ts3 = (size_t)(gtid3 + radius) * stride3;
    const size_t ts2 = (size_t)(gtid2 + radius) * stride2;
    const size_t base = ts3 + ts2 + gtid1 + radius;

#ifdef CUDA3D_PML_RECOMPUTE_Z
#ifdef CUDA3D_PML_ZMEM_IN_P
#ifdef CUDA3D_PML_PRESSURE_ZRECOMP_SHARED_LINE_CACHE
    const int vz_cache_base = (threadIdx.z * PmlTileBlockSize2 + threadIdx.y) * z_cache_len + threadIdx.x + 4;
    c1=stencil[1]*(vz_line_cache[vz_cache_base]-
		   vz_line_cache[vz_cache_base-1])
      +stencil[2]*(vz_line_cache[vz_cache_base+1]-
		   vz_line_cache[vz_cache_base-2])
      +stencil[3]*(vz_line_cache[vz_cache_base+2]-
		   vz_line_cache[vz_cache_base-3])
      +stencil[4]*(vz_line_cache[vz_cache_base+3]-
		   vz_line_cache[vz_cache_base-4]);
#else
    const float vz0 = recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2,
							     n3, n2, n1, npml, gtid3, gtid2, gtid1, true);
    c1=stencil[1]*(vz0-
		   recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-1, false))
      +stencil[2]*(recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+1, false)-
		   recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-2, false))
      +stencil[3]*(recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+2, false)-
		   recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-3, false))
      +stencil[4]*(recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+3, false)-
		   recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-4, false));
#endif
#else
    c1=stencil[1]*(recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1  )-
		   recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-1))
      +stencil[2]*(recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+1)-
		   recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-2))
      +stencil[3]*(recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+2)-
		   recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-3))
      +stencil[4]*(recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+3)-
		   recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-4));
#endif
#else
    c1=stencil[1]*(__ldg(vz+base  )-__ldg(vz+base-1))
      +stencil[2]*(__ldg(vz+base+1)-__ldg(vz+base-2))
      +stencil[3]*(__ldg(vz+base+2)-__ldg(vz+base-3))
      +stencil[4]*(__ldg(vz+base+3)-__ldg(vz+base-4));
#endif

#ifdef CUDA3D_PML_RECOMPUTE_X
    c2=stencil[1]*(recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2  , gtid1)-
		   recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2-1, gtid1))
      +stencil[2]*(recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2+1, gtid1)-
		   recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2-2, gtid1))
      +stencil[3]*(recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2+2, gtid1)-
		   recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2-3, gtid1))
      +stencil[4]*(recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2+3, gtid1)-
		   recompute_vx_from_p1_mem(p1, mem_dx_v, _dx2, n3, n2, n1, npml, gtid3, gtid2-4, gtid1));
#else
    c2=stencil[1]*(__ldg(vx+ts3+(gtid2+radius  )*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-1)*stride2+gtid1+radius))
      +stencil[2]*(__ldg(vx+ts3+(gtid2+radius+1)*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-2)*stride2+gtid1+radius))
      +stencil[3]*(__ldg(vx+ts3+(gtid2+radius+2)*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-3)*stride2+gtid1+radius))
      +stencil[4]*(__ldg(vx+ts3+(gtid2+radius+3)*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-4)*stride2+gtid1+radius));
#endif

#ifdef CUDA3D_PML_RECOMPUTE_Y
    c3=stencil[1]*(recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3  , gtid2, gtid1)-
		   recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3-1, gtid2, gtid1))
      +stencil[2]*(recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3+1, gtid2, gtid1)-
		   recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3-2, gtid2, gtid1))
      +stencil[3]*(recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3+2, gtid2, gtid1)-
		   recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3-3, gtid2, gtid1))
      +stencil[4]*(recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3+3, gtid2, gtid1)-
		   recompute_vy_from_p1_mem(p1, mem_dy_v, _dy2, n3, n2, n1, npml, gtid3-4, gtid2, gtid1));
#else
    c3=stencil[1]*(__ldg(vy+(gtid3+radius  )*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-1)*stride3+ts2+gtid1+radius))
      +stencil[2]*(__ldg(vy+(gtid3+radius+1)*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-2)*stride3+ts2+gtid1+radius))
      +stencil[3]*(__ldg(vy+(gtid3+radius+2)*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-3)*stride3+ts2+gtid1+radius))
      +stencil[4]*(__ldg(vy+(gtid3+radius+3)*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-4)*stride3+ts2+gtid1+radius));
#endif

    c1*=_dz2;
    c2*=_dx2;
    c3*=_dy2;

#ifdef CUDA3D_PML_TILE_MASK_FASTPATH
    if (tile_axes == 0u) {
      // No PML-memory axis intersects this CTA; only the pressure update is needed.
    } else if (tile_mask == PML_TILE_MASK_Z) {
      if(gtid1<npml) {
	pind=gtid3*npml*n2 + gtid2*npml + gtid1;
	const float coef = c_bz_pml[gtid1];
	mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
	c1+=mem_dzz[pind];
      } else if (gtid1>=n1-npml) {
	ic=gtid1-n1+npml;
	pind= n3*n2*npml+gtid3*npml*n2 + gtid2*npml + ic;
	const float coef = c_az_pml[ic];
	mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
	c1+=mem_dzz[pind];
      }
    } else if (tile_mask == PML_TILE_MASK_X) {
      if (gtid2<npml){
	pind=gtid3*npml*n1 + gtid2*n1 + gtid1;
	const float coef = c_bx_pml[gtid2];
	mem_dxx[pind]=mem_dxx[pind]*coef+c2*(coef-1);
	c2+=mem_dxx[pind];
      } else if (gtid2>=n2-npml){
	ic=gtid2-n2+npml;
	pind=n3*npml*n1+gtid3*(npml*n1) + ic*n1 + gtid1;
	const float coef = c_ax_pml[ic];
	mem_dxx[pind]=mem_dxx[pind]*coef+c2*(coef-1);
	c2+=mem_dxx[pind];
      }
    } else if (tile_mask == PML_TILE_MASK_Y) {
      if(gtid3<npml) {
	pind=gtid3*n2*n1 + gtid2*n1 + gtid1;
	const float coef = c_by_pml[gtid3];
	mem_dyy[pind]=mem_dyy[pind]*coef+c3*(coef-1);
	c3+=mem_dyy[pind];
      } else if (gtid3>=n3-npml){
	ic=gtid3-n3+npml;
	pind= npml*n2*n1+ic*n2*n1 + gtid2*(n1) + gtid1;
	const float coef = c_ay_pml[ic];
	mem_dyy[pind]=mem_dyy[pind]*coef+c3*(coef-1);
	c3+=mem_dyy[pind];
      }
    } else
#endif
    {
      if(gtid1<npml) {
	pind=gtid3*npml*n2 + gtid2*npml + gtid1;
	const float coef = c_bz_pml[gtid1];
	mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
	c1+=mem_dzz[pind];
      }
      if (gtid1>=n1-npml) {
	ic=gtid1-n1+npml;
	pind= n3*n2*npml+gtid3*npml*n2 + gtid2*npml + ic;
	const float coef = c_az_pml[ic];
	mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
	c1+=mem_dzz[pind];
      }

      if (gtid2<npml){
	pind=gtid3*npml*n1 + gtid2*n1 + gtid1;
	const float coef = c_bx_pml[gtid2];
	mem_dxx[pind]=mem_dxx[pind]*coef+c2*(coef-1);
	c2+=mem_dxx[pind];
      }
      if (gtid2>=n2-npml){
	ic=gtid2-n2+npml;
	pind=n3*npml*n1+gtid3*(npml*n1) + ic*n1 + gtid1;
	const float coef = c_ax_pml[ic];
	mem_dxx[pind]=mem_dxx[pind]*coef+c2*(coef-1);
	c2+=mem_dxx[pind];
      }

      if(gtid3<npml) {
	pind=gtid3*n2*n1 + gtid2*n1 + gtid1;
	const float coef = c_by_pml[gtid3];
	mem_dyy[pind]=mem_dyy[pind]*coef+c3*(coef-1);
	c3+=mem_dyy[pind];
      }
      if (gtid3>=n3-npml){
	ic=gtid3-n3+npml;
	pind= npml*n2*n1+ic*n2*n1 + gtid2*(n1) + gtid1;
	const float coef = c_ay_pml[ic];
	mem_dyy[pind]=mem_dyy[pind]*coef+c3*(coef-1);
	c3+=mem_dyy[pind];
      }
    }

    p0[outIndex]=2*__ldg(p1+outIndex)-p0[outIndex]
      +__ldg(cw2+outIndex)*dt*(c1+c2+c3);
  }
}

#ifdef CUDA3D_PML_ZFACE_P_SPECIALIZE
__global__ void cuda_fd3d_p_pml_zface_ns(float *p0, const float *__restrict__ p1, const float *__restrict__ vy, const float *__restrict__ vx, const float *__restrict__ vz,
				   float *cw2, float _dy2, float _dx2, float _dz2,
				   int n3, int n2, int n1, int npml, float dt,
				   float *mem_dzz, const float *__restrict__ mem_dz_v,
				   const PmlTile *__restrict__ tiles, int ntile){
  if (blockIdx.x >= ntile) return;
  const PmlTile tile = tiles[blockIdx.x];
  const int gtid1 = tile.z0 + threadIdx.x;
  const int gtid2 = tile.x0 + threadIdx.y;
  const int gtid3 = tile.y0 + threadIdx.z;

  if (gtid1 >= n1 || gtid2 >= n2 || gtid3 >= n3) return;
  if (!pml_zface_p_special_point(gtid1, gtid2, gtid3, n3, n2, n1, npml)) return;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  const size_t ts3 = (size_t)(gtid3 + radius) * stride3;
  const size_t ts2 = (size_t)(gtid2 + radius) * stride2;
  const size_t base = ts3 + ts2 + gtid1 + radius;
  const size_t outIndex = base;

  float c1, c2, c3;
  size_t ic, pind;

#ifdef CUDA3D_PML_RECOMPUTE_Z
  c1=stencil[1]*(recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1  )-
		 recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-1))
    +stencil[2]*(recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+1)-
		 recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-2))
    +stencil[3]*(recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+2)-
		 recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-3))
    +stencil[4]*(recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+3)-
		 recompute_vz_from_p1_mem(p1, mem_dz_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-4));
#else
  c1=stencil[1]*(__ldg(vz+base  )-__ldg(vz+base-1))
    +stencil[2]*(__ldg(vz+base+1)-__ldg(vz+base-2))
    +stencil[3]*(__ldg(vz+base+2)-__ldg(vz+base-3))
    +stencil[4]*(__ldg(vz+base+3)-__ldg(vz+base-4));
#endif

  c2=stencil[1]*(__ldg(vx+ts3+(gtid2+radius  )*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-1)*stride2+gtid1+radius))
    +stencil[2]*(__ldg(vx+ts3+(gtid2+radius+1)*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-2)*stride2+gtid1+radius))
    +stencil[3]*(__ldg(vx+ts3+(gtid2+radius+2)*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-3)*stride2+gtid1+radius))
    +stencil[4]*(__ldg(vx+ts3+(gtid2+radius+3)*stride2+gtid1+radius)-__ldg(vx+ts3+(gtid2+radius-4)*stride2+gtid1+radius));

  c3=stencil[1]*(__ldg(vy+(gtid3+radius  )*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-1)*stride3+ts2+gtid1+radius))
    +stencil[2]*(__ldg(vy+(gtid3+radius+1)*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-2)*stride3+ts2+gtid1+radius))
    +stencil[3]*(__ldg(vy+(gtid3+radius+2)*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-3)*stride3+ts2+gtid1+radius))
    +stencil[4]*(__ldg(vy+(gtid3+radius+3)*stride3+ts2+gtid1+radius)-__ldg(vy+(gtid3+radius-4)*stride3+ts2+gtid1+radius));

  c1*=_dz2;
  c2*=_dx2;
  c3*=_dy2;

  if(gtid1<npml) {
    pind=(size_t)gtid3*npml*n2 + (size_t)gtid2*npml + gtid1;
    const float coef = c_bz_pml[gtid1];
    mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
    c1+=mem_dzz[pind];
  } else {
    ic=gtid1-n1+npml;
    pind=(size_t)n3*n2*npml+(size_t)gtid3*npml*n2 + (size_t)gtid2*npml + ic;
    const float coef = c_az_pml[ic];
    mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
    c1+=mem_dzz[pind];
  }

  p0[outIndex]=2*__ldg(p1+outIndex)-p0[outIndex]
    +__ldg(cw2+outIndex)*dt*(c1+c2+c3);
}
#endif

#ifdef CUDA3D_PML_REGION_FUSED_VP_ZFACE_ONLY
__device__ __forceinline__ float pml_fused_second_from_p1(const float *__restrict__ p1,
							  size_t base,
							  size_t stride,
							  float inv_d) {
  const float d2 = inv_d * inv_d;
  const float center = __ldg(p1 + base);
  float value = -2.8751201527567405f * center;
  value += 1.6234617233276367f * (__ldg(p1 + base + stride) + __ldg(p1 + base - stride));
  value += -0.21382331848144528f * (__ldg(p1 + base + 2 * stride) + __ldg(p1 + base - 2 * stride));
  value += 0.030927128261990015f * (__ldg(p1 + base + 3 * stride) + __ldg(p1 + base - 3 * stride));
  value += -0.003195444742838541f * (__ldg(p1 + base + 4 * stride) + __ldg(p1 + base - 4 * stride));
  value += 0.0002028528849283854f * (__ldg(p1 + base + 5 * stride) + __ldg(p1 + base - 5 * stride));
  value += -0.000013351440429687502f * (__ldg(p1 + base + 6 * stride) + __ldg(p1 + base - 6 * stride));
  value += 0.000000486568528778699f * (__ldg(p1 + base + 7 * stride) + __ldg(p1 + base - 7 * stride));
  return d2 * value;
}

__global__ void cuda_fd3d_pml_fused_vp_zface_ns(float *p0,
					       const float *__restrict__ p1,
					       float *cw2, float _dy2, float _dx2, float _dz2,
					       int n3, int n2, int n1, int npml, float dt,
					       float *mem_dzz,
					       const float *__restrict__ mem_dz_v,
					       float *mem_dz_next_v,
					       const PmlTile *__restrict__ tiles, int ntile) {
  if (blockIdx.x >= ntile) return;
  const PmlTile tile = tiles[blockIdx.x];
  const int gtid1 = tile.z0 + threadIdx.x;
  const int gtid2 = tile.x0 + threadIdx.y;
  const int gtid3 = tile.y0 + threadIdx.z;

  if (gtid1 >= n1 || gtid2 >= n2 || gtid3 >= n3) return;
  if (!pml_fused_zface_pressure_point(gtid1, gtid2, gtid3, n3, n2, n1, npml)) return;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  const size_t ts3 = (size_t)(gtid3 + radius) * stride3;
  const size_t ts2 = (size_t)(gtid2 + radius) * stride2;
  const size_t base = ts3 + ts2 + gtid1 + radius;

  const float vz0 = recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2,
							   n3, n2, n1, npml, gtid3, gtid2, gtid1, true);
  float c1 = stencil[1]*(vz0-
			 recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-1, false))
    +stencil[2]*(recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+1, false)-
		 recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-2, false))
    +stencil[3]*(recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+2, false)-
		 recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-3, false))
    +stencil[4]*(recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1+3, false)-
		 recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1-4, false));

  c1 *= _dz2;
  const float c2 = pml_fused_second_from_p1(p1, base, (size_t)stride2, _dx2);
  const float c3 = pml_fused_second_from_p1(p1, base, (size_t)stride3, _dy2);

  size_t pind, ic;
  if (gtid1 < npml) {
    pind=(size_t)gtid3*npml*n2 + (size_t)gtid2*npml + gtid1;
    const float coef = c_bz_pml[gtid1];
    mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
    c1+=mem_dzz[pind];
  } else {
    ic=gtid1-n1+npml;
    pind=(size_t)n3*n2*npml+(size_t)gtid3*npml*n2 + (size_t)gtid2*npml + ic;
    const float coef = c_az_pml[ic];
    mem_dzz[pind]=mem_dzz[pind]*coef+c1*(coef-1);
    c1+=mem_dzz[pind];
  }

  p0[base]=2*__ldg(p1+base)-p0[base]
    +__ldg(cw2+base)*dt*(c1+c2+c3);
}
#endif

#ifdef CUDA3D_PML_ZFACE_SHARED_VP_DEBUG
__device__ __forceinline__ int pml_zface_shared_index(int lz, int lx, int ly) {
  const int sz = PmlZFaceSharedOut1 + 2 * PmlZFaceSharedHalo;
  const int sx = PmlZFaceSharedOut2 + 2 * PmlZFaceSharedHalo;
  return (ly * sx + lx) * sz + lz;
}

__device__ __forceinline__ float pml_zface_shared_p(const float *__restrict__ tile,
						    int lz, int lx, int ly) {
  return tile[pml_zface_shared_index(lz, lx, ly)];
}

__device__ __forceinline__ float pml_zface_shared_vx(const float *__restrict__ tile,
						     int lz, int lx, int ly,
						     float _dx2) {
  return _dx2 *
    (stencil[1] * (pml_zface_shared_p(tile, lz, lx + 1, ly) - pml_zface_shared_p(tile, lz, lx,     ly)) +
     stencil[2] * (pml_zface_shared_p(tile, lz, lx + 2, ly) - pml_zface_shared_p(tile, lz, lx - 1, ly)) +
     stencil[3] * (pml_zface_shared_p(tile, lz, lx + 3, ly) - pml_zface_shared_p(tile, lz, lx - 2, ly)) +
     stencil[4] * (pml_zface_shared_p(tile, lz, lx + 4, ly) - pml_zface_shared_p(tile, lz, lx - 3, ly)));
}

__device__ __forceinline__ float pml_zface_shared_vy(const float *__restrict__ tile,
						     int lz, int lx, int ly,
						     float _dy2) {
  return _dy2 *
    (stencil[1] * (pml_zface_shared_p(tile, lz, lx, ly + 1) - pml_zface_shared_p(tile, lz, lx, ly    )) +
     stencil[2] * (pml_zface_shared_p(tile, lz, lx, ly + 2) - pml_zface_shared_p(tile, lz, lx, ly - 1)) +
     stencil[3] * (pml_zface_shared_p(tile, lz, lx, ly + 3) - pml_zface_shared_p(tile, lz, lx, ly - 2)) +
     stencil[4] * (pml_zface_shared_p(tile, lz, lx, ly + 4) - pml_zface_shared_p(tile, lz, lx, ly - 3)));
}

#ifdef CUDA3D_PML_ZFACE_SHARED_VP_STAGE_V
__device__ __forceinline__ int pml_zface_stage_vx_index(int oz, int ox, int oy) {
  const int sx = PmlZFaceSharedOut2 + 2 * radius - 1;
  return (oy * sx + ox) * PmlZFaceSharedOut1 + oz;
}

__device__ __forceinline__ int pml_zface_stage_vy_index(int oz, int ox, int oy) {
  const int sy = PmlZFaceSharedOut3 + 2 * radius - 1;
  return (oy * PmlZFaceSharedOut2 + ox) * PmlZFaceSharedOut1 + oz;
}
#endif

__global__ void cuda_fd3d_pml_zface_shared_vp_debug_ns(float *p0,
						       const float *__restrict__ p1,
						       float *cw2, float _dy2, float _dx2, float _dz2,
						       int n3, int n2, int n1, int npml, float dt,
						       float *mem_dzz,
						       const float *__restrict__ mem_dz_v,
						       float *mem_dz_next_v) {
  extern __shared__ float shared_p[];
  const int sz = PmlZFaceSharedOut1 + 2 * PmlZFaceSharedHalo;
  const int sx = PmlZFaceSharedOut2 + 2 * PmlZFaceSharedHalo;
  const int sy = PmlZFaceSharedOut3 + 2 * PmlZFaceSharedHalo;
  const int shared_count = sz * sx * sy;
  const int tid = threadIdx.x;
#ifdef CUDA3D_PML_ZFACE_SHARED_VP_STAGE_V
  float *shared_vx = shared_p + shared_count;
  const int vx_count = PmlZFaceSharedOut1 * (PmlZFaceSharedOut2 + 2 * radius - 1) * PmlZFaceSharedOut3;
  float *shared_vy = shared_vx + vx_count;
  const int vy_count = PmlZFaceSharedOut1 * PmlZFaceSharedOut2 * (PmlZFaceSharedOut3 + 2 * radius - 1);
#endif

  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;

  const int tile_z0 = (blockIdx.z == 0) ? 0 : n1 - npml;
  const int tile_x0 = core2_lo + blockIdx.x * PmlZFaceSharedOut2;
  const int tile_y0 = core3_lo + blockIdx.y * PmlZFaceSharedOut3;
  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);

  for (int idx = tid; idx < shared_count; idx += blockDim.x) {
    const int lz = idx % sz;
    const int t = idx / sz;
    const int lx = t % sx;
    const int ly = t / sx;
    const int gz = tile_z0 + lz - PmlZFaceSharedHalo;
    const int gx = tile_x0 + lx - PmlZFaceSharedHalo;
    const int gy = tile_y0 + ly - PmlZFaceSharedHalo;
    float value = 0.0f;
    if (gz >= -radius && gz < n1 + radius &&
	gx >= -radius && gx < n2 + radius &&
	gy >= -radius && gy < n3 + radius) {
      value = __ldg(p1 + (size_t)(gy + radius) * stride3 +
		    (size_t)(gx + radius) * stride2 + (gz + radius));
    }
    shared_p[idx] = value;
  }
  __syncthreads();

#ifdef CUDA3D_PML_ZFACE_SHARED_VP_STAGE_V
  for (int idx = tid; idx < vx_count; idx += blockDim.x) {
    const int oz = idx % PmlZFaceSharedOut1;
    const int t = idx / PmlZFaceSharedOut1;
    const int ox = t % (PmlZFaceSharedOut2 + 2 * radius - 1);
    const int oy = t / (PmlZFaceSharedOut2 + 2 * radius - 1);
    const int lz = oz + PmlZFaceSharedHalo;
    const int lx = ox + PmlZFaceSharedHalo - radius;
    const int ly = oy + PmlZFaceSharedHalo;
    shared_vx[idx] = pml_zface_shared_vx(shared_p, lz, lx, ly, _dx2);
  }
  for (int idx = tid; idx < vy_count; idx += blockDim.x) {
    const int oz = idx % PmlZFaceSharedOut1;
    const int t = idx / PmlZFaceSharedOut1;
    const int ox = t % PmlZFaceSharedOut2;
    const int oy = t / PmlZFaceSharedOut2;
    const int lz = oz + PmlZFaceSharedHalo;
    const int lx = ox + PmlZFaceSharedHalo;
    const int ly = oy + PmlZFaceSharedHalo - radius;
    shared_vy[idx] = pml_zface_shared_vy(shared_p, lz, lx, ly, _dy2);
  }
  __syncthreads();
#endif

  const int output_count = PmlZFaceSharedOut1 * PmlZFaceSharedOut2 * PmlZFaceSharedOut3;
  for (int out = tid; out < output_count; out += blockDim.x) {
    const int oz = out % PmlZFaceSharedOut1;
    const int tx = out / PmlZFaceSharedOut1;
    const int ox = tx % PmlZFaceSharedOut2;
    const int oy = tx / PmlZFaceSharedOut2;
    const int gtid1 = tile_z0 + oz;
    const int gtid2 = tile_x0 + ox;
    const int gtid3 = tile_y0 + oy;

    if (!pml_fused_zface_pressure_point(gtid1, gtid2, gtid3, n3, n2, n1, npml)) continue;
    if (gtid2 >= core2_hi || gtid3 >= core3_hi) continue;

    const int lz = oz + PmlZFaceSharedHalo;
    const int lx = ox + PmlZFaceSharedHalo;
    const int ly = oy + PmlZFaceSharedHalo;
    const size_t base = (size_t)(gtid3 + radius) * stride3 +
      (size_t)(gtid2 + radius) * stride2 + (gtid1 + radius);

    const float vz0 = recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2,
							     n3, n2, n1, npml, gtid3, gtid2, gtid1, true);
    float c1 = stencil[1] * (vz0 -
			     recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1 - 1, false))
      + stencil[2] * (recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1 + 1, false) -
		      recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1 - 2, false))
      + stencil[3] * (recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1 + 2, false) -
		      recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1 - 3, false))
      + stencil[4] * (recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1 + 3, false) -
		      recompute_vz_after_update_from_old_mem(p1, mem_dz_v, mem_dz_next_v, _dz2, n3, n2, n1, npml, gtid3, gtid2, gtid1 - 4, false));
    c1 *= _dz2;

#ifdef CUDA3D_PML_ZFACE_SHARED_VP_STAGE_V
    const int vx_base = ox + radius;
    float c2 = stencil[1] * (shared_vx[pml_zface_stage_vx_index(oz, vx_base,     oy)] - shared_vx[pml_zface_stage_vx_index(oz, vx_base - 1, oy)])
      + stencil[2] * (shared_vx[pml_zface_stage_vx_index(oz, vx_base + 1, oy)] - shared_vx[pml_zface_stage_vx_index(oz, vx_base - 2, oy)])
      + stencil[3] * (shared_vx[pml_zface_stage_vx_index(oz, vx_base + 2, oy)] - shared_vx[pml_zface_stage_vx_index(oz, vx_base - 3, oy)])
      + stencil[4] * (shared_vx[pml_zface_stage_vx_index(oz, vx_base + 3, oy)] - shared_vx[pml_zface_stage_vx_index(oz, vx_base - 4, oy)]);
#else
    float c2 = stencil[1] * (pml_zface_shared_vx(shared_p, lz, lx,     ly, _dx2) - pml_zface_shared_vx(shared_p, lz, lx - 1, ly, _dx2))
      + stencil[2] * (pml_zface_shared_vx(shared_p, lz, lx + 1, ly, _dx2) - pml_zface_shared_vx(shared_p, lz, lx - 2, ly, _dx2))
      + stencil[3] * (pml_zface_shared_vx(shared_p, lz, lx + 2, ly, _dx2) - pml_zface_shared_vx(shared_p, lz, lx - 3, ly, _dx2))
      + stencil[4] * (pml_zface_shared_vx(shared_p, lz, lx + 3, ly, _dx2) - pml_zface_shared_vx(shared_p, lz, lx - 4, ly, _dx2));
#endif
    c2 *= _dx2;

#ifdef CUDA3D_PML_ZFACE_SHARED_VP_STAGE_V
    const int vy_base = oy + radius;
    float c3 = stencil[1] * (shared_vy[pml_zface_stage_vy_index(oz, ox, vy_base    )] - shared_vy[pml_zface_stage_vy_index(oz, ox, vy_base - 1)])
      + stencil[2] * (shared_vy[pml_zface_stage_vy_index(oz, ox, vy_base + 1)] - shared_vy[pml_zface_stage_vy_index(oz, ox, vy_base - 2)])
      + stencil[3] * (shared_vy[pml_zface_stage_vy_index(oz, ox, vy_base + 2)] - shared_vy[pml_zface_stage_vy_index(oz, ox, vy_base - 3)])
      + stencil[4] * (shared_vy[pml_zface_stage_vy_index(oz, ox, vy_base + 3)] - shared_vy[pml_zface_stage_vy_index(oz, ox, vy_base - 4)]);
#else
    float c3 = stencil[1] * (pml_zface_shared_vy(shared_p, lz, lx, ly,     _dy2) - pml_zface_shared_vy(shared_p, lz, lx, ly - 1, _dy2))
      + stencil[2] * (pml_zface_shared_vy(shared_p, lz, lx, ly + 1, _dy2) - pml_zface_shared_vy(shared_p, lz, lx, ly - 2, _dy2))
      + stencil[3] * (pml_zface_shared_vy(shared_p, lz, lx, ly + 2, _dy2) - pml_zface_shared_vy(shared_p, lz, lx, ly - 3, _dy2))
      + stencil[4] * (pml_zface_shared_vy(shared_p, lz, lx, ly + 3, _dy2) - pml_zface_shared_vy(shared_p, lz, lx, ly - 4, _dy2));
#endif
    c3 *= _dy2;

    size_t pind, ic;
    if (gtid1 < npml) {
      pind = (size_t)gtid3 * npml * n2 + (size_t)gtid2 * npml + gtid1;
      const float coef = c_bz_pml[gtid1];
      mem_dzz[pind] = mem_dzz[pind] * coef + c1 * (coef - 1);
      c1 += mem_dzz[pind];
    } else {
      ic = gtid1 - n1 + npml;
      pind = (size_t)n3 * n2 * npml + (size_t)gtid3 * npml * n2 + (size_t)gtid2 * npml + ic;
      const float coef = c_az_pml[ic];
      mem_dzz[pind] = mem_dzz[pind] * coef + c1 * (coef - 1);
      c1 += mem_dzz[pind];
    }

    p0[base] = 2 * __ldg(p1 + base) - p0[base] +
      __ldg(cw2 + base) * dt * (c1 + c2 + c3);
  }
}
#endif

__global__ void cuda_fd3d_p_pml_shared_ns(float *p0, float *p1, float *vy, float *vx, float *vz,
				   float *cw2, float _dy2, float _dx2, float _dz2,
				   int n3, int n2, int n1, int npml, float dt,
				   float *ay, float *by, float *ax, float *bx, float *az, float *bz,
				   float *mem_dyy, float *mem_dxx, float *mem_dzz){
  __shared__ float tile_vz[BlockSize3][BlockSize2][BlockSize1 + 2 * radius];
  __shared__ float tile_vx[BlockSize3][BlockSize2 + 2 * radius][BlockSize1];
  __shared__ float tile_vy[BlockSize3 + 2 * radius][BlockSize2][BlockSize1];

  const int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;
  const int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;
  const int gtid3 = blockIdx.z * blockDim.z + threadIdx.z;
  const int block1 = blockIdx.x * blockDim.x;
  const int block2 = blockIdx.y * blockDim.y;
  const int block3 = blockIdx.z * blockDim.z;
  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  const int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const int nthreads = blockDim.x * blockDim.y * blockDim.z;
  const int core1_lo = npml + CorePmlMargin;
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;

  if (block1 >= core1_lo && block1 + blockDim.x <= core1_hi &&
      block2 >= core2_lo && block2 + blockDim.y <= core2_hi &&
      block3 >= core3_lo && block3 + blockDim.z <= core3_hi) return;

  const int vz_count = BlockSize3 * BlockSize2 * (BlockSize1 + 2 * radius);
  for (int idx = tid; idx < vz_count; idx += nthreads) {
    const int l1 = idx % (BlockSize1 + 2 * radius);
    const int t = idx / (BlockSize1 + 2 * radius);
    const int l2 = t % BlockSize2;
    const int l3 = t / BlockSize2;
    const int t1 = block1 + l1;
    const int g2 = block2 + l2;
    const int g3 = block3 + l3;
    float value = 0.0f;
    if (t1 >= 0 && t1 < n1 + 2 * radius && g2 < n2 && g3 < n3) {
      const int t2 = g2 + radius;
      const int t3 = g3 + radius;
      value = vz[(size_t)t3 * stride3 + (size_t)t2 * stride2 + t1];
    }
    tile_vz[l3][l2][l1] = value;
  }

  const int vx_count = BlockSize3 * (BlockSize2 + 2 * radius) * BlockSize1;
  for (int idx = tid; idx < vx_count; idx += nthreads) {
    const int l1 = idx % BlockSize1;
    const int t = idx / BlockSize1;
    const int l2 = t % (BlockSize2 + 2 * radius);
    const int l3 = t / (BlockSize2 + 2 * radius);
    const int g1 = block1 + l1;
    const int t2 = block2 + l2;
    const int g3 = block3 + l3;
    float value = 0.0f;
    if (g1 < n1 && t2 >= 0 && t2 < n2 + 2 * radius && g3 < n3) {
      const int t1 = g1 + radius;
      const int t3 = g3 + radius;
      value = vx[(size_t)t3 * stride3 + (size_t)t2 * stride2 + t1];
    }
    tile_vx[l3][l2][l1] = value;
  }

  const int vy_count = (BlockSize3 + 2 * radius) * BlockSize2 * BlockSize1;
  for (int idx = tid; idx < vy_count; idx += nthreads) {
    const int l1 = idx % BlockSize1;
    const int t = idx / BlockSize1;
    const int l2 = t % BlockSize2;
    const int l3 = t / BlockSize2;
    const int g1 = block1 + l1;
    const int g2 = block2 + l2;
    const int t3 = block3 + l3;
    float value = 0.0f;
    if (g1 < n1 && g2 < n2 && t3 >= 0 && t3 < n3 + 2 * radius) {
      const int t1 = g1 + radius;
      const int t2 = g2 + radius;
      value = vy[(size_t)t3 * stride3 + (size_t)t2 * stride2 + t1];
    }
    tile_vy[l3][l2][l1] = value;
  }

  __syncthreads();

  if (gtid1 >= n1 || gtid2 >= n2 || gtid3 >= n3) return;

  if ((gtid1 >= core1_lo) && (gtid1 < core1_hi) &&
      (gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
      (gtid3 >= core3_lo) && (gtid3 < core3_hi)) return;

  const int l1 = threadIdx.x + radius;
  const int l2 = threadIdx.y + radius;
  const int l3 = threadIdx.z + radius;

  float c1 = stencil[1] * (tile_vz[threadIdx.z][threadIdx.y][l1]     - tile_vz[threadIdx.z][threadIdx.y][l1 - 1])
           + stencil[2] * (tile_vz[threadIdx.z][threadIdx.y][l1 + 1] - tile_vz[threadIdx.z][threadIdx.y][l1 - 2])
           + stencil[3] * (tile_vz[threadIdx.z][threadIdx.y][l1 + 2] - tile_vz[threadIdx.z][threadIdx.y][l1 - 3])
           + stencil[4] * (tile_vz[threadIdx.z][threadIdx.y][l1 + 3] - tile_vz[threadIdx.z][threadIdx.y][l1 - 4]);

  float c2 = stencil[1] * (tile_vx[threadIdx.z][l2][threadIdx.x]     - tile_vx[threadIdx.z][l2 - 1][threadIdx.x])
           + stencil[2] * (tile_vx[threadIdx.z][l2 + 1][threadIdx.x] - tile_vx[threadIdx.z][l2 - 2][threadIdx.x])
           + stencil[3] * (tile_vx[threadIdx.z][l2 + 2][threadIdx.x] - tile_vx[threadIdx.z][l2 - 3][threadIdx.x])
           + stencil[4] * (tile_vx[threadIdx.z][l2 + 3][threadIdx.x] - tile_vx[threadIdx.z][l2 - 4][threadIdx.x]);

  float c3 = stencil[1] * (tile_vy[l3][threadIdx.y][threadIdx.x]     - tile_vy[l3 - 1][threadIdx.y][threadIdx.x])
           + stencil[2] * (tile_vy[l3 + 1][threadIdx.y][threadIdx.x] - tile_vy[l3 - 2][threadIdx.y][threadIdx.x])
           + stencil[3] * (tile_vy[l3 + 2][threadIdx.y][threadIdx.x] - tile_vy[l3 - 3][threadIdx.y][threadIdx.x])
           + stencil[4] * (tile_vy[l3 + 3][threadIdx.y][threadIdx.x] - tile_vy[l3 - 4][threadIdx.y][threadIdx.x]);

  c1 *= _dz2;
  c2 *= _dx2;
  c3 *= _dy2;

  float vzz_loc = c1;
  float vxx_loc = c2;
  float vyy_loc = c3;
  size_t pind, ic;

  if (gtid1 < npml) {
    pind = (size_t)gtid3 * npml * n2 + (size_t)gtid2 * npml + gtid1;
    const float coef = c_bz_pml[gtid1];
    mem_dzz[pind] = mem_dzz[pind] * coef + c1 * (coef - 1.0f);
    vzz_loc += mem_dzz[pind];
  }
  if (gtid1 >= n1 - npml) {
    ic = gtid1 - n1 + npml;
    pind = (size_t)n3 * n2 * npml + (size_t)gtid3 * npml * n2 + (size_t)gtid2 * npml + ic;
    const float coef = c_az_pml[ic];
    mem_dzz[pind] = mem_dzz[pind] * coef + c1 * (coef - 1.0f);
    vzz_loc += mem_dzz[pind];
  }

  if (gtid2 < npml) {
    pind = (size_t)gtid3 * npml * n1 + (size_t)gtid2 * n1 + gtid1;
    const float coef = c_bx_pml[gtid2];
    mem_dxx[pind] = mem_dxx[pind] * coef + c2 * (coef - 1.0f);
    vxx_loc += mem_dxx[pind];
  }
  if (gtid2 >= n2 - npml) {
    ic = gtid2 - n2 + npml;
    pind = (size_t)n3 * npml * n1 + (size_t)gtid3 * (npml * n1) + ic * n1 + gtid1;
    const float coef = c_ax_pml[ic];
    mem_dxx[pind] = mem_dxx[pind] * coef + c2 * (coef - 1.0f);
    vxx_loc += mem_dxx[pind];
  }

  if (gtid3 < npml) {
    pind = (size_t)gtid3 * n2 * n1 + (size_t)gtid2 * n1 + gtid1;
    const float coef = c_by_pml[gtid3];
    mem_dyy[pind] = mem_dyy[pind] * coef + c3 * (coef - 1.0f);
    vyy_loc += mem_dyy[pind];
  }
  if (gtid3 >= n3 - npml) {
    ic = gtid3 - n3 + npml;
    pind = (size_t)npml * n2 * n1 + ic * n2 * n1 + (size_t)gtid2 * n1 + gtid1;
    const float coef = c_ay_pml[ic];
    mem_dyy[pind] = mem_dyy[pind] * coef + c3 * (coef - 1.0f);
    vyy_loc += mem_dyy[pind];
  }

  const int t1 = gtid1 + radius;
  const int t2 = gtid2 + radius;
  const int t3 = gtid3 + radius;
  const size_t outIndex = (size_t)t3 * stride3 + (size_t)t2 * stride2 + t1;
  p0[outIndex] = 2.0f * p1[outIndex] - p0[outIndex] + cw2[outIndex] * dt * (vzz_loc + vxx_loc + vyy_loc);
}

// !!!!!!!!!!! not correct, but worth trying to improve with shared memory in the future
__global__ void cuda_fd3d_v_pml2(float *p1, float *vy, float *vx, float *vz,
				float _dy2, float _dx2, float _dz2, 
				int n3, int n2, int n1, int npml, float dt,
				float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
				float *mem_dy, float *mem_dx, float *mem_dz){
  float c1, c2, c3;
  bool validr = true;
  bool validw = true;
  int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;
  int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;
  int gtid3 = blockIdx.z * blockDim.z + threadIdx.z;
  const int ltid1 = threadIdx.x;
  const int ltid2 = threadIdx.y;
  const int work1 = blockDim.x;
  const int work2 = blockDim.y;
  __shared__ float tile[BlockSize2 + 2 * radius][BlockSize1 + 2 * radius];

  float infront[radius];
  float behind[radius];
  float current;

  size_t inIndex = 0;
  size_t outIndex = 0;
  size_t ic, pind;

  const int lt1 = ltid1 + radius;
  const int lt2 = ltid2 + radius;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  int i, i3;

  // Advance inputIndex to start of inner volume, the begining of the area
  inIndex += radius * stride2 + radius;

    // Advance inputIndex to target element
  //  inIndex += gtid2 * stride2 + gtid1;

  // Check in bounds
  //  while (gtid1 < n1+radius && gtid2 < n2+radius){
  if ((gtid1 >= n1 + radius) ||(gtid2 >= n2 + radius) || gtid3 >=n3+radius) validr = false;
  if ((gtid1 >= n1) || (gtid2 >= n2) || gtid3 >=n3) validw = false;

  //    // Advance inputIndex to target element
  inIndex += gtid2 * stride2 + gtid1;


    // Preload the "infront" and "behind" data
    for (i = radius - 2 ; i >= 0 ; i--){
      if (validr) behind[i] = p1[inIndex];
      inIndex += stride3;
    }

    if (validr)	current = p1[inIndex];

    outIndex = inIndex;
    inIndex += stride3;

    for (i = 0 ; i < radius ; i++){
      if (validr) infront[i] = p1[inIndex];
      inIndex += stride3;
    }

    // Step through the zx-planes
    //#pragma unroll 9
    //    for (i3 = 0 ; i3 < n3 ; i3++){
      // Advance the slice (move the thread-front)
      for (i = radius - 1 ; i > 0 ; i--) 
	behind[i] = behind[i - 1];

      behind[0] = current;
      current = infront[0];
#pragma unroll 4
      for (i = 0 ; i < radius - 1 ; i++) 
	infront[i] = infront[i + 1];
      
      if (validr) 
	infront[radius - 1] = p1[inIndex];
    
      inIndex += stride3; // needed???
      outIndex += stride3;
      __syncthreads();
      
      // Update the data slice in the local tile
      // Halo above & below // why not 9 tile initiated???
      if (ltid2 < radius){
	tile[ltid2][lt1]                  = p1[outIndex - radius * stride2];
	tile[ltid2 + work2 + radius][lt1] = p1[outIndex + work2 * stride2]; // not might out of bound????
	//	ic=(int)(MIN(gtid2+work2+radius, n2+2*radius));
	//	tile[ltid2 + work2 + radius][lt1] = p1[ic*stride2+gtid1+radius];
      }
      // Halo left & right
      if (ltid1 < radius){
	tile[lt2][ltid1]                  = p1[outIndex - radius];
	tile[lt2][ltid1 + work1 + radius] = p1[outIndex + work1];
      }

      tile[lt2][lt1] = current;
      __syncthreads();

      // Compute the output value
      c1=c2=c3=0.0;

      c1=stencil[1]*(tile[lt2][lt1+1]-tile[lt2][lt1])
	+stencil[2]*(tile[lt2][lt1+2]-tile[lt2][lt1-1])
	+stencil[3]*(tile[lt2][lt1+3]-tile[lt2][lt1-2])
	+stencil[4]*(tile[lt2][lt1+4]-tile[lt2][lt1-3]);

      c2=stencil[1]*(tile[lt2+1][lt1]-tile[lt2][lt1])
	+stencil[2]*(tile[lt2+2][lt1]-tile[lt2-1][lt1])
	+stencil[3]*(tile[lt2+3][lt1]-tile[lt2-2][lt1])
	+stencil[4]*(tile[lt2+4][lt1]-tile[lt2-3][lt1]);

      c3=stencil[1]*(infront[0]-current)
	+stencil[2]*(infront[1]-behind[0])
	+stencil[3]*(infront[2]-behind[1])
	+stencil[4]*(infront[3]-behind[2]);
      c1*=_dz2;
      c2*=_dx2;
      c3*=_dy2;
    //if (validw) {vz[outIndex]+=(-dt*c1);vx[outIndex]+=(-dt*c2);vy[outIndex]+=(-dt*c3);}

      if (validw) {
	vz[outIndex]=c1;
	vx[outIndex]=c2;
	vy[outIndex]=c3;
		
	//PML Zone
	//Start Z-PML
	if(gtid1<npml) {
	  //Apply PML in Z-direction, pind is index inside PML zone
	  pind=gtid3*n2*npml + gtid2*npml + gtid1;
	  mem_dz[pind]=mem_dz[pind]*bz_h[gtid1]+c1*(bz_h[gtid1]-1.);
	  vz[outIndex]+=mem_dz[pind];
	}
	if (gtid1>=n1-npml) {
	  //Apply PML in Z-direction, pind is index inside PML zone
	  ic=gtid1-n1+npml;
	  pind=  n3*n2*npml+gtid3*n2*npml + gtid2*npml + ic;
	  mem_dz[pind]=mem_dz[pind]*az_h[ic]+c1*(az_h[ic]-1.);
	  vz[outIndex]+=mem_dz[pind];
	}
	//End ZPML
	
	//Start X-PML
	if (gtid2<npml){
	  //Apply PML in X-direction, pind is index inside PML zone
	  pind=gtid3*npml*n1 + gtid2*n1 + gtid1;
	  mem_dx[pind]=mem_dx[pind]*bx_h[gtid2]+c2*(bx_h[gtid2]-1.);
	  vx[outIndex]+=mem_dx[pind];
	}
	if (gtid2>=n2-npml){
	  //Apply PML in X-direction, pind is index inside PML zone
	  ic=gtid2-n2+npml;
	  pind=n3*npml*n1+gtid3*(npml*n1) + ic*n1 + gtid1;
	  mem_dx[pind]=mem_dx[pind]*ax_h[ic]+c2*(ax_h[ic]-1.);
	  vx[outIndex]+=mem_dx[pind];
	}
	//End XPML
	
	// Start Y-PML
	if(gtid3<npml) {
	  //Apply PML in Y-direction, pind is index inside PML zone
	  pind=gtid3*n2*n1 + gtid2*n1 + gtid1;
	  mem_dy[pind]=mem_dy[pind]*by_h[gtid3]+c3*(by_h[gtid3]-1.);
	  vy[outIndex]+=mem_dy[pind];
	}
	if (gtid3>=n3-npml){
	  //Apply PML in Y-direction, pind is index inside PML zone
	  ic=gtid3-n3+npml;
	  pind= npml*n2*n1+ic*n2*n1 + gtid2*(n1) + gtid1;
	  mem_dy[pind]=mem_dy[pind]*ay_h[ic]+c3*(ay_h[ic]-1.);
	  vy[outIndex]+=mem_dy[pind];
	}
	//END YPML
      } // end valid domain
      //    }//end step through xz plane/ y direction
    //    gtid1 =gtid1+ blockDim.x * gridDim.x;  //???
    //    gtid2 =gtid2+ blockDim.y * gridDim.y;/// ??
    //  }
}

/*< step forward: 3-D FD, order=8 >*/
// !!!!!!!!!!! not correct, but worth trying to improve with shared memory in the future
__global__ void cuda_fd3d_p_pml2(float *p0, float *p1, float *vy, float *vx, float *vz,
				float *vyy, float *vxx, float *vzz,
				float *cw2, float _dy2, float _dx2, float _dz2, 
				int n3, int n2, int n1, int npml, float dt, 
				float *ay, float *by, float *ax, float *bx, float *az, float *bz,
				float *mem_dyy, float *mem_dxx, float *mem_dzz){
  float c1, c2, c3;
  bool validr = true;
  bool validw = true;
  int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;
  int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;
  int gtid3 = blockIdx.z * blockDim.z + threadIdx.z;
  const int ltid1 = threadIdx.x;
  const int ltid2 = threadIdx.y;
  const int work1 = blockDim.x;
  const int work2 = blockDim.y;
  __shared__ float tile1[BlockSize2 + 2 * radius][BlockSize1 + 2 * radius];
  __shared__ float tile2[BlockSize2 + 2 * radius][BlockSize1 + 2 * radius];
  // comment by zz    __shared__ float tile3[BlockSize2 + 2 * radius][BlockSize1 + 2 * radius];

  float infront1[radius];float infront2[radius];float infront3[radius];
  float behind1[radius];float behind2[radius];float behind3[radius];
  float current1;float current2;float current3;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);

  size_t inIndex = 0;
  size_t outIndex = 0;
  size_t ic, pind;
  int i, i3;


  const int lt1 = ltid1 + radius;
  const int lt2 = ltid2 + radius;

  // Advance inputIndex to start of inner volume that skips the radius region
  inIndex += radius * stride2 + radius;

  // Check in bounds
  //  while ( gtid1 <n1+radius && gtid2 < n2+radius ){
  if ((gtid1 >= n1 + radius) ||(gtid2 >= n2 + radius) || gtid3>=n3+radius) validr = false;
  if ((gtid1 >= n1) || (gtid2 >= n2) || gtid3 >=n3) validw = false;

    // Advance inputIndex to target element /global target location
    inIndex += gtid2 * stride2 + gtid1;

    // Preload the "infront" and "behind" data
    for (i = radius - 2 ; i >= 0 ; i--){
      if (validr) {
	behind1[i] = vz[inIndex];
	behind2[i] = vx[inIndex];
	behind3[i] = vy[inIndex];
      }
      inIndex += stride3;
    }

    if (validr) {
      current1 = vz[inIndex];
      current2 = vx[inIndex];
      current3 = vy[inIndex];
    }
    
    outIndex = inIndex;
    inIndex += stride3;
    
    for (i = 0 ; i < radius ; i++){
      if (validr) {
	infront1[i] = vz[inIndex];
	infront2[i] = vx[inIndex];
	infront3[i] = vy[inIndex];
      }
      inIndex += stride3;
    }

    // Step through the zx-planes
    //#pragma unroll 9
    //    for (i3 = 0 ; i3 < n3 ; i3++){
      // Advance the slice (move the thread-front)
      for (i = radius - 1 ; i > 0 ; i--) {
	behind1[i] = behind1[i - 1];
	behind2[i] = behind2[i - 1];
	behind3[i] = behind3[i - 1];
      }

      behind1[0] = current1;
      behind2[0] = current2;
      behind3[0] = current3;

      current1 = infront1[0];
      current2 = infront2[0];
      current3 = infront3[0];
#pragma unroll 4
      for (i = 0 ; i < radius - 1 ; i++) {
	infront1[i] = infront1[i + 1];
	infront2[i] = infront2[i + 1];
	infront3[i] = infront3[i + 1];
      }

      if (validr) {
	infront1[radius - 1] = vz[inIndex];
	infront2[radius - 1] = vx[inIndex];
	infront3[radius - 1] = vy[inIndex];
      }

      inIndex += stride3;
      outIndex += stride3;
      __syncthreads();

      // Update the data slice in the local tile
      // Halo above & below
      if (ltid2 < radius){
	tile1[ltid2][lt1]                  = vz[outIndex - radius * stride2];
	//	ic=(int)(MIN(gtid2+work2+radius, n2+2*radius));
	tile1[ltid2 + work2 + radius][lt1] = vz[outIndex + work2 * stride2]; // outIndex might need change

	tile2[ltid2][lt1]                  = vx[outIndex - radius * stride2];
	tile2[ltid2 + work2 + radius][lt1] = vx[outIndex + work2 * stride2];
      }

      // Halo left & right
      if (ltid1 < radius){
	tile1[lt2][ltid1]                  = vz[outIndex - radius];
	tile1[lt2][ltid1 + work1 + radius] = vz[outIndex + work1];

	tile2[lt2][ltid1]                  = vx[outIndex - radius];
	tile2[lt2][ltid1 + work1 + radius] = vx[outIndex + work1];
      }

      tile1[lt2][lt1] = current1; 
      tile2[lt2][lt1] = current2; //[t2][t1] = current3;
      __syncthreads();

      // Compute the output value
  
      //c1=stencil[0]*current1;c2=stencil[0]*current2;c3=stencil[0]*current3;
      c1=c2=c3=0.0;

      c1=stencil[1]*(tile1[lt2][lt1]-tile1[lt2][lt1-1])
	+stencil[2]*(tile1[lt2][lt1+1]-tile1[lt2][lt1-2])
	+stencil[3]*(tile1[lt2][lt1+2]-tile1[lt2][lt1-3])
	+stencil[4]*(tile1[lt2][lt1+3]-tile1[lt2][lt1-4]);

      c2=stencil[1]*(tile2[lt2][lt1]-tile2[lt2-1][lt1])
	+stencil[2]*(tile2[lt2+1][lt1]-tile2[lt2-2][lt1])
	+stencil[3]*(tile2[lt2+2][lt1]-tile2[lt2-3][lt1])
	+stencil[4]*(tile2[lt2+3][lt1]-tile2[lt2-4][lt1]);

      c3=stencil[1]*(current3-behind3[0])
	+stencil[2]*(infront3[0]-behind3[1])
	+stencil[3]*(infront3[1]-behind3[2])
	+stencil[4]*(infront3[2]-behind3[3]);
      c1*=_dz2;
      c2*=_dx2;
      c3*=_dy2;


      //if (validw) {p0[outIndex]=p1[outIndex]-vel[outIndex]*(c1+c2+c3);}//{p0[outIndex]=2*p1[outIndex]-p0[outIndex]-vel[outIndex]*(c1+c2+c3);}
      //right
      if (validw) {
	vzz[outIndex]=c1;
	vxx[outIndex]=c2;
	vyy[outIndex]=c3;

	
	//Start Z-PML
	if(gtid1<npml) { 
	  //Apply PML in Z-direction  //pind is index inside PML zone
	  pind=gtid3*npml*n2 + gtid2*npml + gtid1;
	  mem_dzz[pind]=mem_dzz[pind]*bz[gtid1]+c1*(bz[gtid1]-1);
	  vzz[outIndex]+=mem_dzz[pind];
	}
	if (gtid1>=n1-npml) {
	  //Apply PML in Z-direction	  //pind is index inside PML zone
	  ic=gtid1-n1+npml;
	  pind=  n3*n2*npml+gtid3*npml*n2 + gtid2*npml + ic;
	  mem_dzz[pind]=mem_dzz[pind]*az[ic]+c1*(az[ic]-1);
	  vzz[outIndex]+=mem_dzz[pind];
	}
	//End ZPML

	//Start X-PML
	if (gtid2<npml){
	  //Apply PML in X-direction	  //pind is index inside PML zone
	  pind=gtid3*npml*n1 + gtid2*n1 + gtid1;
	  mem_dxx[pind]=mem_dxx[pind]*bx[gtid2]+c2*(bx[gtid2]-1);
	  vxx[outIndex]+=mem_dxx[pind];
	}
	if (gtid2>=n2-npml){
	  //Apply PML in X-direction	  //pind is index inside PML zone
	  ic=gtid2-n2+npml;
	  pind=n3*npml*n1+gtid3*(npml*n1) + ic*n1 + gtid1;
	  mem_dxx[pind]=mem_dxx[pind]*ax[ic]+c2*(ax[ic]-1);
	  vxx[outIndex]+=mem_dxx[pind];
	}
	//End XPML

	// Start Y-PML
	if(gtid3<npml) {
	  //Apply PML in Y-direction	  //pind is index inside PML zone
	  pind=gtid3*n2*n1 + gtid2*n1 + gtid1;
	  mem_dyy[pind]=mem_dyy[pind]*by[gtid3]+c3*(by[gtid3]-1);
	  vyy[outIndex]+=mem_dyy[pind];
	}
	if (gtid3>=n3-npml){
	  //Apply PML in Y-direction	  //pind is index inside PML zone
	  ic=gtid3-n3+npml;
	  pind= npml*n2*n1+ic*n2*n1 + gtid2*(n1) + gtid1;
	  mem_dyy[pind]=mem_dyy[pind]*ay[ic]+c3*(ay[ic]-1);
	  vyy[outIndex]+=mem_dyy[pind];
	}
	//END YPML
	
	p0[outIndex]=2*p1[outIndex]-p0[outIndex]
	  +cw2[outIndex]*dt*(vzz[outIndex]+vxx[outIndex]+vyy[outIndex]);

      }// validw end

      //    }// end through xz plane
    //    gtid1 =gtid1+ blockDim.x * gridDim.x; //???
    //    gtid2 =gtid2+ blockDim.y * gridDim.y; //???
    //  }
}
