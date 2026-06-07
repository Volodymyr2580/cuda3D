#include "single_solver.h"
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


__global__ void cuda_fd3d_v_pml_ns(float *p1, float *vy, float *vx, float *vz,
				   float _dy2, float _dx2, float _dz2, 
				   int n3, int n2, int n1, int npml, float dt,
				   float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
				   float *mem_dy, float *mem_dx, float *mem_dz){
  float c1, c2, c3;
  //  bool validr = true;
  //  bool validw = true;
  int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;
  int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;
  int gtid3 = blockIdx.z * blockDim.z + threadIdx.z;
  //  const int ltid1 = threadIdx.x;
  //  const int ltid2 = threadIdx.y;
  //  const int ltid3 = threadIdx.z;
  //  const int work1 = blockDim.x;
  //  const int work2 = blockDim.y;
  //  const int work3 = blockDim.z;
  //  __shared__ float tile[BlockSize2 + 2 * radius][BlockSize1 + 2 * radius];

  //  float infront[radius];
  //  float behind[radius];
  //  float current;

  size_t inIndex = 0;
  size_t outIndex = 0;

  //  const int lt1 = ltid1 + radius;
  //  const int lt2 = ltid2 + radius;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  size_t ic, pind;
  int t1, t2, t3;
  size_t ts3, ts2;
  const int core1_lo = npml + CorePmlMargin;
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;
  // Advance inputIndex to start of inner volume, the begining of the area
  inIndex += radius * stride2 + radius +radius*stride3;

  // Advance inputIndex to target element
  //  inIndex += gtid2 * stride2 + gtid1;

  // Check in bounds
  //  while (gtid1 < n1+radius && gtid2 < n2+radius){
  //  if ((gtid1 >= n1 + radius) ||(gtid2 >= n2 + radius) || (gtid3 >= n3+radius)) validr = false;
  //  if ((gtid1 >= n1) || (gtid2 >= n2) || (gtid3 >= n3)) validw = false;

  if( gtid1 <n1 && gtid2 <n2 && gtid3 < n3){
    const bool need_vz = !((gtid1 >= core1_lo + 3) && (gtid1 < core1_hi - 4) &&
			   (gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
			   (gtid3 >= core3_lo) && (gtid3 < core3_hi));
    const bool need_vx = !((gtid1 >= core1_lo) && (gtid1 < core1_hi) &&
			   (gtid2 >= core2_lo + 3) && (gtid2 < core2_hi - 4) &&
			   (gtid3 >= core3_lo) && (gtid3 < core3_hi));
    const bool need_vy = !((gtid1 >= core1_lo) && (gtid1 < core1_hi) &&
			   (gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
			   (gtid3 >= core3_lo + 3) && (gtid3 < core3_hi - 4));

    if (!need_vz && !need_vx && !need_vy) return;

    // Advance inputIndex to target element
    inIndex += gtid3 * stride3 + gtid2 * stride2 + gtid1;
    outIndex = inIndex;

    // Compute the output value
    c1=c2=c3=0.0;
    t1=gtid1+radius;
    t2=gtid2+radius;
    t3=gtid3+radius;
    ts3=t3*stride3;
    ts2=t2*stride2;

    if (need_vz)
      c1=stencil[1]*(p1[ts3+ts2+t1+1]-p1[ts3+ts2+t1  ])
	+stencil[2]*(p1[ts3+ts2+t1+2]-p1[ts3+ts2+t1-1])
	+stencil[3]*(p1[ts3+ts2+t1+3]-p1[ts3+ts2+t1-2])
	+stencil[4]*(p1[ts3+ts2+t1+4]-p1[ts3+ts2+t1-3]);
    
    if (need_vx)
      c2=stencil[1]*(p1[ts3+(t2+1)*stride2+t1]-p1[ts3+(t2  )*stride2+t1])
	+stencil[2]*(p1[ts3+(t2+2)*stride2+t1]-p1[ts3+(t2-1)*stride2+t1])
	+stencil[3]*(p1[ts3+(t2+3)*stride2+t1]-p1[ts3+(t2-2)*stride2+t1])
	+stencil[4]*(p1[ts3+(t2+4)*stride2+t1]-p1[ts3+(t2-3)*stride2+t1]);
    
    if (need_vy)
      c3=stencil[1]*(p1[(t3+1)*stride3+ts2+t1]-p1[(t3  )*stride3+ts2+t1])
	+stencil[2]*(p1[(t3+2)*stride3+ts2+t1]-p1[(t3-1)*stride3+ts2+t1])
	+stencil[3]*(p1[(t3+3)*stride3+ts2+t1]-p1[(t3-2)*stride3+ts2+t1])
	+stencil[4]*(p1[(t3+4)*stride3+ts2+t1]-p1[(t3-3)*stride3+ts2+t1]);
    
    c1*=_dz2;
    c2*=_dx2;
    c3*=_dy2;
    //if (validw) {vz[outIndex]+=(-dt*c1);vx[outIndex]+=(-dt*c2);vy[outIndex]+=(-dt*c3);}
      //      if (validw) {
    if (need_vz) vz[outIndex]=c1;
    if (need_vx) vx[outIndex]=c2;
    if (need_vy) vy[outIndex]=c3;
    
    //PML Zone
    //Start Z-PML
    if(need_vz && gtid1<npml) {
      //Apply PML in Z-direction, pind is index inside PML zone
      pind=gtid3*n2*npml + gtid2*npml + gtid1;
      mem_dz[pind]=mem_dz[pind]*bz_h[gtid1]+c1*(bz_h[gtid1]-1);
      vz[outIndex]+=mem_dz[pind];
    }
    if (need_vz && gtid1>=n1-npml) {
      //Apply PML in Z-direction, pind is index inside PML zone
      ic=gtid1-n1+npml;
      pind=  n3*n2*npml+gtid3*n2*npml + gtid2*npml + ic;
      mem_dz[pind]=mem_dz[pind]*az_h[ic]+c1*(az_h[ic]-1);
      vz[outIndex]+=mem_dz[pind];
    }
    //End ZPML
    
    //Start X-PML
    if (need_vx && gtid2<npml){
      //Apply PML in X-direction, pind is index inside PML zone
      pind=gtid3*npml*n1 + gtid2*n1 + gtid1;
      mem_dx[pind]=mem_dx[pind]*bx_h[gtid2]+c2*(bx_h[gtid2]-1);
      vx[outIndex]+=mem_dx[pind];
    }
    if (need_vx && gtid2>=n2-npml){
      //Apply PML in X-direction, pind is index inside PML zone
      ic=gtid2-n2+npml;
      pind=n3*npml*n1+gtid3*(npml*n1) + ic*n1 + gtid1;
      mem_dx[pind]=mem_dx[pind]*ax_h[ic]+c2*(ax_h[ic]-1);
      vx[outIndex]+=mem_dx[pind];
    }
    //End XPML
    
    // Start Y-PML
    if(need_vy && gtid3<npml) {
      //Apply PML in Y-direction, pind is index inside PML zone
      pind=gtid3*n2*n1 + gtid2*n1 + gtid1;
      mem_dy[pind]=mem_dy[pind]*by_h[gtid3]+c3*(by_h[gtid3]-1);
      vy[outIndex]+=mem_dy[pind];
    }
    if (need_vy && gtid3>=n3-npml){
      //Apply PML in Y-direction, pind is index inside PML zone
      ic=gtid3-n3+npml;
      pind= npml*n2*n1+ic*n2*n1 + gtid2*(n1) + gtid1;
      mem_dy[pind]=mem_dy[pind]*ay_h[ic]+c3*(ay_h[ic]-1);
      vy[outIndex]+=mem_dy[pind];
    }
    //END YPML

    //      } // end valid domain
  }//end step through xz plane/ y direction
  //    gtid1 =gtid1+ blockDim.x * gridDim.x;  //???
  //    gtid2 =gtid2+ blockDim.y * gridDim.y;/// ??
  //  }
}

__device__ __forceinline__ float core_second_axis(const float *p, size_t base, size_t stride){
  return -2.8751201527567405f * p[base]
    + 1.6234617233276367f * (p[base + stride] + p[base - stride])
    - 0.21382331848144528f * (p[base + 2 * stride] + p[base - 2 * stride])
    + 0.030927128261990015f * (p[base + 3 * stride] + p[base - 3 * stride])
    - 0.003195444742838541f * (p[base + 4 * stride] + p[base - 4 * stride])
    + 0.0002028528849283854f * (p[base + 5 * stride] + p[base - 5 * stride])
    - 0.000013351440429687502f * (p[base + 6 * stride] + p[base - 6 * stride])
    + 0.000000486568528778699f * (p[base + 7 * stride] + p[base - 7 * stride]);
}

enum { CoreStencilRadius = 7 };

__global__ void cuda_fd3d_p_core_ns(float *p0, float *p1, float *cw2,
				   float _dy2, float _dx2, float _dz2,
				   int n3, int n2, int n1, int npml, float dt){
  int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;
  int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;
  int gtid3 = blockIdx.z * blockDim.z + threadIdx.z;
  __shared__ float z_tile[PBlockSize3][PBlockSize2][PBlockSize1 + 2 * CoreStencilRadius];

  const int core1_lo = npml + CorePmlMargin;
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;

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
    const int left1 = (int)blockIdx.x * blockDim.x + threadIdx.x - CoreStencilRadius;
    const int right1 = (int)blockIdx.x * blockDim.x + blockDim.x + threadIdx.x;
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


__global__ void cuda_fd3d_p_pml_ns(float *p0, float *p1, float *vy, float *vx, float *vz,
				   float *cw2, float _dy2, float _dx2, float _dz2, 
				   int n3, int n2, int n1, int npml, float dt,
				   float *ay, float *by, float *ax, float *bx, float *az, float *bz,
				   float *mem_dyy, float *mem_dxx, float *mem_dzz){
  float c1, c2, c3;
  float vzz_loc, vxx_loc, vyy_loc;
  //  bool validr = true;
  //  bool validw = true;
  int gtid1 = blockIdx.x * blockDim.x + threadIdx.x;
  int gtid2 = blockIdx.y * blockDim.y + threadIdx.y;
  int gtid3 = blockIdx.z * blockDim.z + threadIdx.z;
  //  const int ltid1 = threadIdx.x;
  //  const int ltid2 = threadIdx.y;
  //  const int work1 = blockDim.x;
  //  const int work2 = blockDim.y;
  //  __shared__ float tile1[BlockSize2 + 2 * radius][BlockSize1 + 2 * radius];
  //  __shared__ float tile2[BlockSize2 + 2 * radius][BlockSize1 + 2 * radius];
  // comment by zz    __shared__ float tile3[BlockSize2 + 2 * radius][BlockSize1 + 2 * radius];

  float infront1[radius];float infront2[radius];float infront3[radius];
  float behind1[radius];float behind2[radius];float behind3[radius];
  float current1;float current2;float current3;

  const int stride2 = n1 + 2 * radius;
  const int stride3 = stride2 * (n2 + 2 * radius);
  
  size_t inIndex = 0;
  size_t outIndex = 0;
  size_t ic, pind;
  int t1, t2, t3;
  size_t ts3, ts2;
  const int core1_lo = npml + CorePmlMargin;
  const int core2_lo = npml + CorePmlMargin;
  const int core3_lo = npml + CorePmlMargin;
  const int core1_hi = n1 - npml - CorePmlMargin;
  const int core2_hi = n2 - npml - CorePmlMargin;
  const int core3_hi = n3 - npml - CorePmlMargin;


  //  const int lt1 = ltid1 + radius;
  //  const int lt2 = ltid2 + radius;
  //  const int lt3 = ltid3 + radius;

  // Advance inputIndex to start of inner volume that skips the radius region
  inIndex += radius * stride2 + radius + radius*stride3;

  // Check in bounds
  //  while ( gtid1 <n1+radius && gtid2 < n2+radius ){
  //  if ((gtid1 >= n1 + radius) ||(gtid2 >= n2 + radius) || (gtid3>=n3+radius) ) validr = false;
  //  if ((gtid1 >= n1) || (gtid2 >= n2) || (gtid3 >=n3) ) validw = false;

  if (gtid1 < n1 && gtid2 < n2 && gtid3 < n3){
    if ((gtid1 >= core1_lo) && (gtid1 < core1_hi) &&
	(gtid2 >= core2_lo) && (gtid2 < core2_hi) &&
	(gtid3 >= core3_lo) && (gtid3 < core3_hi)) return;

    // Advance inputIndex to target element /global target location
    inIndex += gtid3 * stride3 + gtid2 * stride2 + gtid1;
    outIndex = inIndex;

      // Compute the output value
  
      //c1=stencil[0]*current1;c2=stencil[0]*current2;c3=stencil[0]*current3;
      c1=c2=c3=0.0;
      t1=gtid1+radius;
      t2=gtid2+radius;
      t3=gtid3+radius;
      ts3=t3*stride3;
      ts2=t2*stride2;

      c1=stencil[1]*(vz[ts3+ts2+t1  ]-vz[ts3+ts2+t1-1])
	+stencil[2]*(vz[ts3+ts2+t1+1]-vz[ts3+ts2+t1-2])
	+stencil[3]*(vz[ts3+ts2+t1+2]-vz[ts3+ts2+t1-3])
	+stencil[4]*(vz[ts3+ts2+t1+3]-vz[ts3+ts2+t1-4]);


      c2=stencil[1]*(vx[ts3+(t2  )*stride2+t1]-vx[ts3+(t2-1)*stride2+t1])
	+stencil[2]*(vx[ts3+(t2+1)*stride2+t1]-vx[ts3+(t2-2)*stride2+t1])
	+stencil[3]*(vx[ts3+(t2+2)*stride2+t1]-vx[ts3+(t2-3)*stride2+t1])
	+stencil[4]*(vx[ts3+(t2+3)*stride2+t1]-vx[ts3+(t2-4)*stride2+t1]);
      

      c3=stencil[1]*(vy[(t3  )*stride3+ts2+t1]-vy[(t3-1)*stride3+ts2+t1])
	+stencil[2]*(vy[(t3+1)*stride3+ts2+t1]-vy[(t3-2)*stride3+ts2+t1])
	+stencil[3]*(vy[(t3+2)*stride3+ts2+t1]-vy[(t3-3)*stride3+ts2+t1])
	+stencil[4]*(vy[(t3+3)*stride3+ts2+t1]-vy[(t3-4)*stride3+ts2+t1]);

      c1*=_dz2;
      c2*=_dx2;
      c3*=_dy2;

      //if (validw) {p0[outIndex]=p1[outIndex]-vel[outIndex]*(c1+c2+c3);}//{p0[outIndex]=2*p1[outIndex]-p0[outIndex]-vel[outIndex]*(c1+c2+c3);}
      //right
      //      if (validw) {
	vzz_loc=c1;
	vxx_loc=c2;
	vyy_loc=c3;


	//Start Z-PML
	if(gtid1<npml) { 
	  //Apply PML in Z-direction  //pind is index inside PML zone
	  pind=gtid3*npml*n2 + gtid2*npml + gtid1;
	  mem_dzz[pind]=mem_dzz[pind]*bz[gtid1]+c1*(bz[gtid1]-1);
	  vzz_loc+=mem_dzz[pind];
	}
	if (gtid1>=n1-npml) {
	  //Apply PML in Z-direction	  //pind is index inside PML zone
	  ic=gtid1-n1+npml;
	  pind=  n3*n2*npml+gtid3*npml*n2 + gtid2*npml + ic;
	  mem_dzz[pind]=mem_dzz[pind]*az[ic]+c1*(az[ic]-1);
	  vzz_loc+=mem_dzz[pind];
	}
	//End ZPML

	//Start X-PML
	if (gtid2<npml){
	  //Apply PML in X-direction	  //pind is index inside PML zone
	  pind=gtid3*npml*n1 + gtid2*n1 + gtid1;
	  mem_dxx[pind]=mem_dxx[pind]*bx[gtid2]+c2*(bx[gtid2]-1);
	  vxx_loc+=mem_dxx[pind];
	}
	if (gtid2>=n2-npml){
	  //Apply PML in X-direction	  //pind is index inside PML zone
	  ic=gtid2-n2+npml;
	  pind=n3*npml*n1+gtid3*(npml*n1) + ic*n1 + gtid1;
	  mem_dxx[pind]=mem_dxx[pind]*ax[ic]+c2*(ax[ic]-1);
	  vxx_loc+=mem_dxx[pind];
	}
	//End XPML

	// Start Y-PML
	if(gtid3<npml) {
	  //Apply PML in Y-direction	  //pind is index inside PML zone
	  pind=gtid3*n2*n1 + gtid2*n1 + gtid1;
	  mem_dyy[pind]=mem_dyy[pind]*by[gtid3]+c3*(by[gtid3]-1);
	  vyy_loc+=mem_dyy[pind];
	}
	if (gtid3>=n3-npml){
	  //Apply PML in Y-direction	  //pind is index inside PML zone
	  ic=gtid3-n3+npml;
	  pind= npml*n2*n1+ic*n2*n1 + gtid2*(n1) + gtid1;
	  mem_dyy[pind]=mem_dyy[pind]*ay[ic]+c3*(ay[ic]-1);
	  vyy_loc+=mem_dyy[pind];
	}
	//END YPML

	p0[outIndex]=2*p1[outIndex]-p0[outIndex]
	  +cw2[outIndex]*dt*(vzz_loc+vxx_loc+vyy_loc);

	//      }// validw end

  }// end through xz plane
    //    gtid1 =gtid1+ blockDim.x * gridDim.x; //???
    //    gtid2 =gtid2+ blockDim.y * gridDim.y; //???
    //  }
}

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
    mem_dzz[pind] = mem_dzz[pind] * bz[gtid1] + c1 * (bz[gtid1] - 1.0f);
    vzz_loc += mem_dzz[pind];
  }
  if (gtid1 >= n1 - npml) {
    ic = gtid1 - n1 + npml;
    pind = (size_t)n3 * n2 * npml + (size_t)gtid3 * npml * n2 + (size_t)gtid2 * npml + ic;
    mem_dzz[pind] = mem_dzz[pind] * az[ic] + c1 * (az[ic] - 1.0f);
    vzz_loc += mem_dzz[pind];
  }

  if (gtid2 < npml) {
    pind = (size_t)gtid3 * npml * n1 + (size_t)gtid2 * n1 + gtid1;
    mem_dxx[pind] = mem_dxx[pind] * bx[gtid2] + c2 * (bx[gtid2] - 1.0f);
    vxx_loc += mem_dxx[pind];
  }
  if (gtid2 >= n2 - npml) {
    ic = gtid2 - n2 + npml;
    pind = (size_t)n3 * npml * n1 + (size_t)gtid3 * (npml * n1) + ic * n1 + gtid1;
    mem_dxx[pind] = mem_dxx[pind] * ax[ic] + c2 * (ax[ic] - 1.0f);
    vxx_loc += mem_dxx[pind];
  }

  if (gtid3 < npml) {
    pind = (size_t)gtid3 * n2 * n1 + (size_t)gtid2 * n1 + gtid1;
    mem_dyy[pind] = mem_dyy[pind] * by[gtid3] + c3 * (by[gtid3] - 1.0f);
    vyy_loc += mem_dyy[pind];
  }
  if (gtid3 >= n3 - npml) {
    ic = gtid3 - n3 + npml;
    pind = (size_t)npml * n2 * n1 + ic * n2 * n1 + (size_t)gtid2 * n1 + gtid1;
    mem_dyy[pind] = mem_dyy[pind] * ay[ic] + c3 * (ay[ic] - 1.0f);
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
