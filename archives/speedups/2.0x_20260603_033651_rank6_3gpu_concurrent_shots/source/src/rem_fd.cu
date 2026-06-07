#include "rem_fd.h"

void check_gpu_error_2(const char *msg){  
  cudaError_t err = cudaGetLastError ();
  if (cudaSuccess !=err){ 
    printf("Cuda error: %s: %s", msg, cudaGetErrorString(err));
    exit(0); 
  }
} 

#ifdef CUDA3D_DEBUG_CHECKS
#define check_gpu_error_loop(msg) check_gpu_error_2(msg)
#else
#define check_gpu_error_loop(msg) ((void)0)
#endif

void fd_3d_f(float *src, float bscl, float ***cw2, float **h_est,
	     int ny, int nx, int nz, float dy, float dx, float dz,
	     int nt, float dt, int ntw, int ntj, int *vek,
	     int **src0_indx, int **rec0_indx,
	     size_t nr, size_t icc, int ns, int snum, int yl, int xl,
	     float *sw000, float *sw001, float *sw010, float *sw011,
	     float *sw100, float *sw101, float *sw110, float *sw111,
	     float *rw000, float *rw001, float *rw010, float *rw011,
	     float *rw100, float *rw101, float *rw110, float *rw111,
	     float *bt, float *bb, float *bl, float *br, int nbd, float vmax,
	     float *ay, float *by, float *ax, float *bx, float *az, float *bz,
	     float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
	     int mytid, char *order){

  int iy, ix, iz, it, itc, nby, nbx, nbz, nbell=1;
  int nypad, nxpad, nzpad;
  float ss, temp, tdy, tdx, tdz, dt2;
  float *h_bell;
  size_t nxyz, byte, ir, nxyzpad;

  // device variables
  int indxx, indxy, indxz;
  int *d_src0_indx, *d_rec0_indx;
  float *d_bell, *d_src, *d_cw2, *d_est;
  float *d_sw000, *d_sw001, *d_sw010, *d_sw011, *d_sw100, *d_sw101, *d_sw110, *d_sw111;
  float *d_rw000, *d_rw001, *d_rw010, *d_rw011, *d_rw100, *d_rw101, *d_rw110, *d_rw111;
  float *d_memory_dy, *d_memory_dx, *d_memory_dz;
  float *d_memory_dyy, *d_memory_dxx, *d_memory_dzz;

  //wavefields
  float *d_p0, *d_p1, *ptr, *d_vx, *d_vy, *d_vz;
  // pml
  float *d_ax, *d_bx, *d_ay, *d_by, *d_az, *d_bz, *d_ax_h, *d_bx_h, *d_ay_h, *d_by_h, *d_az_h, *d_bz_h;

  // src y, x, and z location index

  indxy=src0_indx[snum][0];
  indxx=src0_indx[snum][1];
  indxz=src0_indx[snum][2];
  if(mytid==0)
    printf(" indxy=%d, indxx=%d, indxz=%d\n", indxy, indxx, indxz);

  nby=ny+2*(nbd);
  nbx=nx+2*(nbd);
  nbz=nz+2*(nbd);
  // need radius!!
  nypad=ny+2*(nbd+radius);
  nxpad=nx+2*(nbd+radius);
  nzpad=nz+2*(nbd+radius);

  nxyz=nbx*nbz*nby;
  byte=sizeof(float)*nxyz;
  nxyzpad=nypad*nxpad*nzpad;
  //  if(mytid==3)
  //    printf("id=%d nzpad, nxpad, nzpad= %d, %d, %d, shot#=%d, ns=%d, nr=%d, nsize=%zu\n", 
  //	   mytid, nypad, nxpad, nzpad, snum, ns, nr, nxyzpad);
  //  printf("lap nbx=%d nby=%d nbz=%d, ny=%d nx=%d nz=%d nbd=%d\n",nbx, nby, nbz, ny, nx, nz, nbd);

  dt2=dt*dt;
  tdy=1./dy;
  tdx=1./dx;
  tdz=1./dz;

  //-------------------- setup bell function ------------------------
  h_bell=alloc1float((2*nbell+1)*(2*nbell+1)*(2*nbell+1));
  ss=0.5*nbell;
  for(iy=-nbell; iy<=nbell; iy++)
    for(ix=-nbell; ix<=nbell; ix++)
      for(iz=-nbell; iz<=nbell; iz++)
	h_bell[(nbell+iy)*(2*nbell+1)*(2*nbell+1)+(nbell+ix)*(2*nbell+1)+nbell+iz]=exp(-bscl*bscl*(iy*iy+iz*iz+ix*ix)/ss);

  cudaMalloc((void**)&d_bell, (2*nbell+1)*(2*nbell+1)*(2*nbell+1)*sizeof(float));
  cudaMemcpy(d_bell, h_bell, (2*nbell+1)*(2*nbell+1)*(2*nbell+1)*sizeof(float), cudaMemcpyHostToDevice);

  // -------------- copy wavelet and velocity to device------------------------
  cudaMalloc((void**)&d_src, nt*sizeof(float));
  cudaMemcpy(d_src, &src[0], nt*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_cw2, nxyzpad*sizeof(float));
  cudaMemcpy(d_cw2, &cw2[0][0][0], nxyzpad*sizeof(float), cudaMemcpyHostToDevice);
  fflush(stdout);
  // ----------------initialize src and rec index and interpolation parameters----------------------
  cudaMalloc((void**)&d_sw000, ns*sizeof(float));   cudaMalloc((void**)&d_sw001, ns*sizeof(float));
  cudaMalloc((void**)&d_sw010, ns*sizeof(float));   cudaMalloc((void**)&d_sw011, ns*sizeof(float));
  cudaMalloc((void**)&d_sw100, ns*sizeof(float));   cudaMalloc((void**)&d_sw101, ns*sizeof(float));
  cudaMalloc((void**)&d_sw110, ns*sizeof(float));   cudaMalloc((void**)&d_sw111, ns*sizeof(float));
  cudaMemcpy(d_sw000, &sw000[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw001, &sw001[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw010, &sw010[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw011, &sw011[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw100, &sw100[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw101, &sw101[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw110, &sw110[0], ns*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sw111, &sw111[0], ns*sizeof(float), cudaMemcpyHostToDevice);

  // receivers
  cudaMalloc((void**)&d_rw000, nr*sizeof(float));   cudaMalloc((void**)&d_rw001, nr*sizeof(float));
  cudaMalloc((void**)&d_rw010, nr*sizeof(float));   cudaMalloc((void**)&d_rw011, nr*sizeof(float));
  cudaMalloc((void**)&d_rw100, nr*sizeof(float));   cudaMalloc((void**)&d_rw101, nr*sizeof(float));
  cudaMalloc((void**)&d_rw110, nr*sizeof(float));   cudaMalloc((void**)&d_rw111, nr*sizeof(float));
  cudaMemcpy(d_rw000, &rw000[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw001, &rw001[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw010, &rw010[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw011, &rw011[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw100, &rw100[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw101, &rw101[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw110, &rw110[icc], nr*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rw111, &rw111[icc], nr*sizeof(float), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_src0_indx, 3*ns*sizeof(int));
  cudaMalloc((void**)&d_rec0_indx, 3*nr*sizeof(int));
  cudaMemcpy(d_src0_indx, &src0_indx[0], 3*ns*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_rec0_indx, &rec0_indx[icc][0], 3*nr*sizeof(int), cudaMemcpyHostToDevice);

  // -------------------initialize wavefields and data----------------------
  cudaMalloc((void**)&d_est, nt*nr*sizeof(float));
  cudaMemset(d_est, 0., nt*nr*sizeof(float));  // note ir is the fast direction in 3D
  check_gpu_error_2("Error in Memset");
 
  cudaMalloc((void**)&d_p0, nxyzpad*sizeof(float));
  cudaMalloc((void**)&d_p1, nxyzpad*sizeof(float));
  cudaMalloc((void**)&d_vy, nxyzpad*sizeof(float));
  cudaMalloc((void**)&d_vx, nxyzpad*sizeof(float));
  cudaMalloc((void**)&d_vz, nxyzpad*sizeof(float));
  cudaMemset(d_p0, 0, nxyzpad*sizeof(float));
  cudaMemset(d_p1, 0, nxyzpad*sizeof(float));
  cudaMemset(d_vy, 0, nxyzpad*sizeof(float));
  cudaMemset(d_vx, 0, nxyzpad*sizeof(float));
  cudaMemset(d_vz, 0, nxyzpad*sizeof(float));
  check_gpu_error_2("Error in Memset");

  // ----------------------------initialize pml arrays-------------------------------
  cudaMalloc((void**)&d_ay, nbd*sizeof(float));
  cudaMalloc((void**)&d_by, nbd*sizeof(float));
  cudaMalloc((void**)&d_ax, nbd*sizeof(float));
  cudaMalloc((void**)&d_bx, nbd*sizeof(float));
  cudaMalloc((void**)&d_az, nbd*sizeof(float));
  cudaMalloc((void**)&d_bz, nbd*sizeof(float));
  cudaMalloc((void**)&d_ay_h, nbd*sizeof(float));
  cudaMalloc((void**)&d_by_h, nbd*sizeof(float));
  cudaMalloc((void**)&d_ax_h, nbd*sizeof(float));
  cudaMalloc((void**)&d_bx_h, nbd*sizeof(float));
  cudaMalloc((void**)&d_az_h, nbd*sizeof(float));
  cudaMalloc((void**)&d_bz_h, nbd*sizeof(float));

  cudaMemcpy(d_ay, ay, nbd*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_by, by, nbd*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ax, ax, nbd*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bx, bx, nbd*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_az, az, nbd*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bz, bz, nbd*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ay_h, ay_h, nbd*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_by_h, by_h, nbd*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ax_h, ax_h, nbd*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bx_h, bx_h, nbd*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_az_h, az_h, nbd*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bz_h, bz_h, nbd*sizeof(float), cudaMemcpyHostToDevice);

  // ----------------- initialize memeory varibles pml note the order of nbd nbx when used-----------
  cudaMalloc((void**)&d_memory_dy, 2*nbd*nbx*nbz*sizeof(float));
  cudaMalloc((void**)&d_memory_dx, nby*2*nbd*nbz*sizeof(float));
  cudaMalloc((void**)&d_memory_dz, nby*nbx*2*nbd*sizeof(float));
  cudaMalloc((void**)&d_memory_dyy, 2*nbd*nbx*nbz*sizeof(float));
  cudaMalloc((void**)&d_memory_dxx, nby*2*nbd*nbz*sizeof(float));
  cudaMalloc((void**)&d_memory_dzz, nby*nbx*2*nbd*sizeof(float));

  cudaMemset(d_memory_dy, 0., 2*nbd*nbx*nbz*sizeof(float));
  cudaMemset(d_memory_dx, 0., nby*2*nbd*nbz*sizeof(float));
  cudaMemset(d_memory_dz, 0., nby*nbx*2*nbd*sizeof(float));
  cudaMemset(d_memory_dyy, 0., 2*nbd*nbx*nbz*sizeof(float));
  cudaMemset(d_memory_dxx, 0., nby*2*nbd*nbz*sizeof(float));
  cudaMemset(d_memory_dzz, 0., nby*nbx*2*nbd*sizeof(float));

  check_gpu_error_2("Error in Memset");

  dim3 dimg_v, dimb_v, dimg_p, dimb_p, dims, dimbs, dimr, dimbr;// dims(1,1), dimbs(2*nbell+1, 2*nbell+1);
  dims.x=1;
  dims.y=1;
  dims.z=1;
  dimbs.x=2*nbell+1;
  dimbs.y=2*nbell+1;
  dimbs.z=2*nbell+1;

  //  dimbs.x=1;
  //  dimbs.y=1;
  int BS=1024;
  dimr.x=((nr+BS-1)/BS); // 1 
  dimr.y=1;
  dimbr.x=BS; //nr // need check!!!!!!
  dimbr.y=1;


  //  dimg.x=(int)((nbz+BlockSize1-1)/BlockSize1);
  //  dimg.y=(int)((nbx+BlockSize2-1)/BlockSize2);
  //  dimb.x=BlockSize1;
  //  dimb.y=BlockSize2;

  dimg_v.x=(int)((nbz+VBlockSize1-1)/VBlockSize1);
  dimg_v.y=(int)((nbx+VBlockSize2-1)/VBlockSize2);
  dimg_v.z=(int)((nby+VBlockSize3-1)/VBlockSize3);
  dimb_v.x=VBlockSize1;
  dimb_v.y=VBlockSize2;
  dimb_v.z=VBlockSize3;

  dimg_p.x=(int)((nbz+PBlockSize1-1)/PBlockSize1);
  dimg_p.y=(int)((nbx+PBlockSize2-1)/PBlockSize2);
  dimg_p.z=(int)((nby+PBlockSize3-1)/PBlockSize3);
  dimb_p.x=PBlockSize1;
  dimb_p.y=PBlockSize2;
  dimb_p.z=PBlockSize3;

  cudaFuncSetCacheConfig(cuda_fd3d_v_pml_ns, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(cuda_fd3d_p_core_ns, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(cuda_fd3d_p_pml_ns, cudaFuncCachePreferL1);

  //  dim3 dimgu, dimbu;
  //  dimgu.x=(nz+BlockSize1-1)/BlockSize1;
  //  dimgu.y=(nx+BlockSize2-1)/BlockSize2;
  //  dimbu.x=BlockSize1;
  //  dimbu.y=BlockSize2;

  //////
  cudaEvent_t t1, t2, t3, t4;
  float mill;
  cudaEventCreate(&t1);
  cudaEventCreate(&t2);
  cudaEventCreate(&t3);
  cudaEventCreate(&t4);

  //////////
  //  float ***out, **out2, **out3;
  //  out=alloc3float(nz, nx, ny);
  //  out2=alloc2float(nz, nx);
  //  out3=alloc2float(nx, ny);
  //  bell=alloc3float(2*nbell+1, 2*nbell+1, 2*nbell+1);
  //  ss=0.5*nbell;

  itc=0;
  fflush(stdout);
  cudaEventRecord(t1,0);

  for(it=0; it<nt; it++){		
    //    fflush(stdout);
    if(it%500==0 && mytid==0)
      printf("FP it=%d\n", it);
    // this is 2nd order time
    cuda_fd3d_v_pml_ns<<<dimg_v, dimb_v>>>(d_p1, d_vy, d_vx, d_vz,
				    tdy, tdx, tdz,
				    nby, nbx, nbz, nbd, dt,
				    d_ay_h, d_by_h, d_ax_h, d_bx_h, d_az_h, d_bz_h,
				    d_memory_dy, d_memory_dx, d_memory_dz);
    check_gpu_error_loop("compute V");
    cuda_fd3d_p_core_ns<<<dimg_p, dimb_p >>>(d_p0, d_p1, d_cw2,
				     tdy, tdx, tdz,
				     nby, nbx, nbz, nbd, dt2);
    check_gpu_error_loop("compute P core");
    cuda_fd3d_p_pml_ns<<<dimg_p, dimb_p >>>(d_p0, d_p1, d_vy, d_vx, d_vz,
				     d_cw2, tdy, tdx, tdz,
				     nby, nbx, nbz, nbd, dt2,
				     d_ay, d_by, d_ax, d_bx, d_az, d_bz,
				     d_memory_dyy, d_memory_dxx, d_memory_dzz);
    check_gpu_error_loop("compute P pml");
    if (nr <= (size_t)BS) {
      lint3d_inject_bell_extract_gpu_zz<<<1, dimbr>>>(d_p0, nbd, yl, xl, it, nt, snum,
					    d_src, d_bell, nbell,
					    indxy, indxx, indxz, nypad, nxpad, nzpad,
					    d_sw000, d_sw001, d_sw010, d_sw011,
					    d_sw100, d_sw101, d_sw110, d_sw111,
					    d_est, d_rec0_indx, nr,
					    d_rw000, d_rw001, d_rw010, d_rw011,
					    d_rw100, d_rw101, d_rw110, d_rw111);
      check_gpu_error_loop("inject src and extract");
    } else {
      // inject bell src
      lint3d_inject_bell_gpu<<<dims, dimbs>>>(d_p0, nbd, yl, xl, it, snum,
					      d_src, d_bell, nbell,
					      indxy, indxx, indxz, nypad, nxpad, nzpad,
					      d_sw000, d_sw001, d_sw010, d_sw011,
					      d_sw100, d_sw101, d_sw110, d_sw111);
      check_gpu_error_loop("inject src");
      // extract 
      lint3d_extract_gpu_zz<<<dimr, dimbr>>>(d_p0, nbd, yl, xl, it, nt,
					     d_est, d_rec0_indx, nr, nypad, nxpad, nzpad, // ir is fast direction in 3D
					     d_rw000, d_rw001, d_rw010, d_rw011,
					     d_rw100, d_rw101, d_rw110, d_rw111);
    }

    ptr=d_p0; d_p0=d_p1; d_p1=ptr;
   
    //might need this or FD CPML, to improve numerical stability for large dt and dx
    //    bc_3d(fu, nby, nbx, nbz, nbd, bt, bb, bl, br);
    //    bc_3d(pfu, nby, nbx, nbz, nbd, bt, bb, bl, br);
  }// end time stepping

  check_gpu_error_2("time stepping");
  cudaMemcpy(&h_est[0][0], d_est, nt*nr*sizeof(float), cudaMemcpyDeviceToHost);
  cudaEventRecord(t2, 0);
  cudaEventElapsedTime(&mill, t1, t2);
  if(mytid==0)
    printf("mod time %fs\n", (float)(mill)/(1000.));

  free1float(h_bell);  //free3float(out); free2float(out2); free2float(out3);
  cudaFree(d_src0_indx); cudaFree(d_rec0_indx);
  cudaFree(d_bell); cudaFree(d_src); cudaFree(d_cw2); cudaFree(d_est);
  cudaFree(d_sw000); cudaFree(d_sw001); cudaFree(d_sw010); cudaFree(d_sw011);
  cudaFree(d_sw100); cudaFree(d_sw101); cudaFree(d_sw110); cudaFree(d_sw111);
  cudaFree(d_rw000); cudaFree(d_rw001); cudaFree(d_rw010); cudaFree(d_rw011);
  cudaFree(d_rw100); cudaFree(d_rw101); cudaFree(d_rw110); cudaFree(d_rw111);
  cudaFree(d_memory_dy); cudaFree(d_memory_dx); cudaFree(d_memory_dz);
  cudaFree(d_memory_dyy); cudaFree(d_memory_dxx); cudaFree(d_memory_dzz);
  cudaFree(d_p0); cudaFree(d_p1);
  cudaFree(d_vy); cudaFree(d_vx); cudaFree(d_vz); 
  cudaFree(d_ay); cudaFree(d_by); cudaFree(d_ax); cudaFree(d_bx); cudaFree(d_az); cudaFree(d_bz);
  cudaFree(d_ay_h); cudaFree(d_by_h); cudaFree(d_ax_h); cudaFree(d_bx_h); cudaFree(d_az_h); cudaFree(d_bz_h);
  check_gpu_error_2("Error in FREE");
}
