#include "optimization_cuda.h"

void cal_fwi_grad_3d(float *obj_xc, float *obj_l,
		     float ***vin, float *grad, float *pre_con,
		     float **wb, int itop, float near_cut, float far_cut, float sht_scl,
		     int *nrec_shot, int *sht_num,
		     int **src0_indx, int **rec0_indx,
		     float *sw000, float *sw001, float *sw010, float *sw011,
		     float *sw100, float *sw101, float *sw110, float *sw111,
		     float *rw000, float *rw001, float *rw010, float *rw011,
		     float *rw100, float *rw101, float *rw110, float *rw111,
		     float *bt, float *bb, float *bl, float *br,
		     float *ay, float *by, float *ax, float *bx, float *az, float *bz,
		     float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
		     int ns, int myns, int ns_s, size_t ntr, int nt, float dt, int ntj, int ntw,
		     int flag0,
		     int ny, int nx, int nz, int npml, float dy, float dx, float dz, float xpad,
		     float *j0k, int mrecu, float irr, float *src, 
		     char *obs_name, char *wb_name, char *tmut_name, char *bmut_name, int L_n,
		     int tmutflag,int bmutflag, int directflag, char *order, float vmax, int fmflag,
		     int ntids, int mytid, int root, MPI_Comm comm){

  int is, iss, snum, yl_pad, yr_pad, xl_pad, xr_pad, iy, ix, iz, ir, it;
  int sy, sx, min_all_x, max_all_x, min_all_y, max_all_y;
  size_t rymin, rymax, rxmin, rxmax;
  int ny_new, nx_new, nypad, nxpad, nzpad, yl, yr, xl, xr, indis;
  size_t nxyz, byte, bytet, icc, nr;
  float *mygrad, *mypre_con, myobj_xc, myobj_l;
  float **h_obs, **d_est, **d_1, **d_2, *h_tmut, *h_bmut, **h_wb_est, maxobs, maxest;
  float dis, decay;
  double t1, t2, t3, t4, t5, t6;
#ifdef CUDA3D_HOST_SETUP_TIMERS
  double timer_shot_start, timer_after_obs, timer_after_domain;
  double timer_before_fd, timer_after_fd, timer_after_write, timer_after_free;
  double timer_after_loop, timer_after_sync, timer_after_copy_reduce;
  double timer_obs_setup = 0.0;
  double timer_domain_setup = 0.0;
  double timer_wavefield_prep = 0.0;
  double timer_fd_call = 0.0;
  double timer_output_write = 0.0;
  double timer_shot_cleanup = 0.0;
  int timer_valid_shots = 0;
#endif
  char h_obs_file[800], tmut_file[800], bmut_file[800], tmp[800], wb_file[800];
  float a2, b2, c, sum, bscl=0.9;
  int isc, myisc;
  /////
  int iii, nzdpad, nzd, itmp, iwb, nxz;
  float ***vc_wb, ***vc_wb_pad, ***vc, ***vc_pad;
  int *vek; //not implemented yet
  iii=4;

  /////
  float *h_grad, *h_pre_con;
  float *d_grad, *d_pre_con;
  /////

  nzpad=nz+2*(npml+radius);
#ifdef CUDA3D_HOST_SETUP_TIMERS
  double host_cal_entry = MPI_Wtime();
#endif
  nxz=nx*nz;
  myisc=0; 
  isc=0;
  *obj_l=0.0;
  *obj_xc=0.0;
  myobj_l=0.0;
  myobj_xc=0.0;
  nxyz=ny*nx*nz;
  byte=sizeof(float)*nxyz;

  mygrad=alloc1float(nxyz);
  mypre_con=alloc1float(nxyz);
  memset(mygrad, 0., byte);
  memset(mypre_con, 0., byte);

  //---------------- allocate device memory-----------------
  cudaMalloc((void**)&d_grad, byte); 
  cudaMalloc((void**)&d_pre_con, byte); 
  cudaMemset(d_grad, 0., byte);
  cudaMemset(d_pre_con, 0., byte);

  //------------ padded domain size----------------
  xl_pad=(int)(xpad/dx);
  xr_pad=(int)(xpad/dx);
  yl_pad=(int)(xpad/dy);
  yr_pad=(int)(xpad/dy);

  if(mytid==root){
    t3=MPI_Wtime();
#ifdef CUDA3D_HOST_SETUP_TIMERS
    printf("HOST_SETUP_TIMER cal pre_gradient_init=%lf\n", t3 - host_cal_entry);
#endif
    printf("\nComputing GRADIENT, Random shot starting point=%d\n",sht_num[mytid]);
  }

  // -----------------start computing--------------------------
  for (is=0; is<myns; is+=1){
    //    snum=mytid*myns*nds+is*nds+iis;
    snum=sht_num[is*ntids+mytid]; // get the shot number from the list
    if(snum >=0 && snum < ns){
#ifdef CUDA3D_HOST_SETUP_TIMERS
      if(mytid==root)
	timer_shot_start = MPI_Wtime();
#endif
      printf("shot =%d \n", snum);
      myisc++;
      if(mytid==root)
	printf("Round %d of %d\n", is+1, myns);
      //      printf("ID=%d, shot #=%d\n    shot 0index iy=%d ix=%d iz=%d\n",mytid, snum, src0_indx[snum][0], src0_indx[snum][1], src0_indx[snum][2]);
      // get reciever starting point in nav and recv num for a shot
      icc=0;
      for(iss=0; iss<snum; iss++)
	icc+=nrec_shot[iss];
      nr=nrec_shot[snum];
      bytet=sizeof(float)*(nr*nt);

      // get file names
      sprintf(h_obs_file,"%s%d.dir",obs_name,snum);
      sprintf(tmut_file,"%s%d.dir",tmut_name,snum);
      sprintf(bmut_file,"%s%d.dir",bmut_name,snum);
      sprintf(wb_file,"%s%d.dir",wb_name,snum);

      // read obs, direct, mute files
      if(mytid==root)
	t5=MPI_Wtime();
      if(fmflag==1){
	h_obs=alloc2float(nr, nt);
	memset(&h_obs[0][0], 0., bytet);
      }
      else
	h_obs=read_dir(h_obs_file, nt, nr); // note nr is fast direction for 3D case

      if(mytid==root)
	t6=MPI_Wtime();
      
      if(tmutflag!=0){
	h_tmut=readdir1d(tmut_file, nr);
	for(ir=0; ir<nr; ir++)
	  h_tmut[ir]=h_tmut[ir]/dt;
      }
      else{
	h_tmut=alloc1float(nr);
	init_1d(h_tmut,nr);
      }
      if(bmutflag!=0){
	h_bmut=readdir1d(bmut_file, nr);
	for(ir=0; ir<nr; ir++){
	  h_bmut[ir]=h_bmut[ir]/dt;
	  h_bmut[ir]=MIN(h_bmut[ir],nt);
	}
      }
      else{
	h_bmut=alloc1float(nr);
	for (ir=0; ir<nr; ir++)
	  h_bmut[ir]=nt;
      }
      
      if(directflag==0){
	h_wb_est=alloc2float(nr, nt);
	memset(&h_wb_est[0][0], 0., bytet);
      }
      else{
	//	h_wb_est=read_dir(wb_file, nt, nr);
	h_wb_est=alloc2float(nr, nt);
	memset(&h_wb_est[0][0], 0., bytet);
      }
    
      d_est=alloc2float(nr, nt);  // note nr fast direction, nt slow direction
      memset(&d_est[0][0], 0., bytet);
#ifdef CUDA3D_HOST_SETUP_TIMERS
      if(mytid==root)
	timer_after_obs = MPI_Wtime();
#endif

      //-------- find modeling domain  probably need work for different acqusition system---------
      sy=src0_indx[snum][0];
      sx=src0_indx[snum][1];
      rymin=9999999;      rymax=0;
      rxmin=9999999;      rxmax=0;
      for(ir=0; ir<nr; ir++){
	rymin=MIN(rymin,rec0_indx[icc+ir][0]);
	rymax=MAX(rymax,rec0_indx[icc+ir][0]);
	rxmin=MIN(rxmin,rec0_indx[icc+ir][1]);
	rxmax=MAX(rxmax,rec0_indx[icc+ir][1]);
      }
      min_all_y=MIN(sy, rymin);  max_all_y=MAX(sy, rymax); //y min and max location index for given shot
      min_all_x=MIN(sx, rxmin);  max_all_x=MAX(sx, rxmax); //x min and max location index for given shot

      if(max_all_x>nx || max_all_y >ny){
	printf("nsy=%d nsx=%d, nrymin=%zu nrymax=%zu, nrxmin=%zu nrxmax=%zu\n", 
	       sy, sx, rymin, rymax, rxmin, rxmax);
	printf("ERROR subdomain error for #%d shot!, acqusition array outside largest domain!!\n", snum);
	exit(0);
      }
      if(min_all_x>=max_all_x || min_all_y>=max_all_y){
	printf("sy=%d sx=%d, rymin=%zu rymax=%zu rxmin=%zu rxmax=%zu \n", sy, sx, rymin, rymax, rxmin, rxmax);
	printf("ERROR subdomain error for #%d shot!\n", snum);
	exit(0);
      }
      yl=MAX(0, min_all_y-yl_pad);  // modeling domain left edge
      yr=MIN(ny-1, max_all_y+yr_pad); // right edge
      xl=MAX(0, min_all_x-xl_pad);  // modeling domain left edge
      xr=MIN(nx-1, max_all_x+xr_pad); // right edge
      ny_new=yr-yl+1;
      nx_new=xr-xl+1;       //modeling domain size
      nypad=ny_new+2*(npml+radius);  // padded modeling domain
      nxpad=nx_new+2*(npml+radius);  // padded modeling domain
#ifdef CUDA3D_HOST_SETUP_TIMERS
      if(mytid==root)
	timer_after_domain = MPI_Wtime();
#endif
      //      printf("ID=%d #shot =%d yl=%d yr=%d xl=%d xr=%d nynew=%d, nxnew=%d\n", mytid, snum, yl, yr, xl, xr, ny_new, nx_new);
      // end of domain truncation
      // end computing domain truncation

      // wavefield
      // -------------------- forward modeling or RTM---------------------
      vc=alloc3float(nz, nx_new, ny_new);
      vc_pad=alloc3float(nzpad, nxpad, nypad);

      for (iy=0; iy<ny_new; iy++)
	for (ix=0; ix<nx_new; ix++)
	  for(iz=0; iz<nz; iz++)
	    vc[iy][ix][iz]=vin[yl+iy][xl+ix][iz]*vin[yl+iy][xl+ix][iz];//*dt2; // need check
      vpad_3d(vc, vc_pad, ny_new, nx_new, nz, npml+radius); // need new check 
#ifdef CUDA3D_HOST_SETUP_TIMERS
      if(mytid==root)
	timer_before_fd = MPI_Wtime();
#endif
      //      writesu(vc[10], nx_new, nz, dx, dz,"v_y_slice.su");
      //      writesu(vc_pad[10], nxpad, nzpad, dx, dz, "test_y_v_pad.su");

      //      f=alloc4float(nz, nx_new, ny_new, nt);  //nx_new1 -> nx_new 07.11.2020 for this 3D implementation, shot gahter is extracted with fu in fd_3d_f not f (which is used for 2D case)
      //      memset(&f[0][0][0][0], 0., sizeof(float)*nz*nx_new*ny_new*nt);
      // forward modeling
      if(mytid==root)
	t1=MPI_Wtime();

	printf("id =%d start modeling\n", mytid);
	fd_3d_f(src, bscl, vc_pad, h_obs,
		ny_new, nx_new, nz, dy, dx, dz,  //ny_new
		nt, dt, ntw, 1, vek,
		src0_indx, rec0_indx,
		nr, icc, ns, snum, yl, xl, 
		sw000, sw001, sw010, sw011,
		sw100, sw101, sw110, sw111,
		rw000, rw001, rw010, rw011,
		rw100, rw101, rw110, rw111,
		bt, bb, bl, br, npml, vmax,
		ay, by, ax, bx, az, bz,
		ay_h, by_h, ax_h, bx_h, az_h, bz_h,
		mytid, order);
#ifdef CUDA3D_HOST_SETUP_TIMERS
      if(mytid==root)
	timer_after_fd = MPI_Wtime();
#endif
	sprintf(tmp,"%s_%d.dir","./d_obs/d_obs_salt_gpu_cpu_checked_ricker1_8hz_3d_ny_384_nx_384_nz95_nbell_1_bscl_0.9_moffy_9.5625_moffx_9.5625_h_obs_nt_1501_dt_2ms_shot", snum);
	writedir(h_obs, nt, nr, tmp);
#ifdef CUDA3D_HOST_SETUP_TIMERS
      if(mytid==root)
	timer_after_write = MPI_Wtime();
#endif
	// need computing gradient, to be implemented
      

           
      free1float(h_tmut);
      free1float(h_bmut);      
      free3float(vc);
      free3float(vc_pad);
      free2float(h_obs);
      free2float(d_est);
      //      free2float(d_1);
      //      free2float(d_2);
      free2float(h_wb_est);
#ifdef CUDA3D_HOST_SETUP_TIMERS
      if(mytid==root){
	timer_after_free = MPI_Wtime();
	timer_obs_setup += timer_after_obs - timer_shot_start;
	timer_domain_setup += timer_after_domain - timer_after_obs;
	timer_wavefield_prep += timer_before_fd - timer_after_domain;
	timer_fd_call += timer_after_fd - timer_before_fd;
	timer_output_write += timer_after_write - timer_after_fd;
	timer_shot_cleanup += timer_after_free - timer_after_write;
	timer_valid_shots++;
      }
#endif
    }    // end positive s_array
    if (mytid==root)
      t2=MPI_Wtime();
  }// end all shots
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    timer_after_loop = MPI_Wtime();
#endif
  cudaDeviceSynchronize();
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    timer_after_sync = MPI_Wtime();
#endif

  // ------------------copy device gradint to host-------------------
  h_grad=alloc1float(nxyz);
  h_pre_con=alloc1float(nxyz);
  memset(h_grad, 0., byte);
  //  memset(h_pre_con, 0., byte);
  cudaMemcpy(&h_grad[0], d_grad, byte, cudaMemcpyDeviceToHost); 
  //  cudaMemcpy(&h_pre_con[0], d_pre_con, byte, cudaMemcpyDeviceToHost);

  MPI_Reduce(&myobj_l, obj_l, 1, MPI_FLOAT, MPI_SUM, root, comm);
  MPI_Reduce(&myobj_xc, obj_xc, 1, MPI_FLOAT, MPI_SUM, root, comm);
  MPI_Reduce(&myisc, &isc,1,MPI_INT,MPI_SUM,root,comm);
  MPI_Reduce(&h_grad[0],&grad[0],nxyz,MPI_FLOAT,MPI_SUM,root,comm);
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    timer_after_copy_reduce = MPI_Wtime();
#endif


  if(mytid==root){
    *obj_xc=*obj_xc/isc;
    t4=MPI_Wtime();
#ifdef CUDA3D_HOST_SETUP_TIMERS
    printf("HOST_SETUP_TIMER cal_loop shots=%d obs_setup=%lf domain_setup=%lf wavefield_prep=%lf fd_call=%lf output_write=%lf cleanup=%lf post_loop_sync=%lf copy_reduce=%lf\n",
	   timer_valid_shots,
	   timer_obs_setup,
	   timer_domain_setup,
	   timer_wavefield_prep,
	   timer_fd_call,
	   timer_output_write,
	   timer_shot_cleanup,
	   timer_after_sync - timer_after_loop,
	   timer_after_copy_reduce - timer_after_sync);
#endif
    printf("Gradient TIME all= %lfs, WP computing time = %lfs, read time =%lfs \n",t4-t3, (t2-t1)*myns, (t6-t5)*myns);
  }
  MPI_Bcast(obj_xc, 1, MPI_FLOAT, root, comm);

  free1float(mygrad);
  free1float(mypre_con);
  MPI_Barrier(comm);
  free1float(h_grad);
  free1float(h_pre_con);
  cudaFree(d_grad);
  cudaFree(d_pre_con);
  //  exit(0);
}
