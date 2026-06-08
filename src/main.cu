#include "cu_common.h"
#define root 0
//#include "fdlbfgs.h"

//#include <sys/stat.h>
//#include <sys/types.h>
//#include <netinet/in.h>
#ifdef CUDA3D_HOST_SETUP_TIMERS
#include <sys/time.h>

static double cuda3d_wall_seconds(void){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}
#endif

void check_gpu_error(const char *msg){
  cudaError_t err = cudaGetLastError ();
  if (cudaSuccess !=err){
    printf("Cuda error: %s: %s", msg, cudaGetErrorString(err));
    exit(0);
  }
}

int main(int argc, char **argv){
  int ntids,mytid;
  int ID,np;
#ifdef CUDA3D_HOST_SETUP_TIMERS
  double process_timer_start = cuda3d_wall_seconds();
  double process_timer_after_mpi_init = process_timer_start;
  double process_timer_before_finalize = process_timer_start;
  double process_timer_after_finalize = process_timer_start;
#endif
  MPI_Init(&argc,&argv);
#ifdef CUDA3D_HOST_SETUP_TIMERS
  process_timer_after_mpi_init = cuda3d_wall_seconds();
#endif
  MPI_Comm comm;
  comm=MPI_COMM_WORLD;
  MPI_Comm_size(comm,&ntids);
  MPI_Comm_rank(comm,&mytid);
#ifdef CUDA3D_HOST_SETUP_TIMERS
  double host_timer_after_mpi = MPI_Wtime();
  double host_timer_after_input_scan = host_timer_after_mpi;
  double host_timer_after_gpu_setup = host_timer_after_mpi;
  double host_timer_after_input_bcast = host_timer_after_mpi;
  double host_timer_after_coeff_init = host_timer_after_mpi;
  double host_timer_after_static_alloc = host_timer_after_mpi;
  double host_timer_after_root_model_read = host_timer_after_mpi;
  double host_timer_after_model_bcast = host_timer_after_mpi;
  double host_timer_after_acqui_read = host_timer_after_mpi;
  double host_timer_after_acqui_bcast = host_timer_after_mpi;
  double host_timer_after_lint = host_timer_after_mpi;
  double host_timer_before_gradient = host_timer_after_mpi;
  double host_timer_after_gradient = host_timer_after_mpi;
  double host_timer_after_main_free = host_timer_after_mpi;
#endif
 
  // REM
  float r, irr, dtr, *j0k, vmax, *bt, *bb, *bl, *br;
  float *ay, *ax, *az, *by, *bx, *bz, *ay_h, *ax_h, *az_h, *by_h, *bx_h, *bz_h;
  int mrecu, ik;

  // shot
  int *nrec_shot, nrmax, ns, ns_s, ns_pad, myns, is;
  size_t ntr_shot;
  float *src, sht_scl;
  float **src_cor, **rec_cor;
  int **src0_indx, **rec0_indx;
  float *sw000, *sw001, *sw010, *sw011, *sw100, *sw101, *sw110, *sw111;
  float *rw000, *rw001, *rw010, *rw011, *rw100, *rw101, *rw110, *rw111;

  // time
  int nt;
  float dt, ddtt, decay, xpad;

  // TD finite difference
  char order[8];
  float near_cut, far_cut, decay_off;

  // MPI
  int extra;

  // fwi
  float *grad_cur, *pre_cur, ***vin;
  float obj_xc_cur, obj_l_cur;

  // model
  int nz, ny, nx, npml, ix, iz, iy;
  float dz, dy, dx;
  size_t nxyz, byte, nxy, nxz, nyz;
  float **wb;

  // other
  int itop;
  size_t ic, icc;

  // files
  char vinfile[800], outfile[800], srcfile[400], wbfile[800]; 
  char shotfile[800], navfile[800], directfile[800];
  char tmutfile[400], bmutfile[400];
  char tmpfile[800];

  // flags
  int vflag, sflag, rflag, directflag;
  int tmutflag, bmutflag, fmflag;

  // shot number
  int *sht_num;

  // Norm
  int L_n;

  // cuda
  cudaEvent_t tt1,tt2;
  float mills;
  int gpus_p_node, aval;
		         
  npml=20;
  mrecu=1;

  //// nav 
  int navflag=0;
  if(navflag!=0){  
  float ox, oy, lx, ly, s0x, s0y, sdx, sdy, rdx, rdy, r0x, r0y, rlx, rly, offx, offy, offx_set, offy_set;
  int snx, sny, rnx, rny, nr0x, nr0y, nrlx, nrly;
  int isx, isy, irx, iry;
  size_t ntr, nc;
  float **geo, ssx, ssy, rrx, rry, ssz, rrz;

  ny=256;
  nx=256;
  dy=0.0375; 
  dx=0.0375;
  ny=384;
  nx=384;
  dy=0.025; 
  dx=0.025;
  //  sdx=1.;  // shot_x spacing
  //  sdy=1.;  // shot_y spacing
  s0x=0.025; // shot_x initial location
  s0y=0.025; // shot_y inital location
  s0x=0.5; // shot_x initial location
  s0y=0.5; // shot_y inital location

  s0x=0.05;
  s0y=0.05;

  rdx=0.0375;  // rec_x spacing
  rdy=0.0375; // rec_y spacing
  rdx=0.025;
  rdy=0.025;

  oy=0.;  // y inital location
  ox=0.; // x inital location
  ly=(ny-1)*dy;  // y direction length
  lx=(nx-1)*dx;  // x direction length

  sny=20; //10
  snx=20; //10
  //  sny=30;
  //  snx=30;

  sdy=((ny-1)*dy-2*s0y)/(sny-1);
  sdx=((nx-1)*dx-2*s0x)/(snx-1);

  //  sny=(int)(((ny-1)*dy-2*s0y)/sdy)+1;  // num of shot along y direction
  //  snx=(int)(((nx-1)*dx-2*s0x)/sdx)+1;  // num of shot along x direction
  //  sny=(int)((ny-1)*dy)/sdy;  // num of shot along y direction
  //  snx=(int)((nx-1)*dx)/sdx;  // num of shot along x direction

  //  rny=(int)((ny-1)*dy/rdy);
  //  rnx=(int)((nx-1)*dx/rdx);

  //  ntr=snx*sny*rny*rnx;
  nc=6; // number of cor/trace 
  if(mytid==root)
    printf("sny=%d snx=%d rny=%d rnx=%d\n",sny, snx, rny, rnx);
  //////
  
  ////
  // find ntr for the setup
  offx_set=10.; // largest offset along x direction
  offy_set=2.5; // largest offset along y direction
  offy_set=1.5; // largest offset along y direction
  offy_set=10.; // largest offset along y direction
  ic=0;
  for (isy=0; isy<sny; isy++){
    ssy=s0y+isy*sdy; // shot_y coordinate
    if(mytid==root)
      printf("isy=%d ssy=%f\n", isy, ssy);
    for (isx=0; isx<snx; isx++){
      ssx=s0x+isx*sdx; // shot_x coordinate

      offx=MIN(offx_set, (nx-1)*dx);
      offy=MIN(offy_set, (ny-1)*dy);
      r0x=MAX(ssx-offx, ox);  // inital rec_x coordinate
      r0y=MAX(ssy-offy, oy);  // inital rec_y coordiante
      rlx=MIN(ssx+offx, lx);  // last rec_x coordinate
      rly=MIN(ssy+offy, ly);  // last rec_y coordinate

      nr0x=(int)((r0x-ssx)/rdx-1.e-5);
      nr0y=(int)((r0y-ssy)/rdy-1.e-5);
      nrlx=(int)((rlx-ssx)/rdx+1.e-5);
      nrly=(int)((rly-ssy)/rdy+1.e-5); // not sure why it has ssx, changed to 0 10.11.2025

      //      nr0x=(int)((r0x)/rdx-1.e-5);
      //      nr0y=(int)((r0y)/rdy-1.e-5);
      //      nrlx=(int)((rlx)/rdx+1.e-5);
      //      nrly=(int)((rly)/rdy+1.e-5);
      rnx=nrlx-nr0x+1;
      rny=nrly-nr0y+1;

      for (iry=0; iry<rny; iry++){
	for (irx=0; irx<rnx; irx++){
	  ic++;
	}
      }
    }
  }
  //  ntr=ic+1;
  ntr=ic;
  printf("ic=%zu   ntr=%zu\n", ic, ntr);

  ic=0;
  geo=alloc2float(nc, ntr);
  for (isy=0; isy<sny; isy++){
    ssy=s0y+isy*sdy;
    if(mytid==root)
      printf("isy=%d ssy=%f\n", isy, ssy);
    for (isx=0; isx<snx; isx++){
      ssx=s0x+isx*sdx;

      offx=MIN(offx_set, (nx-1)*dx);
      offy=MIN(offy_set, (ny-1)*dy);
      r0x=MAX(ssx-offx, ox);
      r0y=MAX(ssy-offy, oy);
      rlx=MIN(ssx+offx, lx);
      rly=MIN(ssy+offy, ly);

      nr0x=(int)((r0x-ssx)/rdx-1.e-5);
      nr0y=(int)((r0y-ssy)/rdy-1.e-5);
      nrlx=(int)((rlx-ssx)/rdx+1.e-5);
      nrly=(int)((rly-ssy)/rdy+1.e-5);

      //      nr0x=(int)((r0x)/rdx-1.e-5);
      //      nr0y=(int)((r0y)/rdy-1.e-5);
      //      nrlx=(int)((rlx)/rdx+1.e-5);
      //      nrly=(int)((rly)/rdy+1.e-5);
      rnx=nrlx-nr0x+1;
      rny=nrly-nr0y+1;

      if(isy==0 && isx==0){
	printf("r0x=%f r0y=%f rlx=%f rly=%f\n",r0x, r0y, rlx, rly);
	printf("nr0x=%d nr0y=%d nrlx=%d nrly=%d, number of nr in x=%d, number of nr in y=%d\n",nr0x, nr0y, nrlx, nrly, rnx, rny);
      }
      // assign src and rec coordinate values
      for (iry=0; iry<rny; iry++){
	rry=r0y+iry*rdy;
	//	rry=ssy+(iry+nr0y)*rdy;
	for (irx=0; irx<rnx; irx++){
	  rrx=r0x+irx*rdx;
		  //	  rrx=ssx+(irx+nr0x)*rdx;

	  //	  ic=isy*snx*rny*rnx+isx*rny*rnx+iry*rnx+irx;
	  	  geo[ic][0]=ssx;
	 	  geo[ic][1]=ssy;
	  	  geo[ic][2]=0.025; // src z cor // 0.0375
	  	  geo[ic][3]=rrx;
	  	  geo[ic][4]=rry;
	  	  geo[ic][5]=0.025; // rec z cor
	  ic++;
	}
      }
    }
  }
  sprintf(tmpfile,"%s_%f_%f_%d_%d_%f_%f_%zu.nav","overthrust_3d_s0y_s0x_nsy_nsx_moffy_moffx_ntr",s0y, s0x, sny, snx, offy, offx, ntr);
  writedir(geo, ntr, 6, tmpfile);
  ////
  FILE *ff;
  ff=fopen("geo_nav_2.txt","w");
  
  for(ic=0;ic<ntr;ic++){
    //    itemp=nrec_shot[ic];
    //    for(ii=0;ii<itemp;ii++){
      fprintf(ff,"shot %zu src& rec loc sy=%f sx=%f ry=%f rx=%f z=%f\n",
              ic+1,geo[ic][1], geo[ic][0], geo[ic][4], geo[ic][3], geo[ic][5]);
      //      icc++;
      //    }
  }
  fclose(ff);

  printf("NAV generated, txt check file done, exit!\n");
  free2float(geo);
  exit(0);

  }
  //  if()
  //  writedir(geo, 6, ntr, )
  
/*
  for (isy=0; isy<sny; isy++){
    ssy=s0y+isy*sdy;
    if(mytid==root)
      printf("isy=%d ssy=%f\n", isy, ssy);
    for (isx=0; isx<snx; isx++){
      ssx=s0x+isx*sdx;
      for (iry=0; iry<rny; iry++){
	rry=r0y+iry*rdy;
	for (irx=0; irx<rnx; irx++){
	  rrx=r0x+irx*rdx;

	  ic=isy*snx*rny*rnx+isx*rny*rnx+iry*rnx+irx;

	  geo[0][ic]=ssx;
	  geo[1][ic]=ssy;
	  geo[2][ic]=0.025;
	  geo[3][ic]=rrx;
	  geo[4][ic]=rry;
	  geo[5][ic]=0.025;
	  //	  ic++;
	}
      }
    }
  }
  printf("ic=%d   ntr=%d\n", ic, ntr);
*/
  ////
  fmflag=1;
  if(mytid==root){
    scanf("%s",shotfile);
    while(getchar()!='\n');
    printf("Shot input file is %s\n\n",shotfile);
    scanf("%f",&sht_scl);
    while(getchar()!='\n');

    scanf("%s",directfile);
    while(getchar()!='\n');
    printf("Shot input file is %s\n\n",directfile);
    scanf("%d",&directflag);
    while(getchar()!='\n');

    scanf("%s",navfile);
    while(getchar()!='\n');
    printf("Nav input file is %s\n\n",navfile);

    scanf("%s",srcfile);
    while(getchar()!='\n');
    printf("Source file is %s\n\n",srcfile);
    scanf("%s",wbfile);
    while(getchar()!='\n');
    printf("wb file is %s\n\n",wbfile);

    scanf("%d",&tmutflag);
    while(getchar()!='\n');
    scanf("%s",tmutfile);
    while(getchar()!='\n');
    printf("wb file is %s\n\n",tmutfile);
    scanf("%d",&bmutflag);
    while(getchar()!='\n');
    scanf("%s",bmutfile);
    while(getchar()!='\n');
    printf("wb file is %s\n\n",bmutfile);

    scanf("%s",outfile);
    while(getchar()!='\n');

    scanf("%zu",&ntr_shot);
    while(getchar()!='\n');

    scanf("%f",&near_cut);
    while(getchar()!='\n');
    scanf("%f",&far_cut);
    while(getchar()!='\n');
    scanf("%f",&decay_off);
    while(getchar()!='\n');

    // misfit norm
    scanf("%d", &L_n);
    while(getchar()!='\n');

    // READ input time
    scanf("%f",&dt);
    while (getchar()!='\n');
    scanf("%d",&nt);
    while (getchar()!='\n');

    // READ V MODEL
    scanf("%s",vinfile);
    while(getchar()!='\n');
    scanf("%d",&vflag);
    while(getchar()!='\n');
    scanf("%f",&vmax);
    while(getchar()!='\n');

    // READ modeling dimension
    scanf("%d",&ny);
    while (getchar()!='\n');
    scanf("%d",&nx);
    while (getchar()!='\n');
    scanf("%d",&nz);
    while (getchar()!='\n');

    scanf("%f",&dy);
    while (getchar()!='\n');
    scanf("%f",&dx);
    while (getchar()!='\n');
    scanf("%f",&dz);
    while (getchar()!='\n');

    scanf("%d",&npml);
    while (getchar()!='\n');
    scanf("%f",&decay);
    while (getchar()!='\n');
    scanf("%f",&xpad);
    while (getchar()!='\n');

    // READ TOP
    scanf("%d",&itop);
    while (getchar()!='\n');

    scanf("%s",order);
    while(getchar()!='\n');


    scanf("%d",&gpus_p_node);
    while (getchar()!='\n');
    if(ntids%gpus_p_node!=0){
      printf("GPUs per node needs to be a dividor of total process!!!\n");
      printf("ntids=%d, gpus_p_node=%d\n", ntids, gpus_p_node);
      printf("\nexit!!!\n");
      exit(0);
    }

  }
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    host_timer_after_input_scan = MPI_Wtime();
#endif
  ////---------- GPU setup and check------------------
  MPI_Bcast(&gpus_p_node, 1, MPI_INT, root, comm);
  //  printf("id=%d  gpus=%d\n", mytid, gpus_p_node);

  cudaGetDeviceCount(&aval);
  if(gpus_p_node!=aval){
    if(mytid==root){
      printf("GPUs_p_node input is not same as avaiable GPU/node, need check!!\n");
      printf("ava=%d\n", aval);
    }
    MPI_Barrier(comm);
    //exit(0);
  }
  //  cudaSetDevice(3);
  cudaSetDevice(mytid%gpus_p_node);
  check_gpu_error("Device selection failed");

  if(mytid==root)
    printf("avail=%d gpus\n", aval);
  MPI_Barrier(comm);
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    host_timer_after_gpu_setup = MPI_Wtime();
#endif

  // ---------------------bcast inputs-------------------------
  MPI_Bcast(&order,1,MPI_CHAR,root,comm);
  MPI_Bcast(&near_cut, 1, MPI_FLOAT, root,comm);
  MPI_Bcast(&far_cut, 1, MPI_FLOAT, root,comm);
  MPI_Bcast(&decay_off, 1, MPI_FLOAT, root,comm);

  MPI_Bcast(&ntr_shot,1,MPI_UNSIGNED_LONG,root,comm);
  MPI_Bcast(&nx,1,MPI_INT,root,comm);
  MPI_Bcast(&nz,1,MPI_INT,root,comm);
  MPI_Bcast(&ny,1,MPI_INT,root,comm);
  MPI_Bcast(&dx,1,MPI_FLOAT,root,comm);
  MPI_Bcast(&dy,1,MPI_FLOAT,root,comm);
  MPI_Bcast(&dz,1,MPI_FLOAT,root,comm);
  MPI_Bcast(&npml,1,MPI_INT,root,comm);
  MPI_Bcast(&itop,1,MPI_INT,root,comm);
  MPI_Bcast(&directflag,1,MPI_INT,root,comm);

  MPI_Bcast(&dt,1,MPI_FLOAT,root,comm);
  MPI_Bcast(&nt,1,MPI_INT,root,comm);
  MPI_Bcast(&decay,1,MPI_FLOAT,root,comm); 
  MPI_Bcast(&xpad,1,MPI_FLOAT,root,comm); 
  MPI_Bcast(&vmax,1,MPI_FLOAT,root,comm);

  MPI_Bcast(&shotfile, 400, MPI_CHAR, root, comm);
  MPI_Bcast(&directfile, 400, MPI_CHAR, root, comm);
  MPI_Bcast(&tmutfile, 400, MPI_CHAR, root, comm);
  MPI_Bcast(&bmutfile, 400, MPI_CHAR, root, comm);

  MPI_Bcast(&sht_scl, 1, MPI_FLOAT, root, comm);
  MPI_Bcast(&tmutflag,1,MPI_INT,root,comm);
  MPI_Bcast(&bmutflag,1,MPI_INT,root,comm);

  MPI_Bcast(&L_n,1,MPI_INT,root,comm);
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    host_timer_after_input_bcast = MPI_Wtime();
#endif

  if(L_n>2){
    printf("!!!!!!!!!!!!!!L norm can only be 1 or 2, check, exit!!!!!!!!!!!!!\n");
    exit(0);
  }

  //------------------ abc initialize----------------------
  bt=alloc1float(npml);
  bb=alloc1float(npml);
  bl=alloc1float(npml);
  br=alloc1float(npml);
  init_bc(bt,bb,bl,br,npml,decay);

  //----------------- cpml initialize--------------------
  ax=alloc1float(npml);  bx=alloc1float(npml);
  ay=alloc1float(npml);  by=alloc1float(npml);
  az=alloc1float(npml);  bz=alloc1float(npml);
  ax_h=alloc1float(npml);  bx_h=alloc1float(npml);
  ay_h=alloc1float(npml);  by_h=alloc1float(npml);
  az_h=alloc1float(npml);  bz_h=alloc1float(npml);
  init_cpml_sg_3d(ay, by, ax, bx, az, bz,
		  ay_h, by_h, ax_h, bx_h, az_h, bz_h, 
		  ny, nx, nz, dy, dx, dz, npml, dt, vmax);
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    host_timer_after_coeff_init = MPI_Wtime();
#endif

  nxyz=nx*nz*ny;
  nxy=nx*ny;
  nxz=nx*nz;
  byte=sizeof(float)*nxyz;

  //----------- REM setup------------- NOT USED
  r=pi*(vmax*sqrt(1.0/(dx*dx)+1.0/(dz*dz)+1./(dy*dy)));
  irr=1./r;
  dtr=dt*r;
  j0k=alloc1float(mrecu);
  
  //----------------------- ids---------------------
  vin=alloc3float(nz, nx, ny);
  src=alloc1float(nt);
  wb=alloc2float(nx, ny);
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    host_timer_after_static_alloc = MPI_Wtime();
#endif

  //--------------- ROOT Read vin, shot, setup grids--------------
  if(mytid==root){
    //-------------- dt check-----------
    ddtt=vmax*dt*sqrt(1./(dx*dx)+1./(dz*dz)+1./(dy*dy)); //?????
    printf("dt=%f, limit ddtt=%f\n\n",dt, ddtt);
    if(ddtt>=1.0){
      printf("input dx, dz, dt do not meet the modeling stable condition ddtt=%f\n",ddtt);
      printf("!!!!!\n exit and check dx, dz, dt !!!!!\n");
      //      exit(0);
    }
    
    // read src
    //    src=readdir1d(srcfile, nt);
    ricker1(src, nt, 8, dt);
    //    write1ddir(src, 1, nt, "ricker1_mfre_8hz_nt_1501_dt_0.002");

    // read water bottom geometry
    //    wb=read_dir(wbfile, ny, nx);
    for(iy=0; iy<ny; iy++)
      for(ix=0; ix<nx; ix++)
	wb[iy][ix]=.25-0.025;
   
    // read v initial
    read_dir_3d(vin, ny, nx, nz, vinfile);
    /*
    for(iy=0; iy<ny; iy++)
      for(ix=0; ix<nx; ix++){
	//	for(iz=0; iz<nz/2; iz++)
	//	  vin[iy][ix][iz]=2.;
	for(iz=0; iz<nz; iz++)
	  vin[iy][ix][iz]=3.;
      }
    */
    //    writesu(vin,nx,nz,dx,dz,"v.initial.su");
    
    // find ns, nr per shot, src, rec coordinates
    sflag=0;
    rflag=0;
    // note ns count depends on changes of nav[ic][0] - shot x coordiantes
    read_acqui_shot_3d(navfile, ntr_shot, ny, nx, &ns, &sflag);
    if(sflag!=0){
      printf("sflag error, exit\n");
      return 0; 
    }    
  }
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    host_timer_after_root_model_read = MPI_Wtime();
#endif
  MPI_Bcast(&src[0],nt,MPI_FLOAT,root,comm);
  MPI_Bcast(&wb[0][0],nxy,MPI_FLOAT,root,comm);
  MPI_Bcast(&vin[0][0][0],nxyz,MPI_FLOAT,root,comm);
  MPI_Bcast(&ns,1,MPI_INT,root,comm);
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    host_timer_after_model_bcast = MPI_Wtime();
#endif

  //----------------- src and rec coordiante setup--------------
  // ns total shots, ns_s, # of shots used in each iteration
  nrec_shot=alloc1int(ns);   // # of rec per shot
  src_cor=alloc2float(3,ns); // s cor
  rec_cor=alloc2float(3,ntr_shot); // r cor 

  init_1d_int(nrec_shot,ns);
  init_2d(src_cor,ns,3);
  init_2d(rec_cor,ntr_shot,3);

  ns_s=ns;
  ns_s=MIN(ns_s, ns);
  extra=ns_s%ntids;
  if(extra==0)
    ns_pad=ns_s;
  else
    ns_pad=((int)(ns_s/ntids)+1)*ntids;
  myns=ns_pad/ntids;

  //------------------- root read acqui file------------------------
  if(mytid==root){
    printf(" TOTAL SHOT=%d, each core process %d shots\n",ns_s, myns );
    // need check
    read_acqui_3d(navfile, ntr_shot, ntr_shot, ns,
		  nrec_shot, src_cor, rec_cor, &rflag);
    /*
    for(is=0; is<ns; is++){
      src_cor[is][0]=(ny-1)*dy/2;
      src_cor[is][1]=(nx-1)*dx/2;
      src_cor[is][2]=dz;
      //      src_cor[is][2]=(nz-1)*dz/4;
    }
    */
  }//  end acqui
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    host_timer_after_acqui_read = MPI_Wtime();
#endif
  // bcast acqui
  MPI_Bcast(&nrec_shot[0],ns,MPI_INT,root,comm);
  MPI_Bcast(&src_cor[0][0],3*ns,MPI_FLOAT,root,comm);      
  MPI_Bcast(&rec_cor[0][0],3*ntr_shot,MPI_FLOAT,root,comm);  
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    host_timer_after_acqui_bcast = MPI_Wtime();
#endif

  //--------- interpolation setup for src rec not sitting on modeling grid---------
  src0_indx=alloc2int(3, ns);       // src0 index
  rec0_indx=alloc2int(3, ntr_shot); // rec0 index
  sw000=alloc1float(ns);  sw001=alloc1float(ns);
  sw010=alloc1float(ns);  sw011=alloc1float(ns);
  sw100=alloc1float(ns);  sw101=alloc1float(ns);
  sw110=alloc1float(ns);  sw111=alloc1float(ns);

  rw000=alloc1float(ntr_shot);  rw001=alloc1float(ntr_shot);
  rw010=alloc1float(ntr_shot);  rw011=alloc1float(ntr_shot);
  rw100=alloc1float(ntr_shot);  rw101=alloc1float(ntr_shot);
  rw110=alloc1float(ntr_shot);  rw111=alloc1float(ntr_shot);

  // get src & rec interpolation location coeff
  lint3d_init(ns, src_cor, ny, nx, nz, dy, dx, dz, src0_indx,
	      sw000, sw001, sw010, sw011, sw100, sw101, sw110, sw111);
    
  if(mytid==root)
    for(ic=0;ic<MIN(10, ns);ic++)
      printf("shot %d, (sy=%5.4f sx=%5.4f,sz=%5.4f), indy=%d indx=%d, indz=%d, sw000=%f sw001=%f, sw010=%f, sw011=%f sw100=%f sw101=%f sw110=%f sw111=%f\n",
             ic+1, src_cor[ic][0], src_cor[ic][1], src_cor[ic][2],
	     src0_indx[ic][0], src0_indx[ic][1], src0_indx[ic][2],
             sw000[ic], sw001[ic], sw010[ic], sw011[ic],
             sw100[ic], sw101[ic], sw110[ic], sw111[ic]);
  
  lint3d_init(ntr_shot, rec_cor, ny, nx, nz, dy, dx, dz, rec0_indx,
	      rw000, rw001, rw010, rw011, rw100, rw101, rw110, rw111);
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root)
    host_timer_after_lint = MPI_Wtime();
#endif
  //------------------ build prior bounds ----------------------
  if(mytid==root){
    grad_cur=alloc1float(nxyz);    pre_cur=alloc1float(nxyz);
    memset(&grad_cur[0], 0., byte);    memset(&pre_cur[0], 0., byte);

    // shot array number
    sht_num=alloc1int(ns_pad);
    for(is=0; is<ns_s; is++)
      sht_num[is]=is;
    for(is=ns_s; is<ns_pad; is++)
      sht_num[is]=-999;
    
  }// end root
  else
    sht_num=alloc1int(ns_pad); // workers

  MPI_Bcast(&sht_num[0],ns_pad,MPI_INT,root,comm);
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root){
    host_timer_before_gradient = MPI_Wtime();
    printf("HOST_SETUP_TIMER main input_scan=%lf gpu_setup=%lf input_bcast=%lf coeff_init=%lf static_alloc=%lf root_model_read=%lf model_bcast=%lf acqui_read=%lf acqui_bcast=%lf lint=%lf shot_list=%lf total_pre_gradient=%lf\n",
	   host_timer_after_input_scan - host_timer_after_mpi,
	   host_timer_after_gpu_setup - host_timer_after_input_scan,
	   host_timer_after_input_bcast - host_timer_after_gpu_setup,
	   host_timer_after_coeff_init - host_timer_after_input_bcast,
	   host_timer_after_static_alloc - host_timer_after_coeff_init,
	   host_timer_after_root_model_read - host_timer_after_static_alloc,
	   host_timer_after_model_bcast - host_timer_after_root_model_read,
	   host_timer_after_acqui_read - host_timer_after_model_bcast,
	   host_timer_after_acqui_bcast - host_timer_after_acqui_read,
	   host_timer_after_lint - host_timer_after_acqui_bcast,
	   host_timer_before_gradient - host_timer_after_lint,
	   host_timer_before_gradient - host_timer_after_mpi);
  }
#endif
  //  if(mytid==ntids-1){
  //    for(is=0; is<ns_pad; is++)
  //      printf("shot_num[%d]=%d\n", is, sht_num[is]);
  //  }
  //  MPI_Barrier(comm);
 
  // ------initial gradient and pre_con------
  cal_fwi_grad_3d(&obj_xc_cur, &obj_l_cur, vin,
		  grad_cur, pre_cur, //// vin=>vwb
		  wb, itop, near_cut, far_cut, sht_scl,
		  nrec_shot, sht_num,
		  src0_indx, rec0_indx,
		  sw000, sw001, sw010, sw011,
		  sw100, sw101, sw110, sw111,
		  rw000, rw001, rw010, rw011,
		  rw100, rw101, rw110, rw111,
		  bt, bb, bl, br,
		  ay, by, ax, bx, az, bz,
		  ay_h, by_h, ax_h, bx_h, az_h, bz_h,
		  ns, myns, ns_s, ntr_shot, nt, dt, 1, nt, 0,
		  ny, nx, nz, npml, dy, dx, dz, xpad,
		  j0k, mrecu, irr, src,
		  shotfile, directfile, tmutfile, bmutfile, L_n,
		  tmutflag, bmutflag, directflag, order, vmax, fmflag,
		  ntids, mytid, root, comm);
#ifdef CUDA3D_HOST_SETUP_TIMERS
  if(mytid==root){
    host_timer_after_gradient = MPI_Wtime();
    printf("HOST_SETUP_TIMER main gradient_call_total=%lf\n",
	   host_timer_after_gradient - host_timer_before_gradient);
  }
#endif

  MPI_Barrier(comm);

  // FREE array
  free1int(nrec_shot); free1int(sht_num);
  free2int(src0_indx); free2int(rec0_indx);
  free1float(bt);  free1float(bb);
  free1float(bl);  free1float(br);
  free1float(ax); free1float(bx);
  free1float(ay); free1float(by);
  free1float(az); free1float(bz);
  free1float(ax_h); free1float(bx_h);
  free1float(ay_h); free1float(by_h);
  free1float(az_h); free1float(bz_h);
  free1float(j0k); free1float(src);
  free1float(sw000); free1float(sw001); free1float(sw010);
  free1float(sw011); free1float(sw100); free1float(sw101);
  free1float(sw110); free1float(sw111);
  free1float(rw000); free1float(rw001); free1float(rw010);
  free1float(rw011); free1float(rw100); free1float(rw101);
  free1float(rw110); free1float(rw111);
  free2float(wb); free2float(src_cor); free2float(rec_cor);
  free3float(vin);

  if(mytid==root){
#ifdef CUDA3D_HOST_SETUP_TIMERS
    host_timer_after_main_free = MPI_Wtime();
#endif
    free1float(grad_cur); free1float(pre_cur);
#ifdef CUDA3D_HOST_SETUP_TIMERS
    printf("HOST_SETUP_TIMER main post_gradient_barrier_and_free=%lf total_after_mpi_to_pre_finalize=%lf\n",
	   host_timer_after_main_free - host_timer_after_gradient,
	   host_timer_after_main_free - host_timer_after_mpi);
#endif

    printf("\n*******************ALL DONE******************\n");
  }

#ifdef CUDA3D_HOST_SETUP_TIMERS
  process_timer_before_finalize = cuda3d_wall_seconds();
#endif
  MPI_Finalize();
#ifdef CUDA3D_HOST_SETUP_TIMERS
  process_timer_after_finalize = cuda3d_wall_seconds();
  if(mytid==root)
    printf("HOST_SETUP_TIMER process rank=%d mpi_init=%lf main_after_mpi_to_pre_finalize=%lf mpi_finalize=%lf process_total=%lf\n",
	   mytid,
	   process_timer_after_mpi_init - process_timer_start,
	   process_timer_before_finalize - process_timer_after_mpi_init,
	   process_timer_after_finalize - process_timer_before_finalize,
	   process_timer_after_finalize - process_timer_start);
#endif
  return 0;
}
