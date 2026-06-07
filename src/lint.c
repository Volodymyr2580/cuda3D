#include "lint.h"

void lint2d_inject_bell(float **f, int nbd, int xl,
			float ss, float **bell, int nbell, int *index,
			float w00, float w01, float w10, float w11){

  int ix, iz;
  float aaa;
  //#pragma omp parallel for default(shared) private(ix, iz)
  for (ix=-nbell; ix<=nbell; ix++){
    for(iz=-nbell; iz<=nbell; iz++){
      aaa=ss*bell[nbell+ix][nbell+iz];
      f[nbd+ix+index[0]  -xl][nbd+iz+index[1]  ]+=aaa*w00;
      f[nbd+ix+index[0]  -xl][nbd+iz+index[1]+1]+=aaa*w01;
      f[nbd+ix+index[0]+1-xl][nbd+iz+index[1]  ]+=aaa*w10;
      f[nbd+ix+index[0]+1-xl][nbd+iz+index[1]+1]+=aaa*w11;

    }
  }
}

void lint2d_init(size_t na, float **aa,
		 int nx, int nz, float dx, float dz,
		 int **index,
		 float *w00, float *w01, float *w10, float *w11){
  size_t ia;
  float f1, f2, tmp0, tmp1;
  //index[i][0] x 
  //index[i][1] z

  for(ia=0;ia<na;ia++){

    if(aa[ia][0]>=0 &&
       aa[ia][0]<=(nx-1)*dx+1e-5 &&
       aa[ia][1]>=0 &&
       aa[ia][1]<=(nz-1)*dz+1e-5){

      //      index[ia][0]=(int)((aa[ia][0]+1e-5)/dx+1e-3);
      //      index[ia][1]=(int)((aa[ia][1]+1e-5)/dz+1e-3);

      //      f2=(aa[ia][0])/dx-index[ia][0];
      //      f1=(aa[ia][1])/dz-index[ia][1];  //need to be positive!!!
      //      tmp0=roundf(aa[ia][0]/dx*10000)/10000;
      //      tmp1=roundf(aa[ia][1]/dz*10000)/10000;
      tmp0=aa[ia][0]*1000/(dx*1000);
      tmp1=aa[ia][1]*1000/(dz*1000);
      index[ia][0]=(int)(tmp0);
      index[ia][1]=(int)(tmp1);

      f2=fabsf(tmp0-index[ia][0]);
      f1=fabsf(tmp1-index[ia][1]);  //need to be positive!!!

      w00[ia]=(1-f1)*(1-f2);
      w01[ia]=(  f1)*(1-f2);
      w10[ia]=(1-f1)*(  f2);
      w11[ia]=(  f1)*(  f2);
    }
    else{
      index[ia][0]=0;
      index[ia][1]=0;

      f1=0.; //?1.
      f2=0.;
      printf("WARNING !!!!!!\n");
      printf("EXT array OUTSIDE THE MODEL, CHECK INPUT ACQUI AND NAV or enlarge the domain!!!!!!!\n");
      exit(0);
      w00[ia]=0.0;
      w01[ia]=0.0;
      w10[ia]=0.0;
      w11[ia]=0.0;
    }

  }

  /********
    00 (x z)        10 (x+1 z)


    01 (x z+1)      11 (x+1 z+1)
   ******/
}
/*
void lint2d_inject_src(double **Bx, double **Bz,
		       int npml, int pad1, int its,
		       double complex ww,
		       int *index,
		       float w00, float w01, float w10, float w11)
//< inject into wavefield >
{
    int   ia;
    float wa;
    
#ifdef _OPENMP
#pragma omp parallel for \
    schedule(dynamic,1) \
    private(ia,wa) \
    shared(ca,ww,uu)
#endif
    
    //    for (ia=0;ia<ca->n;ia++) {
      //	wa = ww[ia];
    //    Bx[its][(-1+sxx+npml)*(pad1-2)+(-1+npml+sz)]=creal(src);
    Bx[its][(-1+npml+index[0]  )*(pad1-2)+(-1+npml+index[1]  )] = creal(ww) * w00;
    Bx[its][(-1+npml+index[0]  )*(pad1-2)+(-1+npml+index[1]+1)] = creal(ww) * w01;
    Bx[its][(-1+npml+index[0]+1)*(pad1-2)+(-1+npml+index[1]  )] = creal(ww) * w10;
    Bx[its][(-1+npml+index[0]+1)*(pad1-2)+(-1+npml+index[1]+1)] = creal(ww) * w11;

    Bz[its][(-1+npml+index[0]  )*(pad1-2)+(-1+npml+index[1]  )] = cimag(ww) * w00;
    Bz[its][(-1+npml+index[0]  )*(pad1-2)+(-1+npml+index[1]+1)] = cimag(ww) * w01;
    Bz[its][(-1+npml+index[0]+1)*(pad1-2)+(-1+npml+index[1]  )] = cimag(ww) * w10;
    Bz[its][(-1+npml+index[0]+1)*(pad1-2)+(-1+npml+index[1]+1)] = cimag(ww) * w11;
    //	uu[ ca->jx[ia]   ][ ca->jz[ia]+1 ] += wa * w01[ia];
    //	uu[ ca->jx[ia]+1 ][ ca->jz[ia]   ] += wa * w10[ia];
    //	uu[ ca->jx[ia]+1 ][ ca->jz[ia]+1 ] += wa * w11[ia];
	//}
}
*/
void lint2d_inject_rec(double **Bx, double **Bz,
		       int npml, int pad1, int its,
		       float *wwr, float *wwi, int ir,
		       int *index,
		       float w00, float w01, float w10, float w11)
/*< inject into wavefield >*/
{
  //    int   ia;
  //    float wa;
    /*
#ifdef _OPENMP
#pragma omp parallel for \
    schedule(dynamic,1) \
    private(ia,wa) \
    shared(ca,ww,uu)
#endif
    */
    //    for (ia=0;ia<ca->n;ia++) {
      //	wa = ww[ia];
    //    Bx[its][(-1+sxx+npml)*(pad1-2)+(-1+npml+sz)]=creal(src);
    Bx[its][(-1+npml+index[0]  )*(pad1-2)+(-1+npml+index[1]  )] = wwr[ir] * w00;
    Bx[its][(-1+npml+index[0]  )*(pad1-2)+(-1+npml+index[1]+1)] = wwr[ir] * w01;
    Bx[its][(-1+npml+index[0]+1)*(pad1-2)+(-1+npml+index[1]  )] = wwr[ir] * w10;
    Bx[its][(-1+npml+index[0]+1)*(pad1-2)+(-1+npml+index[1]+1)] = wwr[ir] * w11;

    Bz[its][(-1+npml+index[0]  )*(pad1-2)+(-1+npml+index[1]  )] = -wwi[ir] * w00;
    Bz[its][(-1+npml+index[0]  )*(pad1-2)+(-1+npml+index[1]+1)] = -wwi[ir] * w01;
    Bz[its][(-1+npml+index[0]+1)*(pad1-2)+(-1+npml+index[1]  )] = -wwi[ir] * w10;
    Bz[its][(-1+npml+index[0]+1)*(pad1-2)+(-1+npml+index[1]+1)] = -wwi[ir] * w11; // attention sign
    //	uu[ ca->jx[ia]   ][ ca->jz[ia]+1 ] += wa * w01[ia];
    //	uu[ ca->jx[ia]+1 ][ ca->jz[ia]   ] += wa * w10[ia];
    //	uu[ ca->jx[ia]+1 ][ ca->jz[ia]+1 ] += wa * w11[ia];
	//}
}

void lint2d_inject_rec_rem(float **fu, int nbd, int xl,
			   float d, float **bell, int nbell, int *index,
			   float w00, float w01, float w10, float w11){


  int ix, iz;
  float aaa;
  
  /*
  for (ix=-nbell; ix<=nbell; ix++){
    for(iz=-nbell; iz<=nbell; iz++){
      aaa=d*bell[nbell+ix][nbell+iz];
      fu[nbd+ix+index[0]  -xl][nbd+iz+index[1]  ]=aaa*w00;
      fu[nbd+ix+index[0]  -xl][nbd+iz+index[1]+1]=aaa*w01;
      fu[nbd+ix+index[0]+1-xl][nbd+iz+index[1]  ]=aaa*w10;
      fu[nbd+ix+index[0]+1-xl][nbd+iz+index[1]+1]=aaa*w11;

    }
  }
  */
  
  fu[nbd+index[0]  -xl][nbd+index[1]  ]=d*w00;
  fu[nbd+index[0]  -xl][nbd+index[1]+1]=d*w01;
  fu[nbd+index[0]+1-xl][nbd+index[1]  ]=d*w10;
  fu[nbd+index[0]+1-xl][nbd+index[1]+1]=d*w11;
  
}


void lint2d_inject_rec_rem_2(float **fu, int nbd, int xl,
			   float d, int *index,
			   float w00, float w01, float w10, float w11){

  // not quite right
  int ix, iz;
  float aaa;
  //#pragma omp parallel for default(shared) private(ix, iz)
  /*
  for (ix=-nbell; ix<=nbell; ix++){
    for(iz=-nbell; iz<=nbell; iz++){
      aaa=ss*bell[nbell+ix][nbell+iz];
      f[nbd+ix+index[0]  -xl][nbd+iz+index[1]  ]+=aaa*w00;
      f[nbd+ix+index[0]  -xl][nbd+iz+index[1]+1]+=aaa*w01;
      f[nbd+ix+index[0]+1-xl][nbd+iz+index[1]  ]+=aaa*w10;
      f[nbd+ix+index[0]+1-xl][nbd+iz+index[1]+1]+=aaa*w11;

    }
  }
  */
  fu[nbd+index[0]  -xl][nbd+index[1]  ]+=d*w00;
  fu[nbd+index[0]  -xl][nbd+index[1]+1]+=d*w01;
  fu[nbd+index[0]+1-xl][nbd+index[1]  ]+=d*w10;
  fu[nbd+index[0]+1-xl][nbd+index[1]+1]+=d*w11;

}

void lint2d_extract(double **Xx, double **Xz,
		    int npml, int pad1, int its,
		    float *ddr, float *ddi, int ir, int *index,
		    float w00, float w01, float w10, float w11)
/*< extract from wavefield >*/
{
    int ia;
    /*
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1) private(ia) shared(ca,dd,uu)
#endif
    */
    //    for (ia=0;ia<ca->n;ia++) {
    /*
    ddr[ir] = creal(uu[ index[0]  ][ index[1]  ]) * w00 +
      creal(uu[ index[0]  ][ index[1]+1]) * w01 +
      creal(uu[ index[0]+1][ index[1]  ]) * w10 +
      creal(uu[ index[0]+1][ index[1]+1]) * w11;

    ddi[ir] = cimag(uu[ index[0]  ][ index[1]  ]) * w00 +
      cimag(uu[ index[0]  ][ index[1]+1]) * w01 +
      cimag(uu[ index[0]+1][ index[1]  ]) * w10 +
      cimag(uu[ index[0]+1][ index[1]+1]) * w11;
	//    }
	*/

    //      ic=(j-1)*(pad1-2)+(i-1);
      ddr[ir] = 
	(float) Xx[its][(-1+npml+index[0]  )*(pad1-2)+(-1+npml+index[1]  )]*w00+
	(float) Xx[its][(-1+npml+index[0]  )*(pad1-2)+(-1+npml+index[1]+1)]*w01+
	(float) Xx[its][(-1+npml+index[0]+1)*(pad1-2)+(-1+npml+index[1]  )]*w10+
	(float) Xx[its][(-1+npml+index[0]+1)*(pad1-2)+(-1+npml+index[1]+1)]*w11;

      ddi[ir] = 
	(float) Xz[its][(-1+npml+index[0]  )*(pad1-2)+(-1+npml+index[1]  )]*w00+
	(float) Xz[its][(-1+npml+index[0]  )*(pad1-2)+(-1+npml+index[1]+1)]*w01+
	(float) Xz[its][(-1+npml+index[0]+1)*(pad1-2)+(-1+npml+index[1]  )]*w10+
	(float) Xz[its][(-1+npml+index[0]+1)*(pad1-2)+(-1+npml+index[1]+1)]*w11;

}  

void lint2d_extract1(float *gradi,  float *hessi, float *vi,
		     int nx, int nz, int nnz,
		     float *gradd, float *hessd, float *vd, int ir, int *index,
		     float w00, float w01, float w10, float w11)
/*< extract from wavefield >*/
{
    int ia;
    /*
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1) private(ia) shared(ca,dd,uu)
#endif
    */

    gradd[ir] = 
      gradi[(index[0]  )*(nz)+(index[1]  )]*w00+
      gradi[(index[0]  )*(nz)+MIN(nz-1,index[1]+1)]*w01+
      gradi[MIN(nx-1,index[0]+1)*(nz)+(index[1]  )]*w10+
      gradi[MIN(nx-1,index[0]+1)*(nz)+MIN(nz-1,index[1]+1)]*w11;

    hessd[ir] = 
      hessi[(index[0]  )*(nz)+(index[1]  )]*w00+
      hessi[(index[0]  )*(nz)+MIN(nz-1,index[1]+1)]*w01+
      hessi[MIN(nx-1,index[0]+1)*(nz)+(index[1]  )]*w10+
      hessi[MIN(nx-1,index[0]+1)*(nz)+MIN(nz-1,index[1]+1)]*w11;

    vd[ir] = 
      vi[(index[0]  )*(nz)+(index[1]  )]*w00+
      vi[(index[0]  )*(nz)+MIN(nz-1,index[1]+1)]*w01+
      vi[MIN(nx-1,index[0]+1)*(nz)+(index[1]  )]*w10+
      vi[MIN(nx-1,index[0]+1)*(nz)+MIN(nz-1,index[1]+1)]*w11;

}  
void lint2d_extract2(float **in, int nx, int nz,
		     float **dout, int ix, int iz, int *index,
		     float w00, float w01, float w10, float w11)
/*< extract from wavefield >*/
{
    int ia;
    /*
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1) private(ia) shared(ca,dd,uu)
#endif
    */

    dout[ix][iz] = 
      in[(index[0]  )][(index[1]  )]*w00+
      in[(index[0]  )][MIN(nz-1,index[1]+1)]*w01+
      in[MIN(nx-1,index[0]+1)][(index[1]  )]*w10+
      in[MIN(nx-1,index[0]+1)][MIN(nz-1,index[1]+1)]*w11;

}  

void lint2d_extract3(float **in, int min_all,
		     float *dout, int *index,
		     float w00, float w01, float w10, float w11)
{
    *dout = 
      in[(index[0]  )-min_all][(index[1]  )]*w00+
      in[(index[0]  )-min_all][(index[1]+1)]*w01+
      in[(index[0]+1)-min_all][(index[1]  )]*w10+
      in[(index[0]+1)-min_all][(index[1]+1)]*w11;
}  

void lint2d_extract_new(float *gradi,  float *hessi,
			int nx, int nz, int nnz,
			float *gradd, float *hessd, int ir, int *index,
			float w00, float w01, float w10, float w11)
/*< extract from wavefield >*/
{
    int ia;
    /*
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1) private(ia) shared(ca,dd,uu)
#endif
    */
    size_t i1, i2, i3, i4;
    i1=(index[0]  )*(nz)+(index[1]  );
    i2=(index[0]  )*(nz)+MIN(nz-1,index[1]+1);
    i3=MIN(nx-1,index[0]+1)*(nz)+(index[1]  );
    i4=MIN(nx-1,index[0]+1)*(nz)+MIN(nz-1,index[1]+1);
    gradd[ir] = 
      gradi[i1]*w00+
      gradi[i2]*w01+
      gradi[i3]*w10+
      gradi[i4]*w11;

    hessd[ir] = 
      hessi[i1]*w00+
      hessi[i2]*w01+
      hessi[i3]*w10+
      hessi[i4]*w11;
}  


void lint3d_init(size_t na, float **aa,
		 int ny, int nx, int nz, float dy, float dx, float dz,
		 int **index,
		 float *w000, float *w001, float *w010, float *w011, 
		 float *w100, float *w101, float *w110, float *w111){
  size_t ia;
  float f1, f2, f3, tmp0, tmp1, tmp2;
  //index[i][0] y 
  //index[i][1] x
  //index[i][2] z

  for(ia=0;ia<na;ia++){
    if(aa[ia][0]>=0 &&
       aa[ia][0]<=(ny-1)*dy+1.e-5 &&
       aa[ia][1]>=0 &&
       aa[ia][1]<=(nx-1)*dx+1.e-5 &&
       aa[ia][2]>=0 &&
       aa[ia][2]<=(nz-1)*dz+1.e-5){

      tmp0=aa[ia][0]*1000/(dy*1000);
      tmp1=aa[ia][1]*1000/(dx*1000);
      tmp2=aa[ia][2]*1000/(dz*1000);
      index[ia][0]=(int)(tmp0);
      index[ia][1]=(int)(tmp1);
      index[ia][2]=(int)(tmp2);

      f3=fabsf(tmp0-index[ia][0]);
      f2=fabsf(tmp1-index[ia][1]);  //need to be positive!!!
      f1=fabsf(tmp2-index[ia][2]);  //need to be positive!!!

      w000[ia]=(1-f1)*(1-f2)*(1-f3);
      w001[ia]=(  f1)*(1-f2)*(1-f3);
      w010[ia]=(1-f1)*(  f2)*(1-f3);
      w011[ia]=(  f1)*(  f2)*(1-f3);

      w100[ia]=(  f3)*(1-f1)*(1-f2);
      w101[ia]=(  f3)*(  f1)*(1-f2);
      w110[ia]=(  f3)*(1-f1)*(  f2);
      w111[ia]=(  f3)*(  f1)*(  f2);
    }
    else{
      index[ia][0]=0;
      index[ia][1]=0;
      index[ia][2]=0;

      f1=0.; //?1.
      f2=0.;
      f3=0.;
      printf("WARNING !!!!!!\n");

      printf("ia=%zu y=%f x=%f z=%f\n", ia, aa[ia][0], aa[ia][1], aa[ia][2]);
      printf("EXT array OUTSIDE THE MODEL, CHECK INPUT ACQUI AND NAV or enlarge the domain!!!!!!!\n");
      exit(0);
      //      w00[ia]=0.0;
      //      w01[ia]=0.0;
      //      w10[ia]=0.0;
      //      w11[ia]=0.0;
    }

  }

  /********
    000 (y x z)        010 (y x+1 z)


    001 (y x z+1)      011 (y x+1 z+1)


   ******/
}


void lint3d_extract(float *dd,  float ***in, size_t na, int **index,
		    int ny, int nx, int nz, 
		    float *w000, float *w001, float *w010, float *w011,
		    float *w100, float *w101, float *w110, float *w111)
/*< extract from wavefield >*/
{    size_t ia;
    /*
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1) private(ia) shared(ca,dd,uu)
#endif
    */
    for (ia=0; ia<na; ia++)
      dd[ia] = 
	in[index[ia][0]  ][index[ia][1]  ][index[ia][2]  ]*w000[ia]+
	in[index[ia][0]  ][index[ia][1]  ][MIN(nz-1, index[ia][2]+1)]*w001[ia]+
	in[index[ia][0]  ][MIN(nx-1, index[ia][1]+1)][index[ia][2]  ]*w010[ia]+
	in[index[ia][0]  ][MIN(nx-1, index[ia][1]+1)][MIN(nz-1, index[ia][2]+1)]*w011[ia]+
	in[MIN(ny-1, index[ia][0]+1)][index[ia][1]  ][index[ia][2]  ]*w100[ia]+
	in[MIN(ny-1, index[ia][0]+1)][index[ia][1]  ][MIN(nz-1, index[ia][2]+1)]*w101[ia]+
	in[MIN(ny-1, index[ia][0]+1)][MIN(nx-1, index[ia][1]+1)][index[ia][2]  ]*w110[ia]+
	in[MIN(ny-1, index[ia][0]+1)][MIN(nx-1, index[ia][1]+1)][MIN(nz-1, index[ia][2]+1)]*w111[ia];
	//	in[index[ia][0]  ][index[ia][1]  ][index[ia][2]  ]*w000[ia]+
	//	in[index[ia][0]  ][index[ia][1]  ][MIN(nz-1, index[ia][2]+1)]*w001[ia]+
	//	in[index[ia][0]  ][MIN(nx-1, index[ia][1]+1)][index[ia][2]]*w010[ia]+
	//	in[index[ia][0]  ][MIN(nx-1, index[ia][1]+1)][MIN(nz-1, index[ia][2]+1)]*w011[ia]+
}  

void lint3d_extract2(float *dd,  float ***in, size_t na, 
		     int yl, int xl, int **index, size_t icc,
		     int ny, int nx, int nz, int nbd,
		     float *w000, float *w001, float *w010, float *w011,
		     float *w100, float *w101, float *w110, float *w111)
/*< extract from wavefield >*/
{    size_t ia, iaa;
    /*
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic,1) private(ia) shared(ca,dd,uu)
#endif
    */
  for (ia=0; ia<na; ia++){
    iaa=icc+ia;
    dd[ia] = 
      in[index[iaa][0]-yl +nbd][index[iaa][1]  -xl +nbd][index[iaa][2]  +nbd]*w000[iaa]+
      in[index[iaa][0]-yl +nbd][index[iaa][1]  -xl +nbd][index[iaa][2]+1+nbd]*w001[iaa]+
      in[index[iaa][0]-yl +nbd][index[iaa][1]+1-xl +nbd][index[iaa][2]  +nbd]*w010[iaa]+
      in[index[iaa][0]-yl +nbd][index[iaa][1]+1-xl +nbd][index[iaa][2]+1+nbd]*w011[iaa]+
      in[index[iaa][0]+1-yl+nbd][index[iaa][1]  -xl +nbd][index[iaa][2]  +nbd]*w100[iaa]+
      in[index[iaa][0]+1-yl+nbd][index[iaa][1]  -xl +nbd][index[iaa][2]+1+nbd]*w101[iaa]+
      in[index[iaa][0]+1-yl+nbd][index[iaa][1]+1-xl+nbd][index[iaa][2]  +nbd]*w110[iaa]+
      in[index[iaa][0]+1-yl+nbd][index[iaa][1]+1-xl+nbd][index[iaa][2]+1+nbd]*w111[iaa];
    /*
    iaa=icc+ia;
    dd[ia] = 
      in[index[iaa][0]-yl +nbd][index[iaa][1]-xl +nbd][index[iaa][2] +nbd]*w000[iaa]+
      in[index[iaa][0]-yl +nbd][index[iaa][1]-xl +nbd][MIN(nz-1, index[iaa][2]+1)+nbd]*w001[iaa]+
      in[index[iaa][0]-yl +nbd][MIN(nx-1, index[iaa][1]+1)-xl +nbd][index[iaa][2]+nbd]*w010[iaa]+
      in[index[iaa][0]-yl +nbd][MIN(nx-1, index[iaa][1]+1)-xl +nbd][MIN(nz-1, index[iaa][2]+1)+nbd]*w011[iaa]+
      in[MIN(ny-1, index[iaa][0]+1)-yl+nbd][index[iaa][1]-xl +nbd][index[iaa][2]+nbd]*w100[iaa]+
      in[MIN(ny-1, index[iaa][0]+1)-yl+nbd][index[iaa][1]-xl +nbd][MIN(nz-1, index[iaa][2]+1)+nbd]*w101[iaa]+
      in[MIN(ny-1, index[iaa][0]+1)-yl+nbd][MIN(nx-1, index[iaa][1]+1)-xl+nbd][index[iaa][2]+nbd]*w110[iaa]+
      in[MIN(ny-1, index[iaa][0]+1)-yl+nbd][MIN(nx-1, index[iaa][1]+1)-xl+nbd][MIN(nz-1, index[iaa][2]+1)+nbd]*w111[iaa];
*/
    //nbd+index[iaa][2]+1
  }  
}

void lint3d_inject_bell(float ***f, int nbd, int yl, int xl,
			float ss, float ***bell, int nbell, int *index,
			float w000, float w001, float w010, float w011,
			float w100, float w101, float w110, float w111){

  int iy, ix, iz;
  float aaa;
  //#pragma omp parallel for default(shared) private(ix, iz)
  for (iy=-nbell; iy<=nbell; iy++){
    for (ix=-nbell; ix<=nbell; ix++){
      for(iz=-nbell; iz<=nbell; iz++){
	aaa=ss*bell[nbell+iy][nbell+ix][nbell+iz];
	f[nbd+iy+index[0]  -yl][nbd+ix+index[1]  -xl][nbd+iz+index[2]  ]+=aaa*w000;
	f[nbd+iy+index[0]  -yl][nbd+ix+index[1]  -xl][nbd+iz+index[2]+1]+=aaa*w001;
	f[nbd+iy+index[0]  -yl][nbd+ix+index[1]+1-xl][nbd+iz+index[2]  ]+=aaa*w010;
	f[nbd+iy+index[0]  -yl][nbd+ix+index[1]+1-xl][nbd+iz+index[2]+1]+=aaa*w011;
	f[nbd+iy+index[0]+1-yl][nbd+ix+index[1]  -xl][nbd+iz+index[2]  ]+=aaa*w100;
	f[nbd+iy+index[0]+1-yl][nbd+ix+index[1]  -xl][nbd+iz+index[2]+1]+=aaa*w101;
	f[nbd+iy+index[0]+1-yl][nbd+ix+index[1]+1-xl][nbd+iz+index[2]  ]+=aaa*w110;
	f[nbd+iy+index[0]+1-yl][nbd+ix+index[1]+1-xl][nbd+iz+index[2]+1]+=aaa*w111;
	
      }
    }
  }
}



void lint3d_inject_rec(float **d, float ***fu, int it, size_t na, int yl, int xl,
		       int **index, size_t icc, int nbd,
		       float *w000, float *w001, float *w010, float *w011,
		       float *w100, float *w101, float *w110, float *w111){
  size_t ia, iaa;
  //#pragma omp parallel for default(shared) private(ix, iz)
  //  fu[nbd+index[0]  -xl][nbd+index[1]  ]+=d*w00;
  //  fu[nbd+index[0]  -xl][nbd+index[1]+1]+=d*w01;
  //  fu[nbd+index[0]+1-xl][nbd+index[1]  ]+=d*w10;
  //  fu[nbd+index[0]+1-xl][nbd+index[1]+1]+=d*w11;

  for (ia=0; ia<na; ia++){
    iaa=icc+ia;
    fu[nbd+index[iaa][0]  -yl][nbd+index[iaa][1]  -xl][nbd+index[iaa][2]  ]+=d[it][ia]*w000[iaa];
    fu[nbd+index[iaa][0]  -yl][nbd+index[iaa][1]  -xl][nbd+index[iaa][2]+1]+=d[it][ia]*w001[iaa];
    fu[nbd+index[iaa][0]  -yl][nbd+index[iaa][1]+1-xl][nbd+index[iaa][2]  ]+=d[it][ia]*w010[iaa];
    fu[nbd+index[iaa][0]  -yl][nbd+index[iaa][1]+1-xl][nbd+index[iaa][2]+1]+=d[it][ia]*w011[iaa];
    fu[nbd+index[iaa][0]+1-yl][nbd+index[iaa][1]  -xl][nbd+index[iaa][2]  ]+=d[it][ia]*w100[iaa];
    fu[nbd+index[iaa][0]+1-yl][nbd+index[iaa][1]  -xl][nbd+index[iaa][2]+1]+=d[it][ia]*w101[iaa];
    fu[nbd+index[iaa][0]+1-yl][nbd+index[iaa][1]+1-xl][nbd+index[iaa][2]  ]+=d[it][ia]*w110[iaa];
    fu[nbd+index[iaa][0]+1-yl][nbd+index[iaa][1]+1-xl][nbd+index[iaa][2]+1]+=d[it][ia]*w111[iaa];
  }

}
