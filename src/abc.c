/***************************
void init_cpml(float *ax,float *bx,float *az,float *bz,int nx,int nz,float dx,float dz,int nbd,float dt,float vmax)
void init_bc(float *bt,float *bb,float *bl,float *br,int nbd,float decay);
void bc(float **field,int nx,int nz,int nbd,float *bt,float *bb,float *bl,float *br);
***************************/

#include "abc.h"

//#define pi 3.1415926535897932
void init_cpml(float *ax,float *bx,float *az,float *bz,
	       int nx,int nz,float dx,float dz,int nbd,float dt,float vmax){

  float *ddx,*ddz,*alphax,*alphaz,d0x,d0z,abscissa_PML,abscissa_normalized;
  int nbx,nbz,npower,ix,iz;
  nbx=nx+2*nbd;
  nbz=nz+2*nbd;

  ddx=alloc1float(nbd);
  ddz=alloc1float(nbd);
  alphax=alloc1float(nbd);
  alphaz=alloc1float(nbd);
  //  init_1d(alphax,nbd);
  //  init_1d(alphaz,nbd);
  //  init_1d(ddx,nbd);
  //  init_1d(ddz,nbd);
  for(ix=0; ix<nbd; ix++){
    alphax[ix]=0.;
    alphaz[ix]=0.;
    ddx[ix]=0.0;
    ddz[ix]=0.0;
  }

  npower=2;
  d0x = - (npower+1) * (1.)* log(0.0003) / (2.0 * (nbd-1)*dx); //0.0003
  d0z=d0x;
  
#pragma omp parallel for default(shared) private(ix)
  for(ix=0;ix<nbd;ix++){
    abscissa_PML = (nbd-1-ix)*dx;
    abscissa_normalized = abscissa_PML / (dx*(nbd-1));
    ddx[ix] = d0x * pow(abscissa_normalized,npower);
    alphax[ix] = 8*pi*(1.0 - abscissa_normalized);
  }
  /*
  for(ix=nbd+nx;ix<nbx;ix++){
    abscissa_PML=(ix-nbd-nx)*dx;
    abscissa_normalized = abscissa_PML / (dx*nbd);
    ddx[ix] = d0x * pow(abscissa_normalized,npower);
    alphax[ix] = 8*pi*(1.0 - abscissa_normalized);
  }
  */
#pragma omp parallel for default(shared) private(ix)
  for(ix=0;ix<nbd;ix++){
    bx[ix]=exp(-(ddx[ix]+alphax[ix])*dt);
    //    if(fabsf(ddx[ix]+alphax[ix]-0.0)>1e-4)
    if(fabsf(ddx[ix]>1e-6))
      ax[ix]=ddx[ix]*(bx[ix]-1.0)/(ddx[ix]+alphax[ix]);
  }

#pragma omp parallel for default(shared) private(iz)
  for(iz=0;iz<nbd;iz++){
    abscissa_PML = (nbd-1-iz)*dz;
    abscissa_normalized = abscissa_PML / (dz*(nbd-1));
    ddz[iz] = d0z * pow(abscissa_normalized,npower);
    alphaz[iz] = 8*pi*(1.0 - abscissa_normalized);
  }
  /*
  for(iz=nbd+nz;iz<nbz;iz++){
    abscissa_PML=(iz-nbd-nz)*dz;
    abscissa_normalized = abscissa_PML / (dz*nbd);
    ddz[iz] = d0z * pow(abscissa_normalized,npower);
    alphaz[iz] = 8*pi*(1.0 - abscissa_normalized);
  }
  */
  for(iz=0;iz<nbd;iz++){
    bz[iz]=exp(-(ddz[iz]+alphaz[iz])*dt);
    //    if(fabsf(ddz[iz]+alphaz[iz]-0.0)>1e-4)
    if(fabsf(ddz[iz])>1e-6)
      az[iz]=ddz[iz]*(bz[iz]-1.0)/(ddz[iz]+alphaz[iz]);
  }
}



void init_cpml_sg(float *ax,float *bx,float *az,float *bz,
		  float *ax_h, float *bx_h, float *az_h, float *bz_h,
		  int nx,int nz,float dx,float dz,int nbd,float dt,float vmax){

  float *ddx,*ddz,*alphax,*alphaz,d0x,d0z,abscissa_PML,abscissa_normalized;
  float *ddx_h, *ddz_h, *alphax_h, *alphaz_h, fm, tmp, tmp1, tmp2;
  int nbx,nbz,npower,ix,iz;
  nbx=nx+2*nbd;
  nbz=nz+2*nbd;

  ddx=alloc1float(nbd);
  ddz=alloc1float(nbd);
  alphax=alloc1float(nbd);
  alphaz=alloc1float(nbd);

  ddx_h=alloc1float(nbd);
  ddz_h=alloc1float(nbd);
  alphax_h=alloc1float(nbd);
  alphaz_h=alloc1float(nbd);
  /*
  init_1d(alphax,nbd);
  init_1d(alphaz,nbd);
  init_1d(ddx,nbd);
  init_1d(ddz,nbd);

  init_1d(alphax_h,nbd);
  init_1d(alphaz_h,nbd);
  init_1d(ddx_h,nbd);
  init_1d(ddz_h,nbd);
  */
  for(ix=0; ix<nbd; ix++){
    alphax[ix]=0.0;
    alphaz[ix]=0.0;
    alphax_h[ix]=0.0;
    alphaz_h[ix]=0.0;
    ddx[ix]=0.0;
    ddz[ix]=0.0;
    ddx_h[ix]=0.0;
    ddz_h[ix]=0.0;
  }

  fm=1.;
  tmp=fm*pi;
  npower=2;

  d0x =  (npower+1) *vmax* log(100000.) / (2.0 * (nbd)*dx); //0.0003
  d0z=d0x;

  //  d0x= -5.0*1.5*logf(1.e-7)/(2*nbd*dx);
  //  d0z=d0x;

  //#pragma omp parallel for default(shared) private(ix)
  for(ix=0; ix<nbd; ix++){
    tmp1=(float)(nbd-ix);
    tmp2=tmp1-0.5;

    tmp1=tmp1/(1.*nbd);
    tmp2=tmp2/(1.*nbd);
    tmp1=tmp1*tmp1;
    tmp2=tmp2*tmp2;
    bx[ix]=exp(-d0x*tmp1*dt);
    bx_h[ix]=exp(-d0x*tmp2*dt);
  }
  //#pragma omp parallel for default(shared) private(ix)
  for(ix=0; ix<nbd; ix++){
    tmp1=ix+0.5;
    tmp2=tmp1+0.5;

    tmp1=tmp1/(1.*nbd);
    tmp2=tmp2/(1.*nbd);
    tmp1=tmp1*tmp1;
    tmp2=tmp2*tmp2;
    ax[ix]=exp(-d0x*tmp1*dt);
    ax_h[ix]=exp(-d0x*tmp2*dt);
  }
  //#pragma omp parallel for default(shared) private(iz)
  for(iz=0; iz<nbd; iz++){
    tmp1=(float)(nbd-iz);
    tmp2=tmp1-0.5;

    tmp1=tmp1/(1.*nbd);
    tmp2=tmp2/(1.*nbd);
    tmp1=tmp1*tmp1;
    tmp2=tmp2*tmp2;
    bz[iz]=expf(-d0z*tmp1*dt);
    bz_h[iz]=expf(-d0z*tmp2*dt);
  }

  //#pragma omp parallel for default(shared) private(iz)
  for(iz=0; iz<nbd; iz++){
    tmp1=iz+0.5;
    tmp2=tmp1+0.5;

    tmp1=tmp1/(1.*nbd);
    tmp2=tmp2/(1.*nbd);
    tmp1=tmp1*tmp1;
    tmp2=tmp2*tmp2;
    az[iz]=expf(-d0z*tmp1*dt);
    az_h[iz]=expf(-d0z*tmp2*dt);
  }


  
  /*
  // x direction coeff on int grid
  for(ix=0;ix<nbd;ix++){
    ddx[ix] = d0x * pow(1.*(nbd-1-ix)/(nbd),npower);
    alphax[ix]=tmp*pow(1.*ix/(nbd),1);
  }

  // xdirection coeff on hal grid
  for(ix=0;ix<nbd-1;ix++){
    ddx_h[ix]=d0x*pow(1.*(nbd-1.5-ix)/(nbd),npower);
    alphax_h[ix]=tmp*pi*pow(1.*(ix+0.5)/(nbd),1);
  }
  alphax_h[nbd-1]=alphax_h[nbd-2];

  // x direction ax, bx, ax_h, bx_h
  for(ix=0;ix<nbd;ix++){
    bx[ix]=exp(-(ddx[ix]+alphax[ix])*dt);
    bx_h[ix]=exp(-(ddx_h[ix]+alphax_h[ix])*dt);

    ax[ix]=ddx[ix]*(bx[ix]-1.0)/(ddx[ix]+alphax[ix]);
    ax_h[ix]=ddx_h[ix]*(bx_h[ix]-1.0)/(ddx_h[ix]+alphax_h[ix]);
  }

  // z direction coeff on int grid
  for(iz=0;iz<nbd;iz++){
    ddz[iz]=d0z*pow(1.*(nbd-1-iz)/(nbd),npower);
    alphaz[iz]=tmp*pow(1.*iz/(nbd),1);
  }

  // z direction coeff on half grid
  for(iz=0;iz<nbd-1;iz++){
    ddz_h[iz]=d0z*pow(1.*(nbd-1.5-iz)/(nbd),npower);
    alphaz_h[iz]=tmp*pi*pow(1.*(iz+0.5)/(nbd),1);
  }
  alphaz_h[nbd-1]=alphaz_h[nbd-2];

  // z direction az, bz, az_h, bz_h
  for(iz=0;iz<nbd;iz++){
    bz[iz]=exp(-(ddz[iz]+alphaz[iz])*dt);
    bz_h[iz]=exp(-(ddz_h[iz]+alphaz_h[iz])*dt);

    az[iz]=ddz[iz]*(bz[iz]-1.0)/(ddz[iz]+alphaz[iz]);
    az_h[iz]=ddz_h[iz]*(bz_h[iz]-1.0)/(ddz_h[iz]+alphaz_h[iz]);
  }
*/
  free1float(ddx); free1float(ddz); free1float(alphax); free1float(alphaz);
  free1float(ddx_h); free1float(ddz_h); free1float(alphax_h); free1float(alphaz_h);
}

void init_bc(float *bt,float *bb,float *bl,float *br,int nbd,float decay){
  int ix,iz;
  decay*=decay;
  for(iz=0;iz<nbd;iz++){
    bb[nbd-iz-1]=exp(-decay*(nbd-iz-1)*(nbd-iz-1));
    bt[iz]=bb[nbd-iz-1];
  }
  for(ix=0;ix<nbd;ix++){
    br[nbd-ix-1]=exp(-decay*(nbd-ix-1)*(nbd-ix-1));
    bl[ix]=br[nbd-ix-1];
  }
}
void bc(float **field,int nx,int nz,int nbd,float *bt,float *bb,float *bl,float *br){
  int ix,iz;
#pragma omp parallel for default(shared) private(ix, iz)
  for(ix=nbd;ix<nx-nbd;ix++){
    for(iz=0;iz<nbd;iz++){
      field[ix][iz]*=bt[iz];
      field[ix][nz-nbd+iz]*=bb[iz];
    }
  }
#pragma omp parallel for default(shared) private(ix, iz)
  for(iz=nbd;iz<nz-nbd;iz++){
    for(ix=0;ix<nbd;ix++){
      field[ix][iz]*=bl[ix];
      field[nx-nbd+ix][iz]*=br[ix];
    }
  }
#pragma omp parallel for default(shared) private(ix, iz)
  for(ix=0;ix<nbd;ix++){
    for(iz=0;iz<nbd;iz++){
      field[ix][iz]*=iz>ix?bl[ix]:bt[iz];
      field[ix][nz-iz-1]*=iz>ix?bl[ix]:bb[nbd-1-iz];
      field[nx-ix-1][iz]*=iz>ix?br[nbd-1-ix]:bt[iz];
      field[nx-ix-1][nz-iz-1]*=iz>ix?br[nbd-1-ix]:bb[nbd-1-iz];
    }
  }
}

void bc_3d(float ***d, int ny, int nx, int nz, int nbd,
	   float *bt, float *bb, float *bl, float *br){
  int ix, iy, iz, ib, ibz, ibx, iby;
  /*
#pragma omp parallel default(shared) private(iy, ix, iz)
  {
#pragma omp for
    for (ib=0; ib<nbd; ib++){
      //      ibz=nz-nbd+iz;
      ibz=nz-ib-1;
      for (iy=0; iy<ny; iy++){
	for (ix=0; ix<nx; ix++){
	  d[iy][ix][ib ]*=bt[ib];
	  d[iy][ix][ibz]*=bt[ib];
	}
      }

      ibx=nx-ib-1;
      for(iy=0; iy<ny; iy++){
	for(iz=0; iz<nz; iz++){
	  d[iy][ib ][iz]*=bl[ib];
	  d[iy][ibx][iz]*=bl[ib];
	}
      }

      iby=ny-ib-1;
      for(ix=0; ix<nx; ix++){
	for(iz=0; iz<nz; iz++){
	  d[ib][ix][iz]*=bl[ib];
	  d[iby][ix][iz]*=bl[ib];
	}
      }
    }
  }
  */
  
#pragma omp parallel default(shared) private(iy, ix, iz)
  {
#pragma omp for
  for(iy=nbd; iy<ny-nbd; iy++){
    for(ix=nbd;ix<nx-nbd;ix++){
      for(iz=0;iz<nbd;iz++){
	d[iy][ix][iz]*=bt[iz];
	d[iy][ix][nz-nbd+iz]*=bb[iz];
      }
    }
  }
  //#pragma omp parallel for default(shared) private(ix, iz)
#pragma omp for
  for(iy=nbd; iy<ny-nbd; iy++){
    for(ix=0; ix<nbd; ix++){
      for(iz=nbd; iz<nz-nbd; iz++){
	d[iy][ix][iz]*=bl[ix];
	d[iy][nx-nbd+ix][iz]*=br[ix];
      }
    }
  }
#pragma omp for
  for(iy=0; iy<nbd; iy++){
    for(ix=nbd; ix<nx-nbd; ix++){
      for(iz=nbd; iz<nz-nbd; iz++){
	d[iy][ix][iz]*=bl[iy];
	d[ny-nbd+iy][ix][iz]*=br[iy];
      }
    }
  }
#pragma omp for //default(shared) private(ix, iz)
  for(iy=0; iy<nbd; iy++){
    for(ix=0; ix<nbd; ix++){
      for(iz=0; iz<nbd; iz++){
	d[iy][ix][iz]*=iz>ix?bl[ix]:bt[iz];
	d[iy][ix][nz-iz-1]*=iz>ix?bl[ix]:bb[nbd-1-iz];
	d[iy][nx-ix-1][iz]*=iz>ix?br[nbd-1-ix]:bt[iz];
	d[iy][nx-ix-1][nz-iz-1]*=iz>ix?br[nbd-1-ix]:bb[nbd-1-iz];

	d[ny-iy-1][ix][iz]*=iz>ix?br[nbd-1-ix]:bt[iz];
	d[ny-iy-1][ix][nz-iz-1]*=iz>ix?br[nbd-1-ix]:bb[nbd-1-iz];
	d[ny-iy-1][nx-ix-1][iz]*=iz>ix?br[nbd-1-ix]:bt[iz];
	d[ny-iy-1][nx-ix-1][nz-iz-1]*=iz>ix?br[nbd-1-ix]:bb[nbd-1-iz];
      }
    }
  }
  }
  
}
/*
void bc(float **field,int nx,int nz,int nbd,float *bt,float *bb,float *bl,float *br){
  int ix,iz;

  for(ix=nbd;ix<nx-nbd;ix++){
    for(iz=0;iz<nbd;iz++){
      field[ix][iz]*=bt[iz];
      field[ix][nz-nbd+iz]*=bb[iz];
    }
  }

  for(iz=nbd;iz<nz-nbd;iz++){
    for(ix=0;ix<nbd;ix++){
      field[ix][iz]*=bl[ix];
      field[nx-nbd+ix][iz]*=br[ix];
    }
  }

  for(ix=0;ix<nbd;ix++){
    for(iz=0;iz<nbd;iz++){
      field[ix][iz]*=iz>ix?bl[ix]:bt[iz];
      field[ix][nz-iz-1]*=iz>ix?bl[ix]:bb[nbd-1-iz];
      field[nx-ix-1][iz]*=iz>ix?br[nbd-1-ix]:bt[iz];
      field[nx-ix-1][nz-iz-1]*=iz>ix?br[nbd-1-ix]:bb[nbd-1-iz];
    }
  }
}
*/



void init_cpml_sg_3d(float *ay, float *by, float *ax, float *bx, float *az,float *bz,
		     float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
		     int ny, int nx, int nz, float dy, float dx, float dz, int nbd, float dt, float vmax){

  float *ddx,*ddy, *ddz,*alphax, *alphay, *alphaz, d0x, d0y, d0z,abscissa_PML,abscissa_normalized;
  float *ddx_h, *ddy_h, *ddz_h, *alphax_h, *alphay_h, *alphaz_h, fm, tmp, tmp1, tmp2, tmp3;
  int nbx, nby, nbz,npower,ix, iy, iz;
  nbx=nx+2*nbd;
  nbz=nz+2*nbd;
  nby=ny+2*nbd;

  ddy=alloc1float(nbd);
  ddx=alloc1float(nbd);
  ddz=alloc1float(nbd);
  alphay=alloc1float(nbd);
  alphax=alloc1float(nbd);
  alphaz=alloc1float(nbd);

  ddy_h=alloc1float(nbd);
  ddx_h=alloc1float(nbd);
  ddz_h=alloc1float(nbd);
  alphay_h=alloc1float(nbd);
  alphax_h=alloc1float(nbd);
  alphaz_h=alloc1float(nbd);

  for(ix=0; ix<nbd; ix++){
    alphax[ix]=0.0;
    alphaz[ix]=0.0;
    alphay[ix]=0.0;
    alphax_h[ix]=0.0;
    alphaz_h[ix]=0.0;
    alphay_h[ix]=0.0;
    ddx[ix]=0.0;
    ddz[ix]=0.0;
    ddy[ix]=0.0;
    ddx_h[ix]=0.0;
    ddz_h[ix]=0.0;
    ddy_h[ix]=0.0;
  }

  fm=1.;
  tmp=fm*pi;
  npower=2;

  //  d0x =  (npower+1) *1.99*vmax* log(100000.) / (2.0 * (nbd)*dx); //0.0003 //100000
  d0x =  (npower+1) *vmax* log(1000000.) / (2.0 * (nbd)*dx); //0.0003 //100000
  d0z=d0x;
  d0y=d0x;

  //  d0x= -5.0*1.5*logf(1.e-7)/(2*nbd*dx);
  //  d0z=d0x;

  //#pragma omp parallel for default(shared) private(ix)
  for(ix=0; ix<nbd; ix++){
    tmp1=(float)(nbd-ix);
    tmp2=tmp1-0.5;

    tmp1=tmp1/(1.*nbd);
    tmp2=tmp2/(1.*nbd);
    tmp1=tmp1*tmp1;
    tmp2=tmp2*tmp2;
    bx[ix]=exp(-d0x*tmp1*dt);
    bx_h[ix]=exp(-d0x*tmp2*dt);
  }
  //#pragma omp parallel for default(shared) private(ix)
  for(ix=0; ix<nbd; ix++){
    tmp1=ix+0.5;
    tmp2=tmp1+0.5;

    tmp1=tmp1/(1.*nbd);
    tmp2=tmp2/(1.*nbd);
    tmp1=tmp1*tmp1;
    tmp2=tmp2*tmp2;
    ax[ix]=exp(-d0x*tmp1*dt);
    ax_h[ix]=exp(-d0x*tmp2*dt);
  }

  for(iy=0; iy<nbd; iy++){
    tmp1=(float)(nbd-iy);
    tmp2=tmp1-0.5;

    tmp1=tmp1/(1.*nbd);
    tmp2=tmp2/(1.*nbd);
    tmp1=tmp1*tmp1;
    tmp2=tmp2*tmp2;
    by[iy]=expf(-d0y*tmp1*dt);
    by_h[iy]=expf(-d0y*tmp2*dt);
  }

  //#pragma omp parallel for default(shared) private(iz)
  for(iy=0; iy<nbd; iy++){
    tmp1=iy+0.5;
    tmp2=tmp1+0.5;

    tmp1=tmp1/(1.*nbd);
    tmp2=tmp2/(1.*nbd);
    tmp1=tmp1*tmp1;
    tmp2=tmp2*tmp2;
    ay[iy]=expf(-d0y*tmp1*dt);
    ay_h[iy]=expf(-d0y*tmp2*dt);
  }

  //#pragma omp parallel for default(shared) private(iz)
  for(iz=0; iz<nbd; iz++){
    tmp1=(float)(nbd-iz);
    tmp2=tmp1-0.5;

    tmp1=tmp1/(1.*nbd);
    tmp2=tmp2/(1.*nbd);
    tmp1=tmp1*tmp1;
    tmp2=tmp2*tmp2;
    bz[iz]=expf(-d0z*tmp1*dt);
    bz_h[iz]=expf(-d0z*tmp2*dt);
  }

  //#pragma omp parallel for default(shared) private(iz)
  for(iz=0; iz<nbd; iz++){
    tmp1=iz+0.5;
    tmp2=tmp1+0.5;

    tmp1=tmp1/(1.*nbd);
    tmp2=tmp2/(1.*nbd);
    tmp1=tmp1*tmp1;
    tmp2=tmp2*tmp2;
    az[iz]=expf(-d0z*tmp1*dt);
    az_h[iz]=expf(-d0z*tmp2*dt);
  }


  
  /*
  // x direction coeff on int grid
  for(ix=0;ix<nbd;ix++){
    ddx[ix] = d0x * pow(1.*(nbd-1-ix)/(nbd),npower);
    alphax[ix]=tmp*pow(1.*ix/(nbd),1);
  }

  // xdirection coeff on hal grid
  for(ix=0;ix<nbd-1;ix++){
    ddx_h[ix]=d0x*pow(1.*(nbd-1.5-ix)/(nbd),npower);
    alphax_h[ix]=tmp*pi*pow(1.*(ix+0.5)/(nbd),1);
  }
  alphax_h[nbd-1]=alphax_h[nbd-2];

  // x direction ax, bx, ax_h, bx_h
  for(ix=0;ix<nbd;ix++){
    bx[ix]=exp(-(ddx[ix]+alphax[ix])*dt);
    bx_h[ix]=exp(-(ddx_h[ix]+alphax_h[ix])*dt);

    ax[ix]=ddx[ix]*(bx[ix]-1.0)/(ddx[ix]+alphax[ix]);
    ax_h[ix]=ddx_h[ix]*(bx_h[ix]-1.0)/(ddx_h[ix]+alphax_h[ix]);
  }

  // z direction coeff on int grid
  for(iz=0;iz<nbd;iz++){
    ddz[iz]=d0z*pow(1.*(nbd-1-iz)/(nbd),npower);
    alphaz[iz]=tmp*pow(1.*iz/(nbd),1);
  }

  // z direction coeff on half grid
  for(iz=0;iz<nbd-1;iz++){
    ddz_h[iz]=d0z*pow(1.*(nbd-1.5-iz)/(nbd),npower);
    alphaz_h[iz]=tmp*pi*pow(1.*(iz+0.5)/(nbd),1);
  }
  alphaz_h[nbd-1]=alphaz_h[nbd-2];

  // z direction az, bz, az_h, bz_h
  for(iz=0;iz<nbd;iz++){
    bz[iz]=exp(-(ddz[iz]+alphaz[iz])*dt);
    bz_h[iz]=exp(-(ddz_h[iz]+alphaz_h[iz])*dt);

    az[iz]=ddz[iz]*(bz[iz]-1.0)/(ddz[iz]+alphaz[iz]);
    az_h[iz]=ddz_h[iz]*(bz_h[iz]-1.0)/(ddz_h[iz]+alphaz_h[iz]);
  }
*/
  free1float(ddy); free1float(ddx); free1float(ddz); free1float(alphay); free1float(alphax); free1float(alphaz);
  free1float(ddy_h); free1float(ddx_h); free1float(ddz_h); free1float(alphay_h); free1float(alphax_h); free1float(alphaz_h);
}
