/****************************************************************************
float **readsu(char *name,int nx,int nt);
void writesu(float **vout,int nx,int nt,float dx,float dt,char *name);
void write1dsu(float *out,size_t nx,size_t nt,float dx,float dt,char *name);

float **readsegy(char *name,int nx,int nt);
void writesegy(float **vout,int nx,int nt,float dx,float dt,char *name);
float **readdir(char *name,int nx,int nz);
void writedir(float **vout,int nx,int nz,char *name);

void init_2d(float **data,int nx,int nz);
void init_3d(float ***data,int nx,int nz,int nt);

void vpad(float **v,float **vvpad,int nx,int nz,int nbd);
void ricker(float *pulse,int nt,int fre, float dt,float tdelay);
void ricker1(float *pulse,int nt,int fre, float dt);
*****************************************************************************/
#include "utility_zz.h"
#define iacc 40
#define bigno 1.e+10
#define bigni 1.e-10
#define	sgn(x,y) ((x) < (y) ? (1) : (-1))
/**********************************/
void ibm_to_float(int from[], int to[], int n, int endian)
{
  register int fconv, fmant, i, t;

  for (i = 0;i < n; ++i) {
    fconv = from[i];
    /* if little endian, i.e. endian=0 do this */
    if (endian == 0) fconv = (fconv << 24) | ((fconv >> 24) & 0xff) |
		       ((fconv & 0xff00) << 8) | ((fconv & 0xff0000) >> 8);

    if (fconv) {
      fmant = 0x00ffffff & fconv;
      /* The next two lines were added by Toralf Foerster */
      /* to trap non-IBM format data i.e. conv=0 data  */
      if (fmant == 0)
	printf("mantissa is zero data may not be in IBM FLOAT Format !");
      t = (int) ((0x7f000000 & fconv) >> 22) - 130;
      while (!(fmant & 0x00800000)) { --t; fmant <<= 1; }
      if (t > 254) fconv = (0x80000000 & fconv) | 0x7f7fffff;
      else if (t <= 0) fconv = 0;
      else fconv =   (0x80000000 & fconv) | (t << 23)
	     | (0x007fffff & fmant);
    }
    to[i] = fconv;
  }
  return;
}

/******************* read segy file*************************************/
float **readsu(char *name,int nx,int nt){
  segy Tracehead_vp={0};
  FILE *psegy_vp;
  float **vpb;
  int ix=0,it;
  printf("Read?\n");
  vpb=alloc2float(nt,nx);
  printf("Reading %s...\n",name);

  Tracehead_vp.dt=(unsigned short)(0.001*1000000);
  Tracehead_vp.ns=(unsigned short)nt;
  //printf("Yes!!!\n");
  psegy_vp=fopen(name,"rb");
  //printf("Yes!!!!!!\n");
  //fseek(psegy_vp,3600L,0);
  //printf("Yes!!!!!!\n");
  //for(ix=0;ix<nx;ix++)
  while(ix<nx)
    {
      fseek(psegy_vp,240L,1);
      for(it=0;it<nt;it++)
	{
	  fread(&vpb[ix][it],sizeof(float),1,psegy_vp);
	}
      ix +=1;
    }
  printf("There are %d traces loaded\n",ix);
  fclose(psegy_vp);
  return(vpb);
}
/********************** write segy file*********************************/
void writesu(float **vout,int nx,int nt,float dx,float dt,char *name){
  segy Tracehead_pw={0};
  FILE *psegy_p;
  int ix=0,it;
  printf("Writing %s... sgy\n",name);
  Tracehead_pw.dt=(unsigned short)(dt*1000000);
  Tracehead_pw.ns=(unsigned short)nt;
  psegy_p=fopen(name,"wb");
  //fseek(psegy_p,3600L,0);
  while(ix<nx)
    {
      Tracehead_pw.tracf=ix+1;
      Tracehead_pw.offset=ix*(int)dx;
      fwrite(&Tracehead_pw,sizeof(segy),1,psegy_p);
      for(it=0;it<nt;it++)
	{
	  fwrite(&vout[ix][it],sizeof(float),1,psegy_p);
	}
      ix +=1;
    }
  fclose(psegy_p);
}
void write1dsu(float *out,size_t nx,size_t nt,float dx,float dt,char *name){
  size_t ix,it;
  float **out2d;
  out2d=alloc2float(nt,nx);
  
  for(ix=0;ix<nx;ix++)
    for(it=0;it<nt;it++)
      //      out[ix*nt+it]*=.0;
      out2d[ix][it]=out[ix*nt+it];
  writesu(out2d,nx,nt,dx,dt,name);
  
  free2float(out2d);
}
void write1ddir(float *out,size_t nx,size_t nt,char *name){
  size_t ix,it;
  float **out2d;
  out2d=alloc2float(nt,nx);
  
  for(ix=0;ix<nx;ix++)
    for(it=0;it<nt;it++)
      out2d[ix][it]=out[ix*nt+it];
  writedir(out2d,nx,nt,name);
  
  free2float(out2d);
}
/********************** read segy file*********************************/
float **readsegy(char *name,int nx,int nt)
{
  segy Tracehead_vp={0};
  FILE *psegy_vp;
  float **vpb;
  int ix=0,it;
  printf("Read?\n");
  vpb=alloc2float(nt,nx);
  printf("Reading %s...\n",name);

  Tracehead_vp.dt=(unsigned short)(0.001*1000000);
  Tracehead_vp.ns=(unsigned short)nt;
  psegy_vp=fopen(name,"rb");
  fseek(psegy_vp,3600L,0);
  //for(ix=0;ix<nx;ix++)
  while(ix<nx) {
    fseek(psegy_vp,240L,1);
    for(it=0;it<nt;it++)
      {
	fread(&vpb[ix][it],sizeof(float),1,psegy_vp);
      }
    ix +=1;
  }
  printf("There are %d traces loaded\n",ix);
  fclose(psegy_vp);
  return(vpb);
}

/********************** write segy file*********************************/
void writesegy(float **vout,int nx,int nt,float dx,float dt,char *name)
{
  segy Tracehead_pw={0};
  FILE *psegy_p;
  int ix=0,it;
  printf("Writing %s... sgy\n",name);
  Tracehead_pw.dt=(unsigned short)(dt*1000000);
  Tracehead_pw.ns=(unsigned short)nt;
  psegy_p=fopen(name,"wb");
  fseek(psegy_p,3600L,0);
  while(ix<nx){
    Tracehead_pw.tracf=ix+1;
    Tracehead_pw.offset=ix*(int)dx;
    fwrite(&Tracehead_pw,sizeof(segy),1,psegy_p);
    for(it=0;it<nt;it++)
      {
	fwrite(&vout[ix][it],sizeof(float),1,psegy_p);
      }
    ix +=1;
  }
  fclose(psegy_p);
}

/**********************read dir file************************************/
float **read_dir(char *name,int nx,int nz)
{
  FILE *fp=NULL;
  float **vp;
  int i;
  vp=alloc2float(nz,nx);
  fp=fopen(name,"rb+");
  //  printf("Reading %s...\n",name);
  if(fp!=NULL){
    for(i=0;i<nx;i++)
      fread(vp[i],sizeof(float),nz,fp);
      //printf("ok\n");
    fclose(fp);
  }
  else{
    printf("!!!! CAN FIND %s file!!! EXIT\n", name);
    exit(1);
  }
  return(vp);
}

void read_dir_3d(float ***d, int ny, int nx, int nz, char *name){
  FILE *fp=NULL;
  size_t i, n, iy, ix, iz;

  fp=fopen(name,"rb");
  if(fp!=NULL){
    for(iy=0; iy<ny; iy++)
      for(ix=0; ix<nx; ix++)
	fread(d[iy][ix],sizeof(float),nz,fp);
    fclose(fp);
  }
  else{
    printf("!!!! CAN FIND %s file!!! EXIT\n", name);
    exit(1);
  }
}

float *readdir1d(char *name, size_t nx)
{
  FILE *fp=NULL;
  float *vp;
  int i;
  vp=alloc1float(nx);
  fp=fopen(name,"rb+");
  if(fp!=NULL){
    //    printf("Reading %s...\n",name);
    fread(vp,sizeof(float),nx,fp);
  fclose(fp);
  }
  else{
    printf("!!! can not find %s exit!!!",name);
    exit(1);
  }
  return(vp);
}

void writedir(float **vout,int nx,int nz,char *name){
  FILE *fp=NULL;
  int ix;
  //  printf("Writing %s dir...\n",name);
  fp=fopen(name,"wb");
  for(ix=0;ix<nx;ix++)
    {
      fwrite(vout[ix],sizeof(float),nz,fp);
    }
  fclose(fp);
}
void write_dir(float *out, size_t np, char *name){
  FILE *fp=NULL;
 
  fp=fopen(name,"wb");
  fwrite(out,sizeof(float),np,fp);
  fclose(fp);
}

void init_1d(float *data,size_t nx){
  size_t i;
  for(i=0;i<nx;i++)
    data[i]=0.0;
}

void init_2d(float **data,int nx,int nz){
  int i,j;
  for(i=0;i<nx;i++)
    for(j=0;j<nz;j++)
      data[i][j]=0.0;
}
void init_3d(float ***data,int nx,int nz,int nt){
  int i,j,k;
  for(i=0;i<nx;i++)
    for(j=0;j<nz;j++)
      for(k=0;k<nt;k++)
	data[i][j][k]=0.0;		
}

void init_1d_int(int *data,size_t nx){
  size_t i;
  for(i=0;i<nx;i++)
    data[i]=0;
}
void init_2d_int(int **data,int nx,int nz){
  int i,j;
  for(i=0;i<nx;i++)
    for(j=0;j<nz;j++)
      data[i][j]=0;
}
void init_3d_int(int ***data,int nx,int nz,int nt){
  int i,j,k;
  for(i=0;i<nx;i++)
    for(j=0;j<nz;j++)
      for(k=0;k<nt;k++)
	data[i][j][k]=0;		
}
/*
void init_1d_mklc(MKL_Complex8 *data,size_t nx){
  size_t i;
  for(i=0;i<nx;i++)
    data[i]=mkl_crmul(data[i],0.0);
}

void init_2d_mklc(MKL_Complex8 **data,int nx,int nz){
  int i,j;
  for(i=0;i<nx;i++)
    for(j=0;j<nz;j++)
      data[i][j]=mkl_crmul(data[i][j],0.0);
}
void init_3d_mklc(MKL_Complex8 ***data,int nx,int nz,int nt){
  int i,j,k;
  for(i=0;i<nx;i++)
    for(j=0;j<nz;j++)
      for(k=0;k<nt;k++)
	data[i][j][k]=mkl_crmul(data[i][j][k],0.0);
		
}
*/
void vpad(float **v,float **vvpad,int nx,int nz,int nbd){
  int ix,iz,nbx,nbz;
  nbx=nx+2*nbd;
  nbz=nz+2*nbd;
  for(ix=0;ix<nbx;ix++)
    for(iz=0;iz<nbz;iz++)
      if(ix>=nbd&&ix<(nx+nbd)&&iz>=nbd&&iz<(nz+nbd))
	vvpad[ix][iz]=v[ix-nbd][iz-nbd];		
  for(ix=0;ix<nbx;ix++){
    for(iz=0;iz<nbz;iz++){
      if(ix<nbd)
	vvpad[ix][iz]=vvpad[nbd][iz];
      if(ix>=nbd+nx)
	vvpad[ix][iz]=vvpad[nbd+nx-1][iz];		
    }
  }
  for(ix=0;ix<nbx;ix++){
    for(iz=0;iz<nbz;iz++){
      if(iz<nbd)
	vvpad[ix][iz]=vvpad[ix][nbd];
      if(iz>=nz+nbd)
	vvpad[ix][iz]=vvpad[ix][nbd+nz-1];		
    }
  }
}

void vpad_3d(float ***v, float ***vvpad, 
	     int ny, int nx, int nz, int nbd){
  int iy, ix, iz, nby, nbx, nbz;
  nby=ny+2*nbd;
  nbx=nx+2*nbd;
  nbz=nz+2*nbd;

  for (iy=0; iy<ny; iy++)
    for (ix=0; ix<nx; ix++)
      for (iz=0; iz<nz; iz++)
	vvpad[iy+nbd][ix+nbd][iz+nbd]=v[iy][ix][iz];

  for (iy=0; iy<nby; iy++)
    for (ix=0; ix<nbx; ix++)
      for (iz=0; iz<nbd; iz++){
	vvpad[iy][ix][iz]=vvpad[iy][ix][nbd];
	vvpad[iy][ix][nbz-1-iz]=vvpad[iy][ix][nbz-1-nbd];
      }
  for (iy=0; iy<nby; iy++)
    for (ix=0; ix<nbd; ix++)
      for (iz=0; iz<nbz; iz++){
	vvpad[iy][ix][iz]=vvpad[iy][nbd][iz];
	vvpad[iy][nbx-1-ix][iz]=vvpad[iy][nbx-1-nbd][iz];
      }
  for (iy=0; iy<nbd; iy++)
    for (ix=0; ix<nbx; ix++)
      for (iz=0; iz<nbz; iz++){
	vvpad[iy][ix][iz]=vvpad[nbd][ix][iz];
	vvpad[nby-1-iy][ix][iz]=vvpad[nby-1-nbd][ix][iz];
      }
}

void ricker(float *pulse,int nt,int fre, float dt,float tdelay){
  int it;
  for(it=0;it<nt;it++)
    pulse[it]=(1-2*pi*fre*(it-tdelay)*dt*pi*fre*(it-tdelay)*dt)*exp(-pi*fre*(it-tdelay)*dt*pi*fre*(it-tdelay)*dt);
}
void ricker1(float *pulse,int nt,int fre, float dt){
  int it;
  float tt;
  for(it=0;it<nt;it++){
    tt=pi*fre*(it*dt-1.0/fre);
    tt*=tt;
    pulse[it]=(1-2*tt)*exp(-tt);
  }
}
