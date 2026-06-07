#ifndef abc_h
#define abc_h
#include "common.h"

void init_cpml(float *ax,float *bx,float *az,float *bz,int nx,int nz,float dx,float dz,int nbd,float dt,float vmax);


void init_cpml_sg(float *ax,float *bx,float *az,float *bz,
		  float *ax_h, float *bx_h, float *az_h, float *bz_h,
		  int nx,int nz,float dx,float dz,int nbd,float dt,float vmax);

void init_bc(float *bt,float *bb,float *bl,float *br,int nbd,float decay);
void bc(float **field,int nx,int nz,int nbd,float *bt,float *bb,float *bl,float *br);

void bc_3d(float ***d, int ny, int nx, int nz, int nbd,
	   float *bt, float *bb, float *bl, float *br);

void bc_mklc(MKL_Complex8 **field,int nx,int nz,int nbd,float *bt,float *bb,float *bl,float *br);

void init_cpml_sg_3d(float *ay, float *by, float *ax, float *bx, float *az,float *bz,
		     float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
		     int ny, int nx, int nz, float dy, float dx, float dz, int nbd, float dt, float vmax);
#endif
