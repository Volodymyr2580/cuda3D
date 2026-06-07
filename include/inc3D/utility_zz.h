#ifndef zz_utility_h
#define zz_utility_h
#include "common.h"

void ibm_to_float(int from[], int to[], int n, int endian);
float **readsu(char *name,int nx,int nt);
void writesu(float **vout,int nx,int nt,float dx,float dt,char *name);
void write1dsu(float *out,size_t nx,size_t nt,float dx,float dt,char *name);
void write1ddir(float *out,size_t nx,size_t nt,char *name);
void write_dir(float *out, size_t np, char *name);
float **readsegy(char *name,int nx,int nt);
void writesegy(float **vout,int nx,int nt,float dx,float dt,char *name);
float **read_dir(char *name,int nx,int nz);
float *readdir1d(char *name, size_t nx);

void read_dir_3d(float ***d, int ny, int nx, int nz, char *name);

void writedir(float **vout,int nx,int nz,char *name);

void init_1d(float *data,size_t nx);
void init_2d(float **data,int nx,int nz);
void init_3d(float ***data,int nx,int nz,int nt);
void init_1d_int(int *data,size_t nx);
void init_2d_int(int **data,int nx,int nz);
void init_3d_int(int ***data,int nx,int nz,int nt);

void init_1d_mklc(MKL_Complex8 *data,size_t nx);
void init_2d_mklc(MKL_Complex8 **data,int nx,int nz);
void init_3d_mklc(MKL_Complex8 ***data,int nx,int nz,int nt);

void vpad(float **v,float **vvpad,int nx,int nz,int nbd);
void vpad_3d(float ***v, float ***vvpad, 
	     int ny, int nx, int nz, int nbd);

void ricker(float *pulse,int nt,int fre, float dt,float tdelay);
void ricker1(float *pulse,int nt,int fre, float dt);
#endif
