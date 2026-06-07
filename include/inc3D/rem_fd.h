#ifndef rem_h
#define rem_h
#include "cu_common.h"

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
             int mytid, char *order);
#endif
