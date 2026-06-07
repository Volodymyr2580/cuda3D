#ifndef optimization_cuda_h
#define optimization_cuda_h
#include "cu_common.h"

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
		     int ntids, int mytid, int root, MPI_Comm comm);
#endif
