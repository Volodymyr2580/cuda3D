#ifndef _single_solver_cuda_h
#define _single_solver_cuda_h
#include "cu_common.h"

__global__ void cuda_fd3d_v_pml(float *p1, float *vy, float *vx, float *vz,
				float _dy2, float _dx2, float _dz2, 
				int n3, int n2, int n1, int npml, float dt,
				float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
				float *mem_dy, float *mem_dx, float *mem_dz);
__global__ void cuda_fd3d_p_pml(float *p0, float *p1, float *vy, float *vx, float *vz,
				float *vyy, float *vxx, float *vzz,
				float *cw2, float _dy2, float _dx2, float _dz2, 
				int n3, int n2, int n1, int npml, float dt, 
				float *ay, float *by, float *ax, float *bx, float *az, float *bz,
				float *mem_dyy, float *mem_dxx, float *mem_dzz);

__global__ void lint3d_inject_bell_gpu(float *d_u, int nbd, int yl, int xl, int it, int snum,
				       float *src, float *d_bell, int nbell,
				       int indexy, int indexx, int indexz,
				       int ny, int nx, int nz,
				       float *d_sw000, float *d_sw001, float *d_sw010, float *d_sw011, 
				       float *d_sw100, float *d_sw101, float *d_sw110, float *d_sw111);

__global__ void lint3d_extract_gpu_zz(float *din, int nbd, int min_all_y, int min_all_x,
				      int it, int nt, float *dout, int *rec0_indx,
				      size_t nr, int ny, int nx, int nz,
				      float *rw000, float *rw001, float *rw010, float *rw011, 
				      float *rw100, float *rw101, float *rw110, float *rw111);
__global__ void lint3d_inject_bell_extract_gpu_zz(float *d_u, int nbd, int yl, int xl,
				      int it, int nt, int snum,
				      float *src, float *d_bell, int nbell,
				      int indexy, int indexx, int indexz,
				      int ny, int nx, int nz,
				      float *d_sw000, float *d_sw001, float *d_sw010, float *d_sw011,
				      float *d_sw100, float *d_sw101, float *d_sw110, float *d_sw111,
				      float *dout, int *rec0_indx, size_t nr,
				      float *rw000, float *rw001, float *rw010, float *rw011,
				      float *rw100, float *rw101, float *rw110, float *rw111);
//__global__ void check_kernal(float *pu, size_t num);
__global__ void check_kernal(float *pu, size_t num);
__global__ void cuda_p_extract_3d(float *p0, float *pu, size_t itc,
				  size_t n3, size_t n2, size_t n1, int npml, int yl, int xl);

__global__ void cuda_get_misfit_l2(float *d_obs, float *d_est, float *d_wb, float *d_adj,
				   float *tmut, float *bmut, int nr, int nt,
				   float sht_scl, int snum);

__global__ void cuda_vector_mult(float *d1, float *d2, size_t n, float *out);

__global__ void lint3d_inject_rec_gpu(float *d_u, int nbd, int yl, int xl, int it, int nt,
				      float *d_d, int *rec0_indx, int nr,
				      int ny, int nx, int nz,
				      float *rw000, float *rw001, float *rw010, float *rw011, 
				      float *rw100, float *rw101, float *rw110, float *rw111);


__global__ void cuda_fd3d_v_pml_ns(float *p1, float *vy, float *vx, float *vz,
				float _dy2, float _dx2, float _dz2, 
				int n3, int n2, int n1, int npml, float dt,
				float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
				float *mem_dy, float *mem_dx, float *mem_dz);

__global__ void cuda_fd3d_p_pml_ns(float *p0, float *p1, float *vy, float *vx, float *vz,
				float *cw2, float _dy2, float _dx2, float _dz2, 
				int n3, int n2, int n1, int npml, float dt,
				float *ay, float *by, float *ax, float *bx, float *az, float *bz,
				float *mem_dyy, float *mem_dxx, float *mem_dzz);

__global__ void cuda_fd3d_p_pml_shared_ns(float *p0, float *p1, float *vy, float *vx, float *vz,
				float *cw2, float _dy2, float _dx2, float _dz2,
				int n3, int n2, int n1, int npml, float dt,
				float *ay, float *by, float *ax, float *bx, float *az, float *bz,
				float *mem_dyy, float *mem_dxx, float *mem_dzz);

__global__ void cuda_fd3d_p_core_ns(float *p0, float *p1, float *cw2,
				float _dy2, float _dx2, float _dz2,
				int n3, int n2, int n1, int npml, float dt);

__global__ void cuda_fd3d_v_pml2(float *p1, float *vy, float *vx, float *vz,
				float _dy2, float _dx2, float _dz2, 
				int n3, int n2, int n1, int npml, float dt,
				float *ay_h, float *by_h, float *ax_h, float *bx_h, float *az_h, float *bz_h,
				float *mem_dy, float *mem_dx, float *mem_dz);

__global__ void cuda_fd3d_p_pml2(float *p0, float *p1, float *vy, float *vx, float *vz,
				float *vyy, float *vxx, float *vzz,
				float *cw2, float _dy2, float _dx2, float _dz2, 
				int n3, int n2, int n1, int npml, float dt, 
				float *ay, float *by, float *ax, float *bx, float *az, float *bz,
				float *mem_dyy, float *mem_dxx, float *mem_dzz);
#endif
