#ifndef lint_h
#define lint_h
#include "common.h"

void lint2d_inject_bell(float **f, int nbd, int xl,
			float ss, float **bell, int nbell, int *index,
			float w00, float w01, float w10, float w11);

void lint2d_init(size_t na, float **aa,
		 int nx, int nz, float dx, float dz,
		 int **index,
		 float *w00, float *w01, float *w10, float *w11);
/*
void lint2d_inject_src(double **Bx, double **Bz,
		       int npml, int pad1, int its,
		       double complex ww,
		       int *index,
		       float w00, float w01, float w10, float w11);
*/
void lint2d_inject_rec(double **Bx, double **Bz,
		       int npml, int pad1, int its,
		       float *wwr, float *wwi, int ir,
		       int *index,
		       float w00, float w01, float w10, float w11);

void lint2d_inject_rec_rem(float **fu, int nbd, int xl,
			   float d, float **bell, int nbell, int *index,
			   float w00, float w01, float w10, float w11);

void lint2d_inject_rec_rem_2(float **fu, int nbd, int xl,
			    float d, int *index,
			    float w00, float w01, float w10, float w11);

void lint2d_extract(//double complex **uu,
		    double **Xx, double **Xz, 
		    int npml, int pad1, int its,
		    float *ddr, float *ddi, int ir, int *index,
		    float w00, float w01, float w10, float w11);

void lint2d_extract1(float *gradi,  float *hessi, float *vi,
		     int nx, int nz, int nnz,
		     float *gradd, float *hessd, float *vd, int ir, int *index,
		     float w00, float w01, float w10, float w11);

void lint2d_extract2(float **in, int nx, int nz, 
		     float **dout, int ix, int iz, int *index,
		     float w00, float w01, float w10, float w11);

void lint2d_extract3(float **in, int min_all,
		     float *dout, int *index,
		     float w00, float w01, float w10, float w11);

void lint2d_extract_new(float *gradi,  float *hessi,
                        int nx, int nz, int nnz,
                        float *gradd, float *hessd, int ir, int *index,
                        float w00, float w01, float w10, float w11);

void lint3d_init(size_t na, float **aa,
		 int ny, int nx, int nz, float dy, float dx, float dz,
		 int **index,
		 float *w000, float *w001, float *w010, float *w011, 
		 float *w100, float *w101, float *w110, float *w111);

void lint3d_extract(float *dd,  float ***in, size_t na, int **index,
		    int ny, int nx, int nz,
		    float *w000, float *w001, float *w010, float *w011,
		    float *w100, float *w101, float *w110, float *w111);

void lint3d_extract2(float *dd,  float ***in, size_t na,
		     int yl, int xl, int **index, size_t icc,
		     int ny, int nx, int nz, int nbd,
		     float *w000, float *w001, float *w010, float *w011,
		     float *w100, float *w101, float *w110, float *w111);

void lint3d_inject_bell(float ***f, int nbd, int yl, int xl,
			float ss, float ***bell, int nbell, int *index,
			float w000, float w001, float w010, float w011,
			float w100, float w101, float w110, float w111);

void lint3d_inject_rec(float **d, float ***fu, int it, size_t na, int yl, int xl,
                       int **index, size_t icc, int nbd,
                       float *w000, float *w001, float *w010, float *w011,
                       float *w100, float *w101, float *w110, float *w111);
#endif
