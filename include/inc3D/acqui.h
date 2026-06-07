#ifndef acqui_h
#define acqui_h
#include "common.h"

void read_acqui_shot(char *navfile, size_t ntr, int nx, int *ns, int *flag);
void read_acqui_shot_3d(char *navfile, size_t ntr, int ny, int nx, int *ns, int *flag);

void read_acqui_rec(char *navfile, int *rec_count, int *nrmax, size_t ntr);

void read_acqui(char *navfile, size_t ntr, int ntrpad, 
		int ns, int nx, float dx, int *nrec_shot,
		float **src_cor, float **rec_cor,
		int *src_x, int **rec_x, int *flag);

void read_acqui_3d(char *navfile, size_t ntr, int ntrpad, int ns,
		   int *nrec_shot, float **src_cor, float **rec_cor, int *flag);
#endif
