#ifndef COMMON_H
#define COMMON_H
#include "mkl_common.h"
#endif

#ifndef READINPUT
#define READINPUT

//#include "mkl_dfti.h"
void readpar(int mytid,int root,char *seisname,char *seisout,char *directoryin,
	     float referpoint,int pspoflag,char *vfile,int vflag,int npml,int nx,int nz,float dx,float dz,
	     int ffps,int llps,int ffpo,int llpo,int dnps,int dnpo,float fps,float fpo,float dps,float dpo,int nps,int npo,float dplimit,
	     float of,float df,int nf,int storeflag,int iitermax,float crit,int iterflag,char *directory,MPI_Comm comm);
void raysetup(int ntraces,int *npsarry,float dplimit,int nps,int npo,float fps,float fpo,float dps,float dpo,
	      int nnps,int nnpo,int ffps,int ffpo,int dnps,int dnpo,float *psin,float *poin,int pspoflag,
	      float *psarryin);
#endif
