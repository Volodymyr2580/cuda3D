#include "acqui.h"


void read_acqui_shot(char *navfile, size_t ntr, int nx, int *ns, int *flag){
  size_t ic,iscount;
  float *nav;
  int nc;

  nc=6;
  printf("Read NAV file to get ns\n");
  nav=readdir1d(navfile,ntr*nc);
  // check ns number
  iscount=1;
  for(ic=0;ic<ntr-1;ic++)
    if(nav[nc*ic]!=nav[nc*(ic+1)])
      iscount++;

  *ns=iscount;
  if(*ns>nx){
    *flag=1;
    *flag=0;  /// addd 5/6/2019 to make ns > nx acceptable for interpolation
    printf("NOTE!! ns(%d) > nx(%d), check nav file or input model\n", *ns, nx);
    return;
  }
  else
    *flag=0;
  free1float(nav);
}


void read_acqui_shot_3d(char *navfile, size_t ntr, int ny, int nx, int *ns, int *flag){
  size_t ic,iscount;
  float *nav;
  int nc;

  nc=6;
  printf("Read NAV file to get ns\n");
  nav=readdir1d(navfile,ntr*nc);
  // check ns number
  iscount=1;
  for(ic=0;ic<ntr-1;ic++)
    if(nav[nc*ic]!=nav[nc*(ic+1)] || nav[nc*ic+1]!=nav[nc*(ic+1)+1])
      iscount++;

  *ns=iscount;
  if(*ns>nx*ny){
    *flag=1;
    *flag=0;  /// addd 5/6/2019 to make ns > nx acceptable for interpolation
    printf("NOTE!! ns(%d) > nxy(%d), check nav file or input model\n", *ns, nx*ny);
    return;
  }
  else
    *flag=0;
  free1float(nav);
}

void read_acqui_rec(char *navfile, int *rec_count, int *nrmax, size_t ntr){

 // count rec number for each shot
  int iscount, ircount;
  size_t ic,tmp;
  float *nav;

  nav=readdir1d(navfile,ntr*4);
  
  iscount=1;
  ircount=1;
  tmp=0;
  for(ic=0;ic<ntr-1;ic++){
    if(nav[4*ic]==nav[4*(ic+1)]){
      ircount++;
      if(ic==ntr-2)
	rec_count[iscount-1]=ircount;
    }
    else{
      rec_count[iscount-1]=ircount;
      if(ircount>=tmp)
	tmp=ircount;
      iscount++;
      ircount=1;
    }
  }
  *nrmax=tmp;
  free1float(nav);
}

void read_acqui(char *navfile, size_t ntr, int ntrpad,
		int ns, int nx, float dx, int *nrec_shot,
		// float *src_cor, float *rec_cor,
		float **src_cor, float **rec_cor,
		int *src_x, int **rec_x, int *flag){

  int iscount,ircount, nrmax;
  int *rec_count;
  size_t ic,itemp,ii;
  float *nav;
  int nc;

  nc=6;
  printf("Read NAV file for src, and rec locations\n");

  nav=readdir1d(navfile,ntr*nc);

  // assign cordinate 
  //  for(ic=0;ic<ntrpad;ic++){
  //    src_cor[ic]=.0;
  //    rec_cor[ic]=.0;
  //  }
 
  //  for(ic=0;ic<ntr;ic++){
  //    src_cor[ic]=nav[4*ic];
  //    rec_cor[ic]=nav[4*ic+2];
  //  }

  // count rec number for each shot
  //  rec_count=alloc1int(ns);

  iscount=1;
  ircount=1;
  nrmax=0;
  for(ic=0;ic<ntr-1;ic++){
    if(nav[nc*ic]==nav[nc*(ic+1)]){
      ircount++;
      if(ic==ntr-2)
	nrec_shot[iscount-1]=ircount;
    }
    else{
      nrec_shot[iscount-1]=ircount;
      if(ircount>nrmax)
	nrmax=ircount;
      iscount++;
      ircount=1;
      // new 04.17.2019
      if(ic==ntr-2)
	nrec_shot[iscount-1]=ircount;
    }
  }
  printf("ns=%d iscount=%d\n",ns,iscount);
  if(iscount!=ns){
    printf("src count wrong!\n");
    exit(0);
  }
  if(nrmax>nx){
    //    *flag=1;
    printf("NOTE!! nrmax (%d) > nx (%d), check nav file or input model\n", nrmax, nx);
    //    printf("reciever arry is out of nx !!!!! cause problem when truncating domain, check input nav or nx!\n");
    //    return;
  }

  //find x location for src and rec
  size_t icc, iss;
  ic=0;
  icc=0;
  for(iscount=0;iscount<ns;iscount++){
    itemp=nrec_shot[iscount];
    icc+=itemp; // 04.17
    src_cor[iscount][0]=roundf(nav[nc*ic]*10000)/10000; // 04.17
    src_cor[iscount][1]=roundf(nav[nc*ic+2]*10000)/10000; // 04.17
    //    src_cor[iscount][2]=roundf(nav[nc*ic+2]*10000)/10000; // 04.17
    //    printf("shot %d nr=%d\n", iscount, itemp);
    src_x[iscount]=(int)((nav[nc*ic]+0.00001)/dx); 
    for(ircount=0;ircount<itemp;ircount++){
      //      ii=(int)((nav[nc*ic+3]+0.00001)/dx);
      //      printf("shot %d ir=%d, ii=%d\n", iscount, ircount, ii);
      //      rec_x[iscount][ircount]=ii;
      //      rec_x[iscount][ii]=1;

      rec_cor[ic][0]=roundf(nav[nc*ic+3]*10000)/10000; // 04.17
      rec_cor[ic][1]=roundf(nav[nc*ic+5]*10000)/10000; // 04.17

      ic++;
    }
  }

  if(ic!=ntr){
    printf("ic %zu != ntr %zu!!!!! exit\n", ic, ntr);
    exit(1);
  }
  else
    printf("NAV loaded ntr=%zu, ntr_input=%zu\n", ic, ntr);

  ////
  FILE *ff=NULL;
  ff=fopen("s_x_new.txt","w");
  for(ic=0;ic<ns;ic++)
    fprintf(ff,"shot %zu loc x =%f z=%f\n",ic+1,src_cor[ic][0],src_cor[ic][1]);
    //    fprintf(ff,"shot %d loc =%d\n",ic+1,src_x[ic]);
  fclose(ff);

  ff=fopen("nrec_shot_new.txt","w");
  for(ic=0;ic<ns;ic++)
    fprintf(ff,"shot %zu nrec =%d \n",ic+1,nrec_shot[ic]);
    //    fprintf(ff,"shot %d loc =%d\n",ic+1,src_x[ic]);
  fclose(ff);

  ff=fopen("r_x_new.txt","w");
  icc=0;
  for(ic=0;ic<ns;ic++){
    itemp=nrec_shot[ic];
    for(ii=0;ii<itemp;ii++){
      fprintf(ff,"shot %zu  rec loc x =%f z=%f\n",
	      ic+1,rec_cor[icc][0],rec_cor[icc][1]);
      icc++;
    }
      //      fprintf(ff,"shot %d  rec loc =%d\n",ic,rec_x[ic][ii]);
  }
  fclose(ff);

  if(icc!=ntr){
    printf("rec count wrong!!!!\n");
    exit(0);
  }
  ////
  /*  
  float temp;
  int iiii, i, j;
  for(i=0;i<ns;i++){
    for(j=i+1;j<ns;j++){
      temp=src_x[i]-src_x[j];
      if(fabsf(temp)<1e-4){
	printf("Found duplicate shot at nx %d for shot %d.\n Check navfile!!!!\n",src_x[i],i);
	exit(1);
      }
    }
  }
  */
  free1float(nav);
  //  free1int(rec_count);
}

void read_acqui_3d(char *navfile, size_t ntr, int ntrpad, int ns,
		   int *nrec_shot, float **src_cor, float **rec_cor, int *flag){

  //  int iscount,ircount, nrmax;
  int *rec_count;
  size_t ic,itemp,ii, iscount, ircount, nrmax;
  float *nav;
  int nc;

  nc=6;
  printf("Read NAV file for src, and rec locations\n");

  nav=readdir1d(navfile,ntr*nc);

  iscount=1;
  ircount=1;
  nrmax=0;
  for(ic=0;ic<ntr-1;ic++){ // NOTE, this check only works for sx changes the fast
    if(nav[nc*ic]==nav[nc*(ic+1)]){
      ircount++;
      if(ic==ntr-2)
	nrec_shot[iscount-1]=ircount;
    }
    else{
      nrec_shot[iscount-1]=ircount;
      if(ircount>nrmax)
	nrmax=ircount;
      iscount++;
      ircount=1;
      // new 04.17.2019
      if(ic==ntr-2)
	nrec_shot[iscount-1]=ircount;
    }
  }
  printf("ns=%d iscount=%zu\n",ns,iscount);
  if(iscount!=ns){
    printf("src count wrong!\n");
    exit(0);
  }

  //find x location for src and rec
  size_t icc, iss;
  ic=0;
  icc=0;
  for(iscount=0;iscount<ns;iscount++){
    itemp=nrec_shot[iscount];
    icc+=itemp; // 04.17
    src_cor[iscount][0]=roundf(nav[nc*ic+1]*10000)/10000; // cor y
    src_cor[iscount][1]=roundf(nav[nc*ic]*10000)/10000; // cor x
    src_cor[iscount][2]=roundf(nav[nc*ic+2]*10000)/10000; // cor z
    for(ircount=0;ircount<itemp;ircount++){
      rec_cor[ic][0]=roundf(nav[nc*ic+4]*10000)/10000; // cor y
      rec_cor[ic][1]=roundf(nav[nc*ic+3]*10000)/10000; // cor x
      rec_cor[ic][2]=roundf(nav[nc*ic+5]*10000)/10000; // cor z
      ic++;
    }
  }

  if(ic!=ntr){
    printf("ic %zu != ntr %zu!!!!! exit\n", ic, ntr);
    exit(1);
  }
  else
    printf("NAV loaded ntr=%zu, ntr_input=%zu\n", ic, ntr);

  ////
  FILE *ff=NULL;
  ff=fopen("s_cor_new.txt","w");
  for(ic=0;ic<ns;ic++)
    fprintf(ff,"shot %zu loc y= %f x =%f z=%f\n",ic+1,src_cor[ic][0], src_cor[ic][1], src_cor[ic][2]);
  fclose(ff);

  ff=fopen("nrec_shot_new.txt","w");
  for(ic=0;ic<ns;ic++)
    fprintf(ff,"shot %zu nrec =%d \n",ic+1,nrec_shot[ic]);
  fclose(ff);

  ff=fopen("r_cor_new.txt","w");
  icc=0;
  for(ic=0;ic<ns;ic++){
    itemp=nrec_shot[ic];
    for(ii=0;ii<itemp;ii++){
      fprintf(ff,"shot %zu  rec loc y=%f x =%f z=%f\n",
	      ic+1,rec_cor[icc][0],rec_cor[icc][1], rec_cor[icc][2]);
      icc++;
    }
  }
  fclose(ff);

  if(icc!=ntr){
    printf("rec count wrong!!!!\n");
    exit(0);
  }
  ////
  /*  
  float temp;
  int iiii, i, j;
  for(i=0;i<ns;i++){
    for(j=i+1;j<ns;j++){
      temp=src_x[i]-src_x[j];
      if(fabsf(temp)<1e-4){
	printf("Found duplicate shot at nx %d for shot %d.\n Check navfile!!!!\n",src_x[i],i);
	exit(1);
      }
    }
  }
  */
  free1float(nav);
  //  free1int(rec_count);
}
