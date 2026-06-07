/*****************************************************

void *alloc1(size_t n1,size_t size)
void *realloc1(void *v,size_t n1,size_t size)
void **alloc2(size_t n1,size_t n2,size_t size)
void ***alloc3(size_t n1,size_t n2,size_t n3,size_t size)
void ****alloc4(size_t n1,size_t n2,size_t n3,size_t n4,size_t size)
void free1 (void *p)
void free2 (void **p)
void free3 (void ***p)
void free4 (void ****p)
int *alloc1int(size_t n1)
void free1int(int *p)
int *realloc1int(int *v, size_t n1)
int **alloc2int(size_t n1, size_t n2)
void free2int(int **p)
int ***alloc3int(size_t n1, size_t n2, size_t n3)
void free3int(int ***p)
float *alloc1float(size_t n1)
float *realloc1float(float *v, size_t n1)
void free1float(float *p)
float **alloc2float(size_t n1, size_t n2)
void free2float(float **p)
float ***alloc3float(size_t n1, size_t n2, size_t n3)
void free3float(float ***p)
void free4float(float ****p)
double *alloc1double(size_t n1)
double *realloc1double(double *v, size_t n1)
void free1double(double *p)
double **alloc2double(size_t n1, size_t n2)
void free2double(double **p)
double ***alloc3double(size_t n1, size_t n2, size_t n3)
void free3double(double ***p)
complex *alloc1complex(size_t n1)
void free1complex(complex *p)
complex **alloc2complex(size_t n1, size_t n2)
float _Complex **alloc2C(size_t n1,size_t n2)
void free2C(float _Complex **p)
void free2complex(complex **p)
complex ***alloc3complex(size_t n1, size_t n2, size_t n3)
void free3complex(complex ***p)
******************************************************/

#include "alloc.h"
//#include <math.h>
//#include "mkl_dfti.h"
/* allocate a 1-d array */
void *alloc1 (size_t n1, size_t size)
{
	void *p;

	if ((p=malloc(n1*size))==NULL)
		return NULL;
	return p;
}

/* re-allocate a 1-d array */
void *realloc1(void *v, size_t n1, size_t size)
{
	void *p;

	if ((p=realloc(v,n1*size))==NULL)
		return NULL;
	return p;
}

/* free a 1-d array */
void free1 (void *p)
{
	free(p);
}

/* allocate a 2-d array */
void **alloc2 (size_t n1, size_t n2, size_t size)
{
	size_t i2;
	void **p;

	if ((p=(void**)malloc(n2*sizeof(void*)))==NULL) 
		return NULL;
	if ((p[0]=(void*)malloc(n2*n1*size))==NULL) {
		free(p);
		return NULL;
	}
	for (i2=0; i2<n2; i2++)
		p[i2] = (char*)p[0]+size*n1*i2;
	return p;
}

/* free a 2-d array */
void free2 (void **p)
{
	free(p[0]);
	free(p);
}

/* allocate a 3-d array */
void ***alloc3 (size_t n1, size_t n2, size_t n3, size_t size)
{
	size_t i3,i2;
	size_t temp;
	void ***p;
	temp=n3*n2*n1*size*1L;
//	prsize_tf("n3=%d n2=%d n1=%d size=%d temp=%ld\n",n3,n2,n1,size,temp);
	if ((p=(void***)malloc(n3*sizeof(void**)))==NULL){
		return NULL;
		}
	if ((p[0]=(void**)malloc(n3*n2*sizeof(void*)))==NULL) {
		free(p);
		return NULL;
	}
	if ((p[0][0]=(void*)malloc(temp))==NULL) {
		free(p[0]);
		free(p);
		printf("this temp= %ld is tooooooooo large!!!\n",temp);
		return NULL;
	}

	for (i3=0; i3<n3; i3++) {
		p[i3] = p[0]+n2*i3;
		for (i2=0; i2<n2; i2++){
			temp=size*n1*(i2+n2*i3);
			p[i3][i2] = (char*)p[0][0]+temp;
			}
	}
	return p;
}

/* free a 3-d array */
void free3 (void ***p)
{
	free(p[0][0]);
	free(p[0]);
	free(p);
}

/* allocate a 4-d array */
void ****alloc4 (size_t n1, size_t n2, size_t n3, size_t n4, size_t size)
{
	size_t i4,i3,i2;
	void ****p;

	if ((p=(void****)malloc(n4*sizeof(void***)))==NULL)
		return NULL;
	if ((p[0]=(void***)malloc(n4*n3*sizeof(void**)))==NULL) {
		free(p);
		return NULL;
	}
	if ((p[0][0]=(void**)malloc(n4*n3*n2*sizeof(void*)))==NULL) {
		free(p[0]);
		free(p);
		return NULL;
	}
	if ((p[0][0][0]=(void*)malloc(n4*n3*n2*n1*size))==NULL) {
		free(p[0][0]);
		free(p[0]);
		free(p);
		return NULL;
	}
	for (i4=0; i4<n4; i4++) {
		p[i4] = p[0]+i4*n3;
		for (i3=0; i3<n3; i3++) {
			p[i4][i3] = p[0][0]+n2*(i3+n3*i4);
			for (i2=0; i2<n2; i2++)
				p[i4][i3][i2] = (char*)p[0][0][0]+
						size*n1*(i2+n2*(i3+n3*i4));
		}
	}
	return p;
}
/* free a 4-d array */
void free4 (void ****p)
{
	free(p[0][0][0]);
	free(p[0][0]);
	free(p[0]);
	free(p);
}

/* allocate a 1-d array of ints */
int *alloc1int(size_t n1)
{
	return (int*)alloc1(n1,sizeof(int));
}

/* re-allocate a 1-d array of ints */
int *realloc1int(int *v, size_t n1)
{
	return (int*)realloc1(v,n1,sizeof(int));
}

/* free a 1-d array of ints */
void free1int(int *p)
{
	free1(p);
}

/* allocate a 2-d array of ints */
int **alloc2int(size_t n1, size_t n2)
{
	return (int**)alloc2(n1,n2,sizeof(int));
}

/* free a 2-d array of ints */
void free2int(int **p)
{
	free2((void**)p);
}

/* allocate a 3-d array of ints */
int ***alloc3int(size_t n1, size_t n2, size_t n3)
{
	return (int***)alloc3(n1,n2,n3,sizeof(int));
}

/* free a 3-d array of ints */
void free3int(int ***p)
{
	free3((void***)p);
}

/* allocate a 1-d array of floats */
float *alloc1float(size_t n1)
{
	return (float*)alloc1(n1,sizeof(float));
}

/* re-allocate a 1-d array of floats */
float *realloc1float(float *v, size_t n1)
{
	return (float*)realloc1(v,n1,sizeof(float));
}

/* free a 1-d array of floats */
void free1float(float *p)
{
	free1(p);
}

/* allocate a 2-d array of floats */
float **alloc2float(size_t n1, size_t n2)
{
	return (float**)alloc2(n1,n2,sizeof(float));
}

/* free a 2-d array of floats */
void free2float(float **p)
{
	free2((void**)p);
}

/* allocate a 3-d array of floats */
float ***alloc3float(size_t n1, size_t n2, size_t n3)
{
	return (float***)alloc3(n1,n2,n3,sizeof(float));
}

/* free a 3-d array of floats */
void free3float(float ***p)
{
	free3((void***)p);
}
/* allocate a 4-d array of floats, added by Zhaobo Meng, 1997 */
float ****alloc4float(size_t n1, size_t n2, size_t n3, size_t n4)
{
        return (float****)alloc4(n1,n2,n3,n4,sizeof(float));
}

/* free a 4-d array of floats, added by Zhaobo Meng, 1997 */
void free4float(float ****p)
{
        free4((void****)p);
}
/* allocate a 1-d array of doubles */
double *alloc1double(size_t n1)
{
	return (double*)alloc1(n1,sizeof(double));
}

/* re-allocate a 1-d array of doubles */
double *realloc1double(double *v, size_t n1)
{
	return (double*)realloc1(v,n1,sizeof(double));
}


/* free a 1-d array of doubles */
void free1double(double *p)
{
	free1(p);
}

/* allocate a 2-d array of doubles */
double **alloc2double(size_t n1, size_t n2)
{
	return (double**)alloc2(n1,n2,sizeof(double));
}

/* free a 2-d array of doubles */
void free2double(double **p)
{
	free2((void**)p);
}

/* allocate a 3-d array of doubles */
double ***alloc3double(size_t n1, size_t n2, size_t n3)
{
	return (double***)alloc3(n1,n2,n3,sizeof(double));
}

/* free a 3-d array of doubles */
void free3double(double ***p)
{
	free3((void***)p);
}

/* allocate a 1-d array of complexs */
//double complex *alloc1complex(size_t n1)
//{
//	return (double complex*)alloc1(n1,sizeof(double complex));
//}

/* re-allocate a 1-d array of complexs */
//double complex *realloc1complex(double complex *v, size_t n1)
//{
//	return (double complex*)realloc1(v,n1,sizeof(double complex));
//}

/* free a 1-d array of complexs */
//void free1complex(double complex *p)
//{
//	free1(p);
//}

/* allocate a 2-d array of complexs */
//double complex **alloc2complex(size_t n1, size_t n2)
//{
//	return (double complex**)alloc2(n1,n2,sizeof(double complex));
//}
/* allocate a 1-d MKLC8 */
MKL_Complex8 *alloc1MKLC(size_t n1){
  return (MKL_Complex8*)alloc1(n1,sizeof(MKL_Complex8));
}
void free1MKLC(MKL_Complex8 *p){
  free1((void*)p);
}

/* allocate a 2-d MKLC8*/
MKL_Complex8 **alloc2MKLC(size_t n1,size_t n2){
  return (MKL_Complex8 **)alloc2(n1,n2,sizeof(MKL_Complex8));
}

MKL_Complex16 **alloc2MKLC16(size_t n1,size_t n2){
  return (MKL_Complex16 **)alloc2(n1,n2,sizeof(MKL_Complex16));
}

void free2MKLC(MKL_Complex8 **p){
  free2((void**)p);
}

/* allocate a 3-d MKLC8*/
MKL_Complex8 ***alloc3MKLC(size_t n1,size_t n2,size_t n3){
  return (MKL_Complex8 ***)alloc3(n1,n2,n3,sizeof(MKL_Complex8));
}
void free3MKLC(MKL_Complex8 ***p){
  free3((void***)p);
}

/* allocate a 1-d float _Complex array*/
float _Complex *alloc1C(size_t n1){
  return (float _Complex*)alloc1(n1,sizeof(float _Complex));
}
void free1C(float _Complex *p){
  free1((void*)p);
}

float _Complex **alloc2C(size_t n1,size_t n2){
  return (float _Complex**)alloc2(n1,n2,sizeof(float _Complex));
}
void free2C(float _Complex **p){
  free2((void**)p);
}
/* free a 2-d array of complexs */
void free2complex(double complex **p)
{
	free2((void**)p);
}

/* allocate a 3-d array of complexs */
double complex ***alloc3complex(size_t n1, size_t n2, size_t n3)
{
	return (double complex***)alloc3(n1,n2,n3,sizeof(double complex));
}

/* free a 3-d array of complexs */
void free3complex(double complex ***p)
{
	free3((void***)p);
}

/* allocate a 4-d array of complexs */
double complex ****alloc4complex(size_t n1, size_t n2, size_t n3, size_t n4)
{
  return (double complex****)alloc4(n1,n2,n3,n4,sizeof(double complex));
}

/* free a 3-d array of complexs */
void free4complex(double complex ****p)
{
	free4((void****)p);
}
