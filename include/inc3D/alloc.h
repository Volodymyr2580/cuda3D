//#ifndef MKL_COMMON_H
//#define MKL_COMMON_H
//#include "mkl_common.h"
//#endif

#ifndef alloc_h
#define alloc_h

#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
//#include "common.h"

#ifndef MKL_COMPAT_TYPES_H
#define MKL_COMPAT_TYPES_H
typedef struct { float real, imag; } MKL_Complex8;
typedef struct { double real, imag; } MKL_Complex16;
#endif

void *alloc1(size_t n1,size_t size);
void *realloc1(void *v,size_t n1,size_t size);
void **alloc2(size_t n1,size_t n2,size_t size);
void ***alloc3(size_t n1,size_t n2,size_t n3,size_t size);
void ****alloc4 (size_t n1, size_t n2, size_t n3, size_t n4, size_t size);
void free1 (void *p);
void free2 (void **p);
void free3 (void ***p);
void free4 (void ****p);

int *alloc1int(size_t n1);
void free1int(int *p);
int *realloc1int(int *v, size_t n1);

int **alloc2int(size_t n1, size_t n2);
void free2int(int **p);

int ***alloc3int(size_t n1, size_t n2, size_t n3);
void free3int(int ***p);

float *alloc1float(size_t n1);
float *realloc1float(float *v, size_t n1);
void free1float(float *p);

float **alloc2float(size_t n1, size_t n2);
void free2float(float **p);

float ***alloc3float(size_t n1, size_t n2, size_t n3);
void free3float(float ***p);

float ****alloc4float(size_t n1, size_t n2, size_t n3, size_t n4);
void free4float(float ****p);

double *alloc1double(size_t n1);
double *realloc1double(double *v, size_t n1);
void free1double(double *p);

double **alloc2double(size_t n1, size_t n2);
void free2double(double **p);

double ***alloc3double(size_t n1, size_t n2, size_t n3);
void free3double(double ***p);
/*
double complex *alloc1complex(size_t n1);

void free1complex(double complex *p);

double complex **alloc2complex(size_t n1, size_t n2);

void free2complex(double complex **p);

double complex ***alloc3complex(size_t n1, size_t n2, size_t n3);

void free3complex(double complex ***p);

double complex ****alloc4complex(size_t n1, size_t n2, size_t n3, size_t n4);

void free4complex(double complex ****p);
*/
float _Complex **alloc2C(size_t n1,size_t n2);
void free2C(float _Complex **p);
float _Complex *alloc1C(size_t n1);
void free1C(float _Complex *p);

MKL_Complex8 *alloc1MKLC(size_t n1);
void free1MKLC(MKL_Complex8 *p);
MKL_Complex8 **alloc2MKLC(size_t n1,size_t n2);
void free2MKLC(MKL_Complex8 **p);
MKL_Complex8 ***alloc3MKLC(size_t n1,size_t n2,size_t n3);
void free3MKLC(MKL_Complex8 ***p);

MKL_Complex16 **alloc2MKLC16(size_t n1,size_t n2);

#endif
