#ifndef common_h
#define common_h
#include <math.h>
#include <time.h>
#include <malloc.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <errno.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#include <float.h>
#include <unistd.h>
#include <complex.h>
#include <stdbool.h>
#include <mpi.h>

//#ifndef _OPENMP
//#include <omp.h>
//#endif

// common
#include "susgy.h"
#include "alloc.h"
#include "utility_zz.h"

// program related
#include "abc.h"
#include "acqui.h"
#include "lint.h"


//#define UNUSED(x) (void)(x)
#define NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))
#define	MAX(x,y) ((x) > (y) ? (x) : (y))
#define	MIN(x,y) ((x) < (y) ? (x) : (y))
#define pi (3.141592653589793)

#ifndef ABS
#define ABS(x) ((x) < 0 ? -(x) : (x))
#endif

#ifdef UNUSED
#elif defined(__GNUC__)
# define UNUSED(x) UNUSED_ ## x __attribute__((unused))
#elif defined(__LCLINT__)
# define UNUSED(x) x
#else
# define UNUSED(x) x
#endif

//void dcc_mon_siginfo_handler(int UNUSED(whatsig))

#endif
