SRCS =	alloc.c utility_zz.c abc.c acqui.c lint.c 
SRCUS = main.cu optimization_cuda.cu rem_fd.cu single_solver.cu reduce.cu

OBJS=$(SRCS:.c=.o)
OBJCUS=$(SRCUS:.cu=.o)

# excutable
BIN=../bin
DEST=$(BIN)/cuda_3D_FM

NV=nvcc
CC=mpicc
CFLAGS=-O3 -g
# NEED TO CHANGED TO YOUR MKL AND MPI PATH
MKLROOT=/home/opt/intel2020u4/compilers_and_libraries_2020.4.304/linux/mkl
MPI_HOME=/home/opt/intel2020u4/compilers_and_libraries_2020.4.304/linux/mpi/intel64

INCS=-I../include/inc3D
INCMKL= -I$(MKLROOT)/include
INCMPI= -I$(MPI_HOME)/include

LIBMKL= -L$(MKLROOT)/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl #-std=c++11
LIBMPI= -L$(MPI_HOME)/lib/release -lmpi

test: $(DEST)
	@echo @@@@@@@@@@@@@@@@@@@@
	@echo FINISHED COMPILING
	@echo @@@@@@@@@@@@@@@@@@@@

$(DEST):$(OBJS) $(OBJCUS)
	@which gcc
	$(NV) $(CFLAGS) $(LIBMKL) $(LIBMPI) -o $(DEST) $(OBJS) $(OBJCUS)

%.o: %.c
	$(CC) $(CFLAGS) $(INCS) $(INCMKL) $(INCMPI)  -c $< -o $@ 

%.o: %.cu
	$(NV) $(INCS) $(INCMKL) $(INCMPI) -c $< -o $@

clean:
	rm $(OBJS) $(OBJCUS) $(DEST)
