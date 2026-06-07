3 directories
inc3D: all headers
src: all source files, including .c and .cuda files
bin: after make, the excutable is deposite here.

to compile, need mpi, and mkl lib; need to modify the 'makefile' path for mpi and mkl libs first
type 'make' to compile
type 'make clean' to clean all binaries


to run the program, put the excutable, velocity, .nav file, and input.in to a work folder, mkdir d_obs, the 'mpirun -np 2 cuda_3D_FM < input.in'
NOTE:
input.in contains all necessary input paramters, should be straightforward to read
number '2' of -np will initiate 2 cores and 2 GPUs
before run, need to make a 'd_obs' direcotry under the same directory from which the excutable is run.
observed data is deposited in d_obs directory after run
in d_obs data, nr is fast direction, nt is slow direction
