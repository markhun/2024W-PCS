# ----- Make Macros -----

CC = mpicxx

NVCC = nvcc
NVCCFLAGS = -lineinfo -O3 -std=c++14 -gencode arch=compute_80,code=sm_80 -ccbin=mpicxx -Xcompiler -fopenmp -Xptxas="-v" 

CFLAGS = -I/usr/local/cuda-12.6/targets/x86_64-linux/include

# LD_FLAGS = -L${NCCL_DIR}/lib -lnccl
LD_FLAGS =  -L/usr/local/cuda/lib64 -lcudart -lnccl

TARGETS = simple_mpi_nccl
OBJECTS = addOne.o simple_mpi_nccl.o

# ----- Make Rules -----

all:	$(TARGETS)

%.o : %.cpp
	${CC} $< -c -o $@ ${CFLAGS}

%.o : %.cu
	${NVCC} ${NVCCFLAGS} $< -c -o $@ ${CFLAGS}

simple_mpi_nccl: $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.bin core

clean-data:
	rm -f data_*.csv

local_run:
	NCCL_DEBUG=WARN mpirun -np 1 simple_mpi_nccl 2>> data_$$(date +"%FT%T").csv