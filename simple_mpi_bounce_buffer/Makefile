# ----- Make Macros -----

CC = mpicxx
CCFLAGS = -O3 -std=c++14 -march=native

NVCC = nvcc
# use arch=compute_60,code=sm_60 for NVIDIA Tesla P100
# use arch=compute_80,code=sm_80 for NVIDIA RTX 4060
NVCCFLAGS = -lineinfo -O3 -std=c++14 -gencode arch=compute_60,code=sm_60 -ccbin=mpicxx -Xcompiler -fopenmp -Xptxas="-v" 

CFLAGS = -I/usr/local/cuda-12.6/targets/x86_64-linux/include

# LD_FLAGS = -L${NCCL_DIR}/lib -lnccl
LD_FLAGS =  -L/usr/local/cuda/lib64 -lcudart -lnccl

TARGETS = simple_mpi_bounce_buffer
OBJECTS = addOne.o simple_mpi_bounce_buffer.o

# ----- Make Rules -----

all:	$(TARGETS)

%.o : %.cpp
	${CC} $< -c -o $@ ${CFLAGS}

%.o : %.cu
	${NVCC} ${NVCCFLAGS} $< -c -o $@ ${CFLAGS}

simple_mpi_bounce_buffer: $(OBJECTS)
	$(CC) ${CCFLAGS} $(OBJECTS) -o $@ $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.bin core

clean-data:
	rm -f data_*.csv

local_run:
	mpirun -np 2 simple_mpi_bounce_buffer 2>> data_$$(date +"%FT%T").csv