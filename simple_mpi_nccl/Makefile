HOSTS = 10.52.1.52  # Should be a comma-separated list of ip addr: 10.52.1.52,10.52.1.53,10.52.1.54
SLOTS = 2
HOSTS_AND_SLOTS = localhost:$(SLOTS),$(shell echo $(HOSTS) | sed 's/\([^,]*\)/\1:$(SLOTS)/g')

RUN_DATETIME=$$(date +"%FT%T")

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

TARGETS = simple_mpi_nccl
OBJECTS = addOne.o simple_mpi_nccl.o

# ----- Make Rules -----

all:	$(TARGETS)

%.o : %.cpp
	${CC} $< -c -o $@ ${CFLAGS}

%.o : %.cu
	${NVCC} ${NVCCFLAGS} $< -c -o $@ ${CFLAGS}

simple_mpi_nccl: $(OBJECTS)
	$(CC) ${CCFLAGS} $(OBJECTS) -o $@ $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.bin core

clean-data:
	rm -f data_*.csv

local_run: clean simple_mpi_nccl
	NCCL_DEBUG=WARN mpirun -np 2 simple_mpi_nccl 2>> data_$(RUN_DATETIME).csv | tee output_$(RUN_DATETIME).txt

distributed_run: clean simple_mpi_nccl
	@echo "Copying compiled program to hosts ..."
	clush -w ${HOSTS} -c ./simple_mpi_nccl
	@echo "Starting run ..."
	mpirun --host localhost:${NO_OF_SLOTS},${HOSTS_AND_SLOTS} --mca mpi_preconnect_all true ~/2024W-PCS/simple_mpi_nccl/simple_mpi_nccl 2>> data_$(RUN_DATETIME).csv | tee output_$(RUN_DATETIME).txt
