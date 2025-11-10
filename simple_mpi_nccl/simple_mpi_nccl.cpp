#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include <unistd.h>
#include <iostream>

#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>

#define CUDA_CHECK(cmd) do {                         \
  cudaError_t err = cmd;                             \
  if (err != cudaSuccess) {                          \
    printf("Failed: CUDA error %s:%d '%s'\n",        \
        __FILE__, __LINE__, cudaGetErrorString(err));\
    exit(EXIT_FAILURE);                              \
  }                                                  \
} while(0)

#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t res = cmd;                            \
  if (res != ncclSuccess) {                          \
    printf("Failed: NCCL error %s:%d '%s'\n",        \
        __FILE__, __LINE__, ncclGetErrorString(res));\
    exit(EXIT_FAILURE);                              \
  }                                                  \
} while(0)

constexpr size_t NO_WARMUP_RUNS = 5;  
constexpr size_t NO_EXPERIMENT_RUNS = 10;

// Define memory size constants for better readability
constexpr size_t ONE_GB_IN_BYTES = 1ULL << 30;  // 1 GiB = 2^30 bytes
constexpr size_t BYTES_PER_INT = sizeof(int); // Size of one integer in bytes

// Calculate the number of integers needed for send
constexpr size_t NUMBER_OF_GiB_TO_SEND = 4;
constexpr size_t TOTAL_BYTES = NUMBER_OF_GiB_TO_SEND * ONE_GB_IN_BYTES;
constexpr size_t TOTAL_ELEMENTS = TOTAL_BYTES / BYTES_PER_INT;

// forward declare CUDA kernel to add 1 to each element of an array
void launchAddOneKernel(int* deviceArray, size_t numElements);


int perform_experiment(int& myRank, int& nRanks, int* deviceBuffer, ncclComm_t& comm, cudaStream_t& stream, bool warmup) {
    // Timing variables
    double start, stop;
    float ncclBroadcastTimeMS, ncclKernelTimeMS, ncclReduceTimeMS;
    
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    if (myRank == 0) {
        if (warmup)
            printf("Starting warmup run\n");
        else
            printf("Starting benchmark run\n");
    }


    // Initialize array on rank 0
    if (myRank == 0) {
        CUDA_CHECK(cudaMemset(deviceBuffer, 0, TOTAL_BYTES));
        printf("Initialized buffer with zeros on root node\n");
    }

    // Broadcast the buffer from rank 0 to all other ranks
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    if (myRank == 0) printf("Rank %d: Starting broadcast\n", myRank);
    
    // Start timing broadcast
    start = MPI_Wtime();
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    
    NCCL_CHECK(ncclBroadcast(
        deviceBuffer,          // sendbuff
        deviceBuffer,          // recvbuff
        TOTAL_ELEMENTS,    // count
        ncclInt,           // datatype
        0,                 // root
        comm,              // communicator
        stream));          // CUDA stream
    
    // Wait for broadcast to complete and stop timing
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaEventElapsedTime(&ncclBroadcastTimeMS, startEvent, stopEvent));
    double broadcastTime = MPI_Wtime() - start;
    
    if (myRank == 0) {
        printf("Broadcast completed: %.3f ms (GPU), %.3f ms (wall)\n", 
            ncclBroadcastTimeMS, broadcastTime*1000);
    }
    
    // Execute AddOne kernel on each node's buffer
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    
    std::cout << "Process " << myRank << ": Processing data on GPU..." << std::endl;
    
    // Start timing kernel execution
    start = MPI_Wtime();
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    
    // Launch CUDA kernel
    launchAddOneKernel(deviceBuffer, TOTAL_ELEMENTS);
    
    // Wait for kernel to complete and stop timing
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaEventElapsedTime(&ncclKernelTimeMS, startEvent, stopEvent));
    double kernelTime = MPI_Wtime() - start;
    
    if (myRank == 0) {
        printf("AddOne kernel completed: %.3f ms (GPU), %.3f ms (wall)\n", 
            ncclKernelTimeMS, kernelTime*1000);
    }
    
    // Allocate buffer on root node to receive gathered data
    int *deviceResults = nullptr;
    if (myRank == 0) {
        CUDA_CHECK(cudaMalloc(&deviceResults, TOTAL_BYTES));
    }
    
    // Gather all data to root node
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    std::cout << "Process " << myRank << ": Starting Reduce operation..." << std::endl;
    
    // Start timing gather operation
    start = MPI_Wtime();
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    
    NCCL_CHECK(ncclReduce(
        deviceBuffer,          // sendbuff
        deviceResults,          // recvbuff
        TOTAL_ELEMENTS,    // count
        ncclInt,           // datatype
        ncclSum,           // reduction operation
        0,                 // root
        comm,              // communicator
        stream));          // CUDA stream
    
    // Wait for gather to complete and stop timing
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaEventElapsedTime(&ncclReduceTimeMS, startEvent, stopEvent));
    double reduceTime = MPI_Wtime() - start;
    
    if (myRank == 0) {
        printf("Reduce completed: %.3f ms (GPU), %.3f ms (wall)\n", 
            ncclReduceTimeMS, reduceTime*1000);
    }
    
    // Verify on root node
    if (myRank == 0) {
        // Number of elements to verify
        const int verifyCount = 1000;
        
        // Allocate host memory for verification
        int *h_result = (int*)malloc(sizeof(int) * verifyCount);
        CUDA_CHECK(cudaMemcpy(h_result, deviceResults, sizeof(int) * verifyCount, cudaMemcpyDeviceToHost));
        
        // Check if all elements equal the number of processes
        bool verificationPassed = true;
        int expectedValue = nRanks; // Each process added 1, so we expect mpi_size
        for (int i = 0; i < verifyCount; i++) {
            if (h_result[i] != expectedValue) {
                printf("ERROR: Verification failed at index %d - expected %d but got %d\n", 
                       i, expectedValue, h_result[i]);
                verificationPassed = false;
                break;
            }
        }
        
        if (verificationPassed) {
            printf("VERIFICATION PASSED: All checked elements equal to %d (number of processes)\n", expectedValue);
            std::cout << "Each element now equals " << expectedValue 
            << " (1 from each of the " << nRanks << " processes)" << std::endl;

            // Print the first 10 elements as sample output
            // printf("Sample output (first 10 elements):\n");
            // for (int i = 0; i < 10; i++) {
            //     printf("result[%d] = %d\n", i, h_result[i]);
            // }
        }
        
        // Print performance summary
        std::cout << "\nPerformance Summary (Rank 0):" << std::endl;
        std::cout << "  NCCL Broadcast time: " << ncclBroadcastTimeMS << " ms" << std::endl;
        std::cout << "  Wall Broadcast time: " << broadcastTime << " seconds" << std::endl;
        std::cout << "  NCCL Kernel execution: " << ncclKernelTimeMS << " ms" << std::endl;
        std::cout << "  Wall Kernel execution: " << kernelTime << " seconds" << std::endl;
        std::cout << "  NCCL Reduce operation: " << ncclReduceTimeMS << " ms" << std::endl;
        std::cout << "  Wall Reduce operation: " << reduceTime << " seconds" << std::endl;
        std::cout << "  Total data size: " << NUMBER_OF_GiB_TO_SEND << " GiB (" << TOTAL_ELEMENTS << " integers)" << std::endl;
        std::cout << std::endl;

        if (!warmup) {  // print CSV data to stderr
            std::cerr << ncclBroadcastTimeMS << ","  << broadcastTime << "," << ncclKernelTimeMS << "," << kernelTime << "," << ncclReduceTimeMS << "," << reduceTime << ',' << TOTAL_BYTES << std::endl; 
        }

        std::cout << "------------------------------------------------------------------------" << std::endl;

        free(h_result);
        CUDA_CHECK(cudaFree(deviceResults));
        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));

    }

    return 0;
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    int myRank, nRanks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
       
    // Set device to local rank
    // Each process outputs its local GPU setup
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
       
    // Only rank 0 prints general information
    if (myRank == 0) {
        std::cout << "Running with " << nRanks << " MPI processes" << std::endl;
        std::cout << "Memory allocation details:" << std::endl;
        std::cout << "  - Total size: " << NUMBER_OF_GiB_TO_SEND << " GiB (" << TOTAL_BYTES << " bytes)" << std::endl;
        std::cout << "  - Element size: " << BYTES_PER_INT << " bytes" << std::endl;
        std::cout << "  - Number of elements: " << TOTAL_ELEMENTS << std::endl;
        std::cout << "  - Number of GPUs on node: " << deviceCount << std::endl;
    }

    // Select GPU based on node-local rank (assuming one GPU per process)
    int localRank = 0;
    localRank = myRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(localRank));

    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, localRank);

    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, HOST_NAME_MAX + 1);

    std::cout << "Process " << myRank << " using GPU " << localRank 
              << " (" << prop.name << ") on node " << hostname << std::endl;
    
    // Initialize NCCL
    ncclUniqueId ncclId;
    
   
    // Root gets NCCL ID and broadcasts it to all
    if (myRank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&ncclId));
    }

    // Synchronize all processes before broadcast
    MPI_Barrier(MPI_COMM_WORLD);
    // Start timing for initialization
    double start, stop;
    start = MPI_Wtime();
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize NCCL communicator
    ncclComm_t comm;
    ncclCommInitRank(&comm, nRanks, ncclId, myRank);
    printf("ncclCommInitRank completed\n");
    
    // Allocate device memory
    int *deviceBuffer;
    CUDA_CHECK(cudaMalloc(&deviceBuffer, TOTAL_BYTES));
    printf("Allocated DeviceBuffer\n");
       
    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    stop = MPI_Wtime();
    if (myRank == 0) {
        printf("Initialization time: %.3f ms\n", (stop-start)*1000);
    }
    
    int retval = 0;
    for (int i = 0; i < NO_WARMUP_RUNS; i++) {
        retval = perform_experiment(myRank, nRanks, deviceBuffer, comm, stream, true);
        if (retval)
            return retval;
    }

    // write CSV header to stderr
    if (myRank == 0) {
        std::cerr << "nccl_broadcast_ms,wall_broadcast_s,nccl_kernel_ms,wall_kernel_s,nccl_reduce_ms,wall_reduce_s,total_buffer_size_bytes" << std::endl; 
    }

    for (int i = 0; i < NO_EXPERIMENT_RUNS; i++) {
        retval = perform_experiment(myRank, nRanks, deviceBuffer, comm, stream, false);
        if (retval)
            return retval;
    }
   
    // Clean up
    CUDA_CHECK(cudaFree(deviceBuffer));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));
    
    MPI_Finalize();
    return 0;
}