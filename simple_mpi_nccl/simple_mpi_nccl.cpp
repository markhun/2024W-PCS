#include <cuda_runtime.h>
#include <nccl.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include <unistd.h>
#include <iostream>

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


int perform_experiment(int& myRank, int& nRanks, int* d_buffer, ncclComm_t& comm, cudaStream_t& stream, bool warmup) {
    // Timing variables
    double start, stop;
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    float nccl_broadcast_time_ms, nccl_kernel_time_ms, nccl_reduce_time_ms;

    if (myRank == 0) {
        if (warmup)
            printf("Starting warmup run\n");
        else
            printf("Starting benchmark run\n");
    }


    // Initialize array on rank 0
    if (myRank == 0) {
        CUDA_CHECK(cudaMemset(d_buffer, 0, TOTAL_BYTES));
        printf("Initialized buffer with zeros on root node\n");
    }

    // Broadcast the buffer from rank 0 to all other ranks
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    if (myRank == 0) printf("Rank %d: Starting broadcast\n", myRank);
    
    // Start timing broadcast
    start = MPI_Wtime();
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    
    NCCL_CHECK(ncclBroadcast(
        d_buffer,          // sendbuff
        d_buffer,          // recvbuff
        TOTAL_ELEMENTS,    // count
        ncclInt,           // datatype
        0,                 // root
        comm,              // communicator
        stream));          // CUDA stream
    
    // Wait for broadcast to complete and stop timing
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaEventElapsedTime(&nccl_broadcast_time_ms, startEvent, stopEvent));
    double broadcast_time = MPI_Wtime() - start;
    
    if (myRank == 0) {
        printf("Broadcast completed: %.3f ms (GPU), %.3f ms (wall)\n", 
            nccl_broadcast_time_ms, broadcast_time*1000);
    }
    
    // Execute AddOne kernel on each node's buffer
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    
    std::cout << "Process " << myRank << ": Processing data on GPU..." << std::endl;
    
    // Start timing kernel execution
    start = MPI_Wtime();
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    
    // Launch CUDA kernel
    launchAddOneKernel(d_buffer, TOTAL_ELEMENTS);
    
    // Wait for kernel to complete and stop timing
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaEventElapsedTime(&nccl_kernel_time_ms, startEvent, stopEvent));
    double kernel_time = MPI_Wtime() - start;
    
    if (myRank == 0) {
        printf("AddOne kernel completed: %.3f ms (GPU), %.3f ms (wall)\n", 
            nccl_kernel_time_ms, kernel_time*1000);
    }
    
    // Allocate buffer on root node to receive gathered data
    int *d_result = nullptr;
    if (myRank == 0) {
        CUDA_CHECK(cudaMalloc(&d_result, TOTAL_BYTES));
    }
    
    // Gather all data to root node
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    std::cout << "Process " << myRank << ": Starting Reduce operation..." << std::endl;
    
    // Start timing gather operation
    start = MPI_Wtime();
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    
    NCCL_CHECK(ncclReduce(
        d_buffer,          // sendbuff
        d_result,          // recvbuff
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
    CUDA_CHECK(cudaEventElapsedTime(&nccl_reduce_time_ms, startEvent, stopEvent));
    double reduce_time = MPI_Wtime() - start;
    
    if (myRank == 0) {
        printf("Reduce completed: %.3f ms (GPU), %.3f ms (wall)\n", 
            nccl_reduce_time_ms, reduce_time*1000);
    }
    
    // Verify on root node
    if (myRank == 0) {
        // Number of elements to verify
        const int verify_count = 1000;
        
        // Allocate host memory for verification
        int *h_result = (int*)malloc(sizeof(int) * verify_count);
        CUDA_CHECK(cudaMemcpy(h_result, d_result, sizeof(int) * verify_count, cudaMemcpyDeviceToHost));
        
        // Check if all elements equal the number of processes
        bool verification_passed = true;
        int expected_value = nRanks; // Each process added 1, so we expect mpi_size
        for (int i = 0; i < verify_count; i++) {
            if (h_result[i] != expected_value) {
                printf("ERROR: Verification failed at index %d - expected %d but got %d\n", 
                       i, expected_value, h_result[i]);
                verification_passed = false;
                break;
            }
        }
        
        if (verification_passed) {
            printf("VERIFICATION PASSED: All checked elements equal to %d (number of processes)\n", expected_value);
            std::cout << "Each element now equals " << expected_value 
            << " (1 from each of the " << nRanks << " processes)" << std::endl;

            // Print the first 10 elements as sample output
            // printf("Sample output (first 10 elements):\n");
            // for (int i = 0; i < 10; i++) {
            //     printf("result[%d] = %d\n", i, h_result[i]);
            // }
        }
        
        // Print performance summary
        std::cout << "\nPerformance Summary (Rank 0):" << std::endl;
        std::cout << "  NCCL Broadcast time: " << nccl_broadcast_time_ms << " ms" << std::endl;
        std::cout << "  Wall Broadcast time: " << broadcast_time << " seconds" << std::endl;
        std::cout << "  NCCL Kernel execution: " << nccl_kernel_time_ms << " ms" << std::endl;
        std::cout << "  Wall Kernel execution: " << kernel_time << " seconds" << std::endl;
        std::cout << "  NCCL Reduce operation: " << nccl_reduce_time_ms << " ms" << std::endl;
        std::cout << "  Wall Reduce operation: " << reduce_time << " seconds" << std::endl;
        std::cout << "  Total data size: " << NUMBER_OF_GiB_TO_SEND << " GiB (" << TOTAL_ELEMENTS << " integers)" << std::endl;
        std::cout << std::endl;

        if (!warmup) {  // print CSV data to stderr
            std::cerr << nccl_broadcast_time_ms << ","  << broadcast_time << "," << nccl_kernel_time_ms << "," << kernel_time << "," << nccl_reduce_time_ms << "," << reduce_time << ',' << TOTAL_BYTES << std::endl; 
        }

        std::cout << "------------------------------------------------------------------------" << std::endl;

        free(h_result);
        CUDA_CHECK(cudaFree(d_result));
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
    int device_count;
    cudaGetDeviceCount(&device_count);
       
    // Only rank 0 prints general information
    if (myRank == 0) {
        std::cout << "Running with " << nRanks << " MPI processes" << std::endl;
        std::cout << "Memory allocation details:" << std::endl;
        std::cout << "  - Total size: " << NUMBER_OF_GiB_TO_SEND << " GiB (" << TOTAL_BYTES << " bytes)" << std::endl;
        std::cout << "  - Element size: " << BYTES_PER_INT << " bytes" << std::endl;
        std::cout << "  - Number of elements: " << TOTAL_ELEMENTS << std::endl;
        std::cout << "  - Number of GPUs on node: " << device_count << std::endl;
    }

    // Select GPU based on local rank (assuming one GPU per process)
    int local_rank = 0;
    local_rank = myRank % device_count;
    CUDA_CHECK(cudaSetDevice(local_rank));

    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, local_rank);

    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, HOST_NAME_MAX + 1);

    std::cout << "Process " << myRank << " using GPU " << local_rank 
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
    int *d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, TOTAL_BYTES));
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
        retval = perform_experiment(myRank, nRanks, d_buffer, comm, stream, true);
        if (retval)
            return retval;
    }

    // write CSV header to stderr
    if (myRank == 0) {
        std::cerr << "nccl_broadcast_ms,wall_broadcast_s,nccl_kernel_ms,wall_kernel_s,nccl_reduce_ms,wall_reduce_s,total_buffer_size_bytes" << std::endl; 
    }

    for (int i = 0; i < NO_EXPERIMENT_RUNS; i++) {
        retval = perform_experiment(myRank, nRanks, d_buffer, comm, stream, false);
        if (retval)
            return retval;
    }
   
    // Clean up
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NCCL_CHECK(ncclCommDestroy(comm));
    
    MPI_Finalize();
    return 0;
}