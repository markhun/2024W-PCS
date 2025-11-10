#include <iostream>
#include <climits>
#include <unistd.h>

#include <cuda_runtime.h>
#include <mpi.h>

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


int perform_experiment(int& mpiRank, int& mpiSize, int* hostBuffer, int* resultBuffer, int* deviceBuffer, bool warmup) {
    
    if (mpiRank == 0) {
        if (warmup)
            printf("Starting warmup run\n");
        else
            printf("Starting benchmark run\n");
    }


    // Synchronize all processes before broadcast
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Broadcast the array from rank 0 to all other ranks
    double start_time = MPI_Wtime();
    std::cout << "Process " << mpiRank << ": Starting broadcast..." << std::endl;
    
    MPI_Bcast(hostBuffer, TOTAL_ELEMENTS, MPI_INT, 0, MPI_COMM_WORLD);
    
    double broadcastTime = MPI_Wtime() - start_time;
    std::cout << "Process " << mpiRank << ": Broadcast completed in " 
              << broadcastTime << " seconds" << std::endl;
        
    // Copy data from host to device
    std::cout << "Process " << mpiRank << ": Copying data to GPU..." << std::endl;
    start_time = MPI_Wtime();
    
    cudaError_t error = cudaMemcpy(deviceBuffer, hostBuffer, TOTAL_BYTES, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Process " << mpiRank << ": Error copying to device: " 
                  << cudaGetErrorString(error) << std::endl;
        cudaFree(deviceBuffer);
        cudaFreeHost(hostBuffer);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    double h2dTime = MPI_Wtime() - start_time;
    std::cout << "Process " << mpiRank << ": H2D transfer completed in " 
              << h2dTime << " seconds" << std::endl;
    
    // Launch kernel to add 1 to each element
    std::cout << "Process " << mpiRank << ": Processing data on GPU..." << std::endl;
    
    // Start timing kernel execution
    start_time = MPI_Wtime();
    
    // Launch CUDA kernel
    launchAddOneKernel(deviceBuffer, TOTAL_ELEMENTS);
    
    // Check for kernel launch errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Process " << mpiRank << ": Kernel launch error: " 
                  << cudaGetErrorString(error) << std::endl;
        cudaFree(deviceBuffer);
        cudaFreeHost(hostBuffer);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // Wait for kernel to finish
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "Process " << mpiRank << ": Kernel execution error: " 
                  << cudaGetErrorString(error) << std::endl;
        cudaFree(deviceBuffer);
        cudaFreeHost(hostBuffer);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    double kernelTime = MPI_Wtime() - start_time;
    std::cout << "Process " << mpiRank << ": Kernel execution completed in " 
              << kernelTime << " seconds" << std::endl;
    
    // Copy data back from device to host
    std::cout << "Process " << mpiRank << ": Copying data back to CPU..." << std::endl;
    start_time = MPI_Wtime();
    
    error = cudaMemcpy(hostBuffer, deviceBuffer, TOTAL_BYTES, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cerr << "Process " << mpiRank << ": Error copying to host: " 
                  << cudaGetErrorString(error) << std::endl;
        cudaFree(deviceBuffer);
        cudaFreeHost(hostBuffer);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    double d2hTime = MPI_Wtime() - start_time;
    std::cout << "Process " << mpiRank << ": D2H transfer completed in " 
              << d2hTime << " seconds" << std::endl;
    
    // Perform MPI_Reduce to sum the arrays at root process (rank 0)
    std::cout << "Process " << mpiRank << ": Starting Reduce operation..." << std::endl;
    start_time = MPI_Wtime();
    
    // Root process is 0
    const int root = 0;
    MPI_Reduce(hostBuffer, resultBuffer, TOTAL_ELEMENTS, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
    
    double reduceTime = MPI_Wtime() - start_time;
    std::cout << "Process " << mpiRank << ": Reduce completed in " 
              << reduceTime << " seconds" << std::endl;
    
    // Verify results on rank 0 only
    if (mpiRank == 0) {
        std::cout << "Rank 0: Verifying results..." << std::endl;
        bool correct = true;
        int expectedValue = mpiSize; // Each process added 1, so we expect mpiSize
        
        // Check first 10 elements
        for (int i = 0; i < 10 && i < TOTAL_ELEMENTS; i++) {
            if (resultBuffer[i] != expectedValue) {
                std::cerr << "Verification failed at index " << i << ": Expected " 
                          << expectedValue << ", got " << resultBuffer[i] << std::endl;
                correct = false;
                break;
            }
        }
        
        // Check last 10 elements
        for (size_t i = TOTAL_ELEMENTS - 10; i < TOTAL_ELEMENTS; i++) {
            if (resultBuffer[i] != expectedValue) {
                std::cerr << "Verification failed at index " << i << ": Expected " 
                          << expectedValue << ", got " << resultBuffer[i] << std::endl;
                correct = false;
                break;
            }
        }
        
        if (correct) {
            std::cout << "All checked elements verified correctly!" << std::endl;
            std::cout << "Each element now equals " << expectedValue 
                      << " (1 from each of the " << mpiSize << " processes)" << std::endl;
        }
        
        // Print performance summary
        std::cout << "\nPerformance Summary (Rank 0):" << std::endl;
        std::cout << "  Broadcast time: " << broadcastTime << " seconds" << std::endl;
        std::cout << "  Host->Device transfer: " << h2dTime << " seconds" << std::endl;
        std::cout << "  Kernel execution: " << kernelTime << " seconds" << std::endl;
        std::cout << "  Device->Host transfer: " << d2hTime << " seconds" << std::endl;
        std::cout << "  Reduce operation: " << reduceTime << " seconds" << std::endl;
        std::cout << "  Total data size: " << NUMBER_OF_GiB_TO_SEND << " GiB (" << TOTAL_ELEMENTS << " integers)" << std::endl;
        std::cout << std::endl;

        if (!warmup) {  // print CSV data to stderr
            std::cerr << broadcastTime << "," << h2dTime << "," << kernelTime << "," << d2hTime << "," << reduceTime << "," << TOTAL_BYTES << std::endl; 
        }

        std::cout << "------------------------------------------------------------------------" << std::endl;
    }


    std::cout << "Process " << mpiRank << ": Reseting host array to zeros..." << std::endl;
    for (size_t i = 0; i < TOTAL_ELEMENTS; i++) {
        hostBuffer[i] = 0;
    }


    return 0;
}


int main(int argc, char** argv) {
    // Initialize MPI
    int mpiRank, mpiSize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    
    // Each process outputs its local GPU setup
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    // Only rank 0 prints general information
    if (mpiRank == 0) {
        std::cout << "Running with " << mpiSize << " MPI processes" << std::endl;
        std::cout << "Memory allocation details:" << std::endl;
        std::cout << "  - Total size: " << NUMBER_OF_GiB_TO_SEND << " GiB (" << TOTAL_BYTES << " bytes)" << std::endl;
        std::cout << "  - Element size: " << BYTES_PER_INT << " bytes" << std::endl;
        std::cout << "  - Number of elements: " << TOTAL_ELEMENTS << std::endl;
        std::cout << "  - Number of GPUs on node: " << deviceCount << std::endl;
    }
       
    // Select GPU based on node-local rank (assuming one GPU per process)
    int localRank = mpiRank % deviceCount;
    cudaSetDevice(localRank);
    
    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, localRank);
    
    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, HOST_NAME_MAX + 1);

    std::cout << "Process " << mpiRank << " using GPU " << localRank 
              << " (" << prop.name << ") on node " << hostname << std::endl;
    
    // Allocate host memory (pinned for faster transfers)
    int* hostBuffer = nullptr;
    cudaError_t error = cudaMallocHost(&hostBuffer, TOTAL_BYTES);
    if (error != cudaSuccess) {
        std::cerr << "Process " << mpiRank << ": Error allocating host memory: " 
                  << cudaGetErrorString(error) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Allocate device memory
    int* deviceBuffer = nullptr;
    error = cudaMalloc(&deviceBuffer, TOTAL_BYTES);
    if (error != cudaSuccess) {
        std::cerr << "Process " << mpiRank << ": Error allocating device memory: " 
                    << cudaGetErrorString(error) << std::endl;
        cudaFreeHost(hostBuffer);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // Initialize array on rank 0
    if (mpiRank == 0) {
        std::cout << "Rank 0: Initializing array to zeros..." << std::endl;
        for (size_t i = 0; i < TOTAL_ELEMENTS; i++) {
            hostBuffer[i] = 0;
        }
    }

    // Allocate buffer for the result on rank 0 only
    int* resultBuffer = nullptr;
    if (mpiRank == 0) {
        error = cudaMallocHost(&resultBuffer, TOTAL_BYTES);
        if (error != cudaSuccess) {
            std::cerr << "Process " << mpiRank << ": Error allocating result buffer: " 
                        << cudaGetErrorString(error) << std::endl;
            cudaFreeHost(hostBuffer);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    int retval = 0;
    for (int i = 0; i < NO_WARMUP_RUNS; i++) {
        retval = perform_experiment(mpiRank, mpiSize, hostBuffer, resultBuffer, deviceBuffer, true);
        if (retval)
            return retval;
    }
    
    // write CSV header to stderr
    if (mpiRank == 0) {
        std::cerr << "broadcast_s,h2d_transfer_s,kernel_execution_s,d2h_transfer_s,reduce_op_s,total_buffer_size_bytes" << std::endl; 
    }

    for (int i = 0; i < NO_EXPERIMENT_RUNS; i++) {
        retval = perform_experiment(mpiRank, mpiSize, hostBuffer, resultBuffer, deviceBuffer, false);
        if (retval)
            return retval;
    }

    if (mpiRank == 0) {
        // Free result array (only on rank 0)
        cudaFreeHost(resultBuffer);
    }
    
    // Free memory
    cudaFree(deviceBuffer);
    cudaFreeHost(hostBuffer);
    
    // Finalize MPI
    MPI_Finalize();
    
    return retval;
}