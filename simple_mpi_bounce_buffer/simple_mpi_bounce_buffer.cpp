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


int perform_experiment(int& mpi_rank, int& mpi_size, int* hostArray, int* resultArray, int* deviceArray, bool warmup) {
    
    if (mpi_rank == 0) {
        if (warmup)
            printf("Starting warmup run\n");
        else
            printf("Starting benchmark run\n");
    }


    // Synchronize all processes before broadcast
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Broadcast the array from rank 0 to all other ranks
    double start_time = MPI_Wtime();
    std::cout << "Process " << mpi_rank << ": Starting broadcast..." << std::endl;
    
    MPI_Bcast(hostArray, TOTAL_ELEMENTS, MPI_INT, 0, MPI_COMM_WORLD);
    
    double broadcast_time = MPI_Wtime() - start_time;
    std::cout << "Process " << mpi_rank << ": Broadcast completed in " 
              << broadcast_time << " seconds" << std::endl;
        
    // Copy data from host to device
    std::cout << "Process " << mpi_rank << ": Copying data to GPU..." << std::endl;
    start_time = MPI_Wtime();
    
    cudaError_t error = cudaMemcpy(deviceArray, hostArray, TOTAL_BYTES, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        std::cerr << "Process " << mpi_rank << ": Error copying to device: " 
                  << cudaGetErrorString(error) << std::endl;
        cudaFree(deviceArray);
        cudaFreeHost(hostArray);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    double h2d_time = MPI_Wtime() - start_time;
    std::cout << "Process " << mpi_rank << ": H2D transfer completed in " 
              << h2d_time << " seconds" << std::endl;
    
    // Launch kernel to add 1 to each element
    std::cout << "Process " << mpi_rank << ": Processing data on GPU..." << std::endl;
    
    // Start timing kernel execution
    start_time = MPI_Wtime();
    
    // Launch CUDA kernel
    launchAddOneKernel(deviceArray, TOTAL_ELEMENTS);
    
    // Check for kernel launch errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Process " << mpi_rank << ": Kernel launch error: " 
                  << cudaGetErrorString(error) << std::endl;
        cudaFree(deviceArray);
        cudaFreeHost(hostArray);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // Wait for kernel to finish
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "Process " << mpi_rank << ": Kernel execution error: " 
                  << cudaGetErrorString(error) << std::endl;
        cudaFree(deviceArray);
        cudaFreeHost(hostArray);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    double kernel_time = MPI_Wtime() - start_time;
    std::cout << "Process " << mpi_rank << ": Kernel execution completed in " 
              << kernel_time << " seconds" << std::endl;
    
    // Copy data back from device to host
    std::cout << "Process " << mpi_rank << ": Copying data back to CPU..." << std::endl;
    start_time = MPI_Wtime();
    
    error = cudaMemcpy(hostArray, deviceArray, TOTAL_BYTES, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        std::cerr << "Process " << mpi_rank << ": Error copying to host: " 
                  << cudaGetErrorString(error) << std::endl;
        cudaFree(deviceArray);
        cudaFreeHost(hostArray);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    double d2h_time = MPI_Wtime() - start_time;
    std::cout << "Process " << mpi_rank << ": D2H transfer completed in " 
              << d2h_time << " seconds" << std::endl;
    
    // Perform MPI_Reduce to sum the arrays at root process (rank 0)
    std::cout << "Process " << mpi_rank << ": Starting Reduce operation..." << std::endl;
    start_time = MPI_Wtime();
    
    // Root process is 0
    const int root = 0;
    MPI_Reduce(hostArray, resultArray, TOTAL_ELEMENTS, MPI_INT, MPI_SUM, root, MPI_COMM_WORLD);
    
    double reduce_time = MPI_Wtime() - start_time;
    std::cout << "Process " << mpi_rank << ": Reduce completed in " 
              << reduce_time << " seconds" << std::endl;
    
    // Verify results on rank 0 only
    if (mpi_rank == 0) {
        std::cout << "Rank 0: Verifying results..." << std::endl;
        bool correct = true;
        int expected_value = mpi_size; // Each process added 1, so we expect mpi_size
        
        // Check first 10 elements
        for (int i = 0; i < 10 && i < TOTAL_ELEMENTS; i++) {
            if (resultArray[i] != expected_value) {
                std::cerr << "Verification failed at index " << i << ": Expected " 
                          << expected_value << ", got " << resultArray[i] << std::endl;
                correct = false;
                break;
            }
        }
        
        // Check last 10 elements
        for (size_t i = TOTAL_ELEMENTS - 10; i < TOTAL_ELEMENTS; i++) {
            if (resultArray[i] != expected_value) {
                std::cerr << "Verification failed at index " << i << ": Expected " 
                          << expected_value << ", got " << resultArray[i] << std::endl;
                correct = false;
                break;
            }
        }
        
        if (correct) {
            std::cout << "All checked elements verified correctly!" << std::endl;
            std::cout << "Each element now equals " << expected_value 
                      << " (1 from each of the " << mpi_size << " processes)" << std::endl;
        }
        
        // Print performance summary
        std::cout << "\nPerformance Summary (Rank 0):" << std::endl;
        std::cout << "  Broadcast time: " << broadcast_time << " seconds" << std::endl;
        std::cout << "  Host->Device transfer: " << h2d_time << " seconds" << std::endl;
        std::cout << "  Kernel execution: " << kernel_time << " seconds" << std::endl;
        std::cout << "  Device->Host transfer: " << d2h_time << " seconds" << std::endl;
        std::cout << "  Reduce operation: " << reduce_time << " seconds" << std::endl;
        std::cout << "  Total data size: " << NUMBER_OF_GiB_TO_SEND << " GiB (" << TOTAL_ELEMENTS << " integers)" << std::endl;
        std::cout << std::endl;

        if (!warmup) {  // print CSV data to stderr
            std::cerr << broadcast_time << "," << h2d_time << "," << kernel_time << "," << d2h_time << "," << reduce_time << "," << TOTAL_BYTES << std::endl; 
        }

        std::cout << "------------------------------------------------------------------------" << std::endl;
    }


    std::cout << "Process " << mpi_rank << ": Reseting host array to zeros..." << std::endl;
    for (size_t i = 0; i < TOTAL_ELEMENTS; i++) {
        hostArray[i] = 0;
    }


    return 0;
}


int main(int argc, char** argv) {
    // Initialize MPI
    int mpi_rank, mpi_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Each process outputs its local GPU setup
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    // Only rank 0 prints general information
    if (mpi_rank == 0) {
        std::cout << "Running with " << mpi_size << " MPI processes" << std::endl;
        std::cout << "Memory allocation details:" << std::endl;
        std::cout << "  - Total size: " << NUMBER_OF_GiB_TO_SEND << " GiB (" << TOTAL_BYTES << " bytes)" << std::endl;
        std::cout << "  - Element size: " << BYTES_PER_INT << " bytes" << std::endl;
        std::cout << "  - Number of elements: " << TOTAL_ELEMENTS << std::endl;
        std::cout << "  - Number of GPUs on node: " << device_count << std::endl;
    }
       
    // Select GPU based on local rank (assuming one GPU per process)
    int local_rank = mpi_rank % device_count;
    cudaSetDevice(local_rank);
    
    // Get GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, local_rank);
    
    char hostname[HOST_NAME_MAX + 1];
    gethostname(hostname, HOST_NAME_MAX + 1);

    std::cout << "Process " << mpi_rank << " using GPU " << local_rank 
              << " (" << prop.name << ") on node " << hostname << std::endl;
    
    // Allocate host memory (pinned for faster transfers)
    int* hostArray = nullptr;
    cudaError_t error = cudaMallocHost(&hostArray, TOTAL_BYTES);
    if (error != cudaSuccess) {
        std::cerr << "Process " << mpi_rank << ": Error allocating host memory: " 
                  << cudaGetErrorString(error) << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Allocate device memory
    int* deviceArray = nullptr;
    error = cudaMalloc(&deviceArray, TOTAL_BYTES);
    if (error != cudaSuccess) {
        std::cerr << "Process " << mpi_rank << ": Error allocating device memory: " 
                    << cudaGetErrorString(error) << std::endl;
        cudaFreeHost(hostArray);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // Initialize array on rank 0
    if (mpi_rank == 0) {
        std::cout << "Rank 0: Initializing array to zeros..." << std::endl;
        for (size_t i = 0; i < TOTAL_ELEMENTS; i++) {
            hostArray[i] = 0;
        }
    }

    // Allocate buffer for the result on rank 0 only
    int* resultArray = nullptr;
    if (mpi_rank == 0) {
        error = cudaMallocHost(&resultArray, TOTAL_BYTES);
        if (error != cudaSuccess) {
            std::cerr << "Process " << mpi_rank << ": Error allocating result buffer: " 
                        << cudaGetErrorString(error) << std::endl;
            cudaFreeHost(hostArray);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    int retval = 0;
    for (int i = 0; i < NO_WARMUP_RUNS; i++) {
        retval = perform_experiment(mpi_rank, mpi_size, hostArray, resultArray, deviceArray, true);
        if (retval)
            return retval;
    }
    
    // write CSV header to stderr
    if (mpi_rank == 0) {
        std::cerr << "broadcast_s,h2d_transfer_s,kernel_execution_s,d2h_transfer_s,reduce_op_s,total_buffer_size_bytes" << std::endl; 
    }

    for (int i = 0; i < NO_EXPERIMENT_RUNS; i++) {
        retval = perform_experiment(mpi_rank, mpi_size, hostArray, resultArray, deviceArray, false);
        if (retval)
            return retval;
    }

    if (mpi_rank == 0) {
        // Free result array (only on rank 0)
        cudaFreeHost(resultArray);
    }
    
    // Free memory
    cudaFree(deviceArray);
    cudaFreeHost(hostArray);
    
    // Finalize MPI
    MPI_Finalize();
    
    return retval;
}