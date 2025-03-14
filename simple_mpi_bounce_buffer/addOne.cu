#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to add 1 to each element of an array
__global__ void addOne(int* arr, size_t n) {
    // Calculate global thread ID
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Make sure we don't go out of bounds
    if (idx < n) {
        arr[idx] += 1;
    }
}

void launchAddOneKernel(int* deviceArray, size_t numElements) {
    const int THREADS_PER_BLOCK = 256;
    const int BLOCKS_IN_GRID = (numElements + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    addOne<<<BLOCKS_IN_GRID, THREADS_PER_BLOCK>>>(deviceArray, numElements);

    // cudaThreadSynchronize();
}