#include <hip/hip_runtime.h>
#include <iostream>

// Your HIP kernel remains the same.
extern "C" __global__ void kernel_addition(const float *A, const float *B, float *C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Host wrapper function that launches the kernel.
// This function will be callable from Python.
extern "C" void launch_kernel_addition(const float *A, const float *B, float *C, size_t N,
                                         int grid_x, int grid_y, int grid_z,
                                         int block_x, int block_y, int block_z) {
    // Create dim3 objects for grid and block dimensions.
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);

    // Launch the kernel with the provided configuration.
    hipLaunchKernelGGL(kernel_addition, grid, block, 0, 0, A, B, C, N);

    // Wait for the kernel to finish.
    hipDeviceSynchronize();
}
