#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

#define HIP_CHECK(status)                                    \
    {                                                        \
        hipError_t err = status;                             \
        if (err != hipSuccess) {                             \
            std::cerr << "HIP error: " << hipGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            exit(err);                                       \
        }                                                    \
    }

__global__ void kernel(float *A, float *B, float *C, int N, int M, int K, float alpha, float beta) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// int main() {
//     float *A, *B, *C;
//     float *d_A, *d_B, *d_C;

//     float alpha, beta;
    
//     // For simplicity, we use a square matrix.
//     int SIZE = 100;
//     size_t mem_size = SIZE * SIZE * sizeof(float);

//     alpha = 1.0f;
//     beta = 0.0f;

//     A = (float*)malloc(mem_size);
//     B = (float*)malloc(mem_size);
//     C = (float*)malloc(mem_size);

//     for (int i = 0; i < SIZE * SIZE; ++i) {
//         A[i] = i%3;
//         B[i] = i%3;
//         C[i] = 0.0f;
//     }

//     HIP_CHECK(hipMalloc(&d_A, mem_size));
//     HIP_CHECK(hipMalloc(&d_B, mem_size));
//     HIP_CHECK(hipMalloc(&d_C, mem_size));

//     HIP_CHECK(hipMemcpy(d_A, A, mem_size, hipMemcpyHostToDevice));
//     HIP_CHECK(hipMemcpy(d_B, B, mem_size, hipMemcpyHostToDevice));
//     HIP_CHECK(hipMemcpy(d_C, C, mem_size, hipMemcpyHostToDevice));

//     dim3 threadsPerBlock(16, 16);
//     dim3 blocksPerGrid((SIZE + threadsPerBlock.x - 1) / threadsPerBlock.x,
//                        (SIZE + threadsPerBlock.y - 1) / threadsPerBlock.y);

//     hipLaunchKernelGGL(kernel, blocksPerGrid, threadsPerBlock, 0, 0, 
//                        d_A, d_B, d_C, SIZE, SIZE, SIZE, alpha, beta);

//     HIP_CHECK(hipDeviceSynchronize());
//     HIP_CHECK(hipMemcpy(C, d_C, mem_size, hipMemcpyDeviceToHost));

//     std::cout << "Result matrix C (first 10 elements):" << std::endl;
//     for (int i = 0; i < 10; ++i) {
//         std::cout << C[i] << " ";
//     }
//     std::cout << std::endl;

//     HIP_CHECK(hipFree(d_A));
//     HIP_CHECK(hipFree(d_B));
//     HIP_CHECK(hipFree(d_C));
//     free(A);
//     free(B);
//     free(C);

//     return 0;
// }
