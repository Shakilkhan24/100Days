#include <hip/hip_runtime.h>
#include <iostream>

// Macro to check HIP errors.
#define CHECK_HIP_ERROR(error)                                     \
    {                                                              \
        if ((error) != hipSuccess)                                 \
        {                                                          \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at line " << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

#define TILESIZE 32

__global__ void kernel(const float *A, const float *B, float *C, int N)
{
    __shared__ float As[TILESIZE][TILESIZE];
    __shared__ float Bs[TILESIZE][TILESIZE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float sum = 0.0f;

    for (int t = 0; t < N; t += TILESIZE)
    {
        Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        As[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];

        __syncthreads();

        for (int k = 0; k < TILESIZE; k++)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < N && col < N)
    {
        C[row * N + col] = sum;
    }
}

int main(){
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    int N = 1024; // Size of the matrix

    size_t size = N*N* sizeof(float);

    // Allocate host memory
    A = (float *)malloc(size);
    B = (float *)malloc(size);
    C = (float *)malloc(size);

    for(int i = 0 ; i < N * N ; i++){
        A[i] = i;
        B[i] = i;
    }

    CHECK_HIP_ERROR(hipMalloc((void**)&d_A,size));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_B,size));
    CHECK_HIP_ERROR(hipMalloc((void**)&d_C,size));

    CHECK_HIP_ERROR(hipMemcpy(d_A, A, size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, B, size, hipMemcpyHostToDevice));

    dim3 Threads(TILESIZE, TILESIZE);
    dim3 Blocks((N+Threads.x-1)/Threads.x, (N+Threads.y-1)/Threads.y);
    hipLaunchKernelGGL(kernel, Blocks, Threads, 0, 0, d_A, d_B, d_C, N);

    CHECK_HIP_ERROR(hipMemcpy(C, d_C, size, hipMemcpyDeviceToHost));

    // Check the result
    for(int i = 0 ; i < 10 ; i++){
        for(int j = 0 ; j < 10 ; j++){
            std::cout << C[i*N+j] << " ";
        }
        std::cout << std::endl;
    }

    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));

    free(A);
    free(B);
    free(C);

    return 0;
}
