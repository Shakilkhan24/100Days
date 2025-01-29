#include <cuda_runtime.h>
#include <iostream>

__global__ void baackwardKernel(float *Q, float *K, float *V, float *O,
                                float *dQ, float *dK, float *dV, float *dO,
                                float *L, int Bc, int Br,
                                int batch_size, int N, int nr_heads, int d)
{
    int Tr = ceil(N / Br);
    int Tc = ceil(N / Bc);

    // Q1 - > size of Br* d size in shared memory
    // O1 - > size of Br* d size in shared memory

    // K1 - > size of Bc *d size in shared memory
    // V1 - > size of Bc *d size in shared memory

    // L - > size of Br each

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    

}