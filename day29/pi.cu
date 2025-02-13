#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__device__ float randomFloat(unsigned int *seed) {
    *seed = (*seed * 1664525u + 1013904223u);
    return (float)(*seed & 0x00FFFFFF) / (float)0x01000000;
}

__global__ void monteCarloPi(int iterations, unsigned long long *d_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int seed = tid;
    unsigned int local_count = 0;

    for (int i = 0; i < iterations; i++) {
        float x = randomFloat(&seed);
        float y = randomFloat(&seed);
        if (x * x + y * y <= 1.0f)
            local_count++;
    }

    atomicAdd(d_count, (unsigned long long)local_count);
}

int main() {
    int iterations = 10000;
    int threadsPerBlock = 256;
    int blocks = 256;

    unsigned long long totalPoints = (unsigned long long)iterations * threadsPerBlock * blocks;

    unsigned long long host_count = 0;
    unsigned long long *d_count;
    cudaMalloc((void**)&d_count, sizeof(unsigned long long));
    cudaMemset(d_count, 0, sizeof(unsigned long long));

    monteCarloPi<<<blocks, threadsPerBlock>>>(iterations, d_count);
    cudaDeviceSynchronize();

    cudaMemcpy(&host_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    float pi = 4.0f * (float)host_count / (float)totalPoints;
    printf("Estimated Pi = %f\n", pi);

    cudaFree(d_count);
    return 0;
}
