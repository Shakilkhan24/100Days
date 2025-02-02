#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <iostream>

__device__ int lane_id() {
    return threadIdx.x & 31;
}

__device__ float reduceMax(float val) {
    // threads of 32 we perform reduction on them
    for (int offset = 16; offset > 0; offset /= 2) {
        float temp = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(val, temp);
    }
    return val;
}

__device__ float atomicMaxFloat(float *addr, float value) {
    // give the adres of input
    // save old adress
    int *addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        if (old_val >= value) {
            return old_val;
        }
        old = atomicCAS(addr_as_int, assumed, __float_as_int(fmaxf(old_val, value)));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void MaxValue(float *data, float *max_value, int N) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    extern __shared__ float reduction[];

    float block_max = -INFINITY;

    for (int i = bx * blockDim.x + tx; i < N; i += gridDim.x * blockDim.x) {
        block_max = fmaxf(block_max, data[i]);
    }

    block_max = reduceMax(block_max);

    reduction[tx] = block_max;
    __syncthreads();

    if (tx == 0) {
        float final_max = -INFINITY;
        for (int i = 0; i < blockDim.x; ++i) {
            final_max = fmaxf(final_max, reduction[i]);
        }
        atomicMaxFloat(max_value, final_max);
    }
}

int main() {
    int N = 1024;
    float *host_data = (float*)malloc(N * sizeof(float));
    float host_result = -INFINITY;

    for (int i = 0; i < N; ++i) {
        host_data[i] = rand()%10000;
        if (host_data[i] > host_result) {
            host_result = host_data[i];
        }
    }

    float *device_data, *device_result;
    cudaMalloc(&device_data, N * sizeof(float));
    cudaMalloc(&device_result, sizeof(float));

    cudaMemcpy(device_data, host_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, &host_result, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    MaxValue<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(device_data, device_result, N);

    cudaMemcpy(&host_result, device_result, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Max value: " << host_result << std::endl;

    free(host_data);
    cudaFree(device_data);
    cudaFree(device_result);

    return 0;
}