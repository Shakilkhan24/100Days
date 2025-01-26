#include <cuda_runtime.h>

inline int prevPow2(int n) {
    if (n == 0) return 0;
    int prev = 1;
    while (prev <= n/2) {
        prev *= 2;
    }
    return prev;
}

__global__ void softmaxKernel(float *input, float *output, int Dim) {
    int batch_idx = blockIdx.x; // Current batch index
    int tid = threadIdx.x;      // Thread index within the block

    extern __shared__ float shared_data[];
    float max_val = -INFINITY;
    for (int i = tid; i < Dim; i += blockDim.x) {
        max_val = fmaxf(max_val, input[batch_idx * Dim + i]);
    }

    shared_data[tid] = max_val;
    __syncthreads();

    // Reduction for max_val
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + stride]);
        }
        __syncthreads();
    }
    max_val = shared_data[0];

    float sum_exp = 0.0f;
    for (int i = tid; i < Dim; i += blockDim.x) {
        output[batch_idx * Dim + i] = expf(input[batch_idx * Dim + i] - max_val);
        sum_exp += output[batch_idx * Dim + i];
    }

    shared_data[tid] = sum_exp;
    __syncthreads();

    // Reduction for sum_exp
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    sum_exp = shared_data[0];

    for (int i = tid; i < Dim; i += blockDim.x) {
        output[batch_idx * Dim + i] /= sum_exp;
    }
}


void CudaSoftmax(float *input, float *output, int BatchSize, int Dim) {
    int max_threads = min(512, Dim);
    int threads = prevPow2(max_threads);
    if (threads == 0) threads = 1; // Ensure at least 1 thread
    size_t shared_mem_size = threads * sizeof(float);
    softmaxKernel<<<BatchSize, threads, shared_mem_size>>>(input, output, Dim);
    cudaDeviceSynchronize(); // Ensure kernel completion
}