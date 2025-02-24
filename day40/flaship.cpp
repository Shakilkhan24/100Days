#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define HIP_CALL(call)                                                       \
    do                                                                       \
    {                                                                        \
        hipError_t err = call;                                               \
        if (err != hipSuccess)                                               \
        {                                                                    \
            fprintf(stderr, "HIP error at %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#ifndef INFINITY
#define INFINITY (1.0f / 0.0f)
#endif

// The kernel uses the following parameters:
// Q, K, V: input matrices (layout: [batch, head, N, d])
// N: sequence length, d: feature dimension per head
// Tc: number of key/value tiles, Tr: number of query tiles
// Bc: number of rows per key tile, Br: number of rows per query tile
// softmax_scale: scaling factor (typically 1/sqrt(d))
// l, m: auxiliary arrays for accumulating softmax denominator and max (layout: [batch, head, N])
// O: output matrix (same layout as Q)

__global__ void flashAttentionKernel(const float *__restrict__ Q,
                                     const float *__restrict__ K,
                                     const float *__restrict__ V,
                                     int N, int d,
                                     int Tc, int Tr,
                                     int Bc, int Br,
                                     float softmax_scale,
                                     float *__restrict__ l,
                                     float *__restrict__ m,
                                     float *__restrict__ O)
{
    const uint32_t tid = threadIdx.x;
    const uint32_t batch = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t BrD = Br * d;
    const uint32_t BcD = Bc * d;
    const uint32_t BcBr = Bc * Br;
    const uint32_t qkvBase = (batch * gridDim.y * N * d) + (head * N * d); // the offest in the memory so that (bx,by) is a block of (N,d)
    const uint32_t accBase = (batch * gridDim.y * N) + (head * N);         // the offset for the accumulator/maximum and

    extern __shared__ float sharedMem[];
    float *sharedQuery = sharedMem;         // begining of it
    float *sharedKey = sharedQuery + BrD;   // next Br * d
    float *sharedValue = sharedKey + BcD;   // next Bc * d
    float *sharedScore = sharedValue + BcD; // next Br * d
    // use this so that it is easier to understand the shared memory usage

    for (uint32_t tileKey = 0; tileKey < Tc; tileKey++)
    {
        // Load the current key and value tile into shared memory.
        for (uint32_t idx = tid; idx < BcD; idx += blockDim.x)
        {
            sharedKey[idx] = K[qkvBase + (tileKey * BcD) + idx];
            sharedValue[idx] = V[qkvBase + (tileKey * BcD) + idx];
        }
        __syncthreads();

        for (uint32_t tileQuery = 0; tileQuery < Tr; tileQuery++)
        {
            // Load the current query tile into shared memory.
            for (uint32_t idx = tid; idx < BrD; idx += blockDim.x)
            {
                sharedQuery[idx] = Q[qkvBase + (tileQuery * BrD) + idx];
            }
            __syncthreads();

            if (tid < Br)
            {
                float local_max = -INFINITY;
                for (uint32_t keyRow = 0; keyRow < Bc; keyRow++)
                {
                    float dot = 0.0f;
                    for (uint32_t x = 0; x < d; x++)
                    {
                        float q_val = sharedQuery[tid * d + x];
                        float k_val = sharedKey[keyRow * d + x];
                        dot += q_val * k_val;
                    }
                    dot *= softmax_scale;
                    sharedScore[tid * Bc + keyRow] = dot;
                    if (dot > local_max)
                        local_max = dot;
                }

                float local_sum = 0.0f;
                for (uint32_t keyRow = 0; keyRow < Bc; keyRow++)
                {
                    float exp_val = expf(sharedScore[tid * Bc + keyRow] - local_max);
                    sharedScore[tid * Bc + keyRow] = exp_val;
                    local_sum += exp_val;
                }

                // The accumulators l (sum) and m (max) are stored per query row.
                uint32_t accIndex = accBase + (tileQuery * Br) + tid;
                float prev_max = m[accIndex];
                float prev_sum = l[accIndex];

                float new_max = (prev_max > local_max) ? prev_max : local_max;
                float new_sum = expf(prev_max - new_max) * prev_sum + expf(local_max - new_max) * local_sum;

                for (uint32_t x = 0; x < d; x++)
                {
                    float weighted_val = 0.0f;
                    for (uint32_t keyRow = 0; keyRow < Bc; keyRow++)
                    {
                        weighted_val += sharedScore[tid * Bc + keyRow] * sharedValue[keyRow * d + x];
                    }
                    uint32_t outIndex = qkvBase + (tileQuery * Br * d) + (tid * d) + x;
                    O[outIndex] = (1.0f / new_sum) *
                                  (expf(prev_max - new_max) * O[outIndex] * prev_sum +
                                   expf(local_max - new_max) * weighted_val);
                }
                m[accIndex] = new_max;
                l[accIndex] = new_sum;
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

int main()
{
    const int batchSize = 1;
    const int numHeads = 1;
    const int seqLength = 64; // N
    const int headDim = 16;   // D

    const int Bc = 32;
    const int Br = 32;
    const int Tc = (seqLength + Bc - 1) / Bc;
    const int Tr = (seqLength + Br - 1) / Br;

    const int qkvElements = batchSize * numHeads * seqLength * headDim;
    const int accElements = batchSize * numHeads * seqLength;

    float softmax_scale = 1.0f / sqrtf(headDim);

    float *h_Q = (float *)malloc(qkvElements * sizeof(float));
    float *h_K = (float *)malloc(qkvElements * sizeof(float));
    float *h_V = (float *)malloc(qkvElements * sizeof(float));
    float *h_O = (float *)malloc(qkvElements * sizeof(float));
    float *h_l = (float *)malloc(accElements * sizeof(float));
    float *h_m = (float *)malloc(accElements * sizeof(float));

    for (int i = 0; i < qkvElements; i++)
    {
        h_Q[i] = 1.0f;
        h_K[i] = 1.0f;
        h_V[i] = 1.0f;
        h_O[i] = 0.0f;
    }

    for (int i = 0; i < accElements; i++)
    {
        h_l[i] = 0.0f;
        h_m[i] = -INFINITY;
    }

    float *d_Q, *d_K, *d_V, *d_O, *d_l, *d_m;
    HIP_CALL(hipMalloc(&d_Q, qkvElements * sizeof(float)));
    HIP_CALL(hipMalloc(&d_K, qkvElements * sizeof(float)));
    HIP_CALL(hipMalloc(&d_V, qkvElements * sizeof(float)));
    HIP_CALL(hipMalloc(&d_O, qkvElements * sizeof(float)));
    HIP_CALL(hipMalloc(&d_l, accElements * sizeof(float)));
    HIP_CALL(hipMalloc(&d_m, accElements * sizeof(float)));

    HIP_CALL(hipMemcpy(d_Q, h_Q, qkvElements * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_K, h_K, qkvElements * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_V, h_V, qkvElements * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_O, h_O, qkvElements * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_l, h_l, accElements * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_m, h_m, accElements * sizeof(float), hipMemcpyHostToDevice));

    // Shared memory size per block:
    // sharedQuery:  Br * d
    // sharedKey:    Bc * d
    // sharedValue:  Bc * d
    // sharedScore:  Br * Bc
    size_t sharedMemSize = ((Br * headDim) + (Bc * headDim) + (Bc * headDim) + (Br * Bc)) * sizeof(float);

    dim3 gridDim(batchSize, numHeads);
    int threadsPerBlock = (Bc > Br) ? Bc : Br;
    dim3 blockDim(threadsPerBlock);

    hipLaunchKernelGGL(flashAttentionKernel,
                       gridDim, blockDim, sharedMemSize, 0,
                       d_Q, d_K, d_V,
                       seqLength, headDim,
                       Tc, Tr,
                       Bc, Br,
                       softmax_scale,
                       d_l, d_m, d_O);

    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipMemcpy(h_O, d_O, qkvElements * sizeof(float), hipMemcpyDeviceToHost));

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_O);
    free(h_l);
    free(h_m);
    HIP_CALL(hipFree(d_Q));
    HIP_CALL(hipFree(d_K));
    HIP_CALL(hipFree(d_V));
    HIP_CALL(hipFree(d_O));
    HIP_CALL(hipFree(d_l));
    HIP_CALL(hipFree(d_m));

    return 0;
}
