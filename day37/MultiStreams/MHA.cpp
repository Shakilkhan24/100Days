#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>

#define HEADS    8
#define SEQ_LEN  128
#define DIM      768  // head dimension

__global__ void addition(const float* query, const float* key, const float* value,
                          float* output, int seq_len, int dim, int head_id) 
        {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * dim;  // work items per head
    if (idx < total) {
        int seq = idx / dim;  // which seq  | on row
        int d   = idx % dim;  // pos in seq | on col
     
        int offset = head_id * (seq_len * dim) + seq * dim;
        output[offset + d] = query[offset + d] + key[offset + d] + value[offset + d];
    }
}

int main(){
    size_t total_elements = HEADS * SEQ_LEN * DIM;
    size_t size = total_elements * sizeof(float);
    
    // Create one HIP stream per head.
    hipStream_t streams[HEADS];
    for (int i = 0; i < HEADS; i++){
        hipStreamCreate(&streams[i]);
    }
    
    float *key   = (float*)malloc(size);
    float *value = (float*)malloc(size);
    float *query = (float*)malloc(size);
    float *output= (float*)malloc(size);

    for (size_t i = 0; i < total_elements; i++){
        key[i]   = 3.0f;
        value[i] = 5.0f;
        query[i] = 6.0f;
    }

    float *d_key, *d_value, *d_query, *d_output;
    hipMalloc(&d_key, size);
    hipMalloc(&d_value, size);
    hipMalloc(&d_query, size);
    hipMalloc(&d_output, size);

    size_t headSize = SEQ_LEN * DIM * sizeof(float);
    
    // [HEADS][SEQ_LEN][DIM]
    for (int head = 0; head < HEADS; head++){
         int offset = head * SEQ_LEN * DIM;
         hipMemcpyAsync(d_key + offset,   key + offset,   headSize, hipMemcpyHostToDevice, streams[head]);
         hipMemcpyAsync(d_value + offset, value + offset, headSize, hipMemcpyHostToDevice, streams[head]);
         hipMemcpyAsync(d_query + offset, query + offset, headSize, hipMemcpyHostToDevice, streams[head]);
    }

    int threadsPerBlock = 256;      // threads per block 16x16 layout 
    int totalWork = SEQ_LEN * DIM;  // elements in a head   
    int blocks = (totalWork + threadsPerBlock - 1) / threadsPerBlock;

    for (int head = 0; head < HEADS; head++){
         hipLaunchKernelGGL(addition, dim3(blocks), dim3(threadsPerBlock), 0, streams[head],
                            d_query, d_key, d_value, d_output, SEQ_LEN, DIM, head);
    }

    for (int head = 0; head < HEADS; head++){
         int offset = head * SEQ_LEN * DIM;
         hipMemcpyAsync(output + offset, d_output + offset, headSize,
                        hipMemcpyDeviceToHost, streams[head]);
    }

    hipDeviceSynchronize();


    for (int i = 0; i < HEADS; i++){
        hipStreamDestroy(streams[i]);
    }

    hipFree(d_key);
    hipFree(d_value);
    hipFree(d_query);
    hipFree(d_output);
    free(key);
    free(value);
    free(query);
    free(output);
    return 0;
}
