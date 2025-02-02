#include <cuda_runtime.h>
#define N 32  
#include <stdio.h>
__device__ int lane_id() {
    return threadIdx.x & 31;
}

__device__ int atomicAggInc(int *ptr) {
    int mask = __match_any_sync(__activemask(), (unsigned long long)ptr);
    int leader = __ffs(mask) - 1;  
    int res;
    if (lane_id() == leader)       
        res = atomicAdd(ptr, __popc(mask));
    res = __shfl_sync(mask, res, leader); 
    return res + __popc(mask & ((1 << lane_id()) - 1)); 
}

__global__ void test_atomicAggInc(int *d_ptr, int *d_results) {
    int old_val = atomicAggInc(d_ptr);
    d_results[threadIdx.x] = old_val;  
}

int main() {
    int *d_ptr, *d_results;
    int h_ptr = 0;   
    int h_results[N];

    cudaMalloc(&d_ptr, sizeof(int));
    cudaMalloc(&d_results, N * sizeof(int));

    cudaMemcpy(d_ptr, &h_ptr, sizeof(int), cudaMemcpyHostToDevice);

    test_atomicAggInc<<<1, N>>>(d_ptr, d_results);

    cudaMemcpy(&h_ptr, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_results, d_results, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Final value of ptr: %d\n", h_ptr);
    printf("Old values returned by each thread:\n");
    for (int i = 0; i < N; i++) {
        printf("Thread %2d -> %d\n", i, h_results[i]);
    }

    cudaFree(d_ptr);
    cudaFree(d_results);

    return 0;
}
