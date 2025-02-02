#include <cuda_runtime.h>
#include <iostream>

#define N 320

__device__ int lane_id(){
    return threadIdx.x & 31;
}

// incremenets threads by pointers
__device__ int atomicIncrement(int * ptr){
    int mask = __match_any_sync(__activemask(), (unsigned long long)ptr);
    int leader = __ffs(mask) -1;
    int res;
    if(lane_id() == leader){
        res = atomicAdd(ptr,__popc(mask)); // add on ptr number of active threads
    }
    __shfl_sync(mask,res,leader);
    return *ptr;
}

__global__ void testatomicIncrement(int *d_ptr, int *d_results){
    int val = atomicIncrement(d_ptr);
    d_results[threadIdx.x] = val;
}
    


int main() {
    int *d_ptr, *d_results;
    int h_ptr = 100;   
    int h_results[N];

    cudaMalloc(&d_ptr, sizeof(int));
    cudaMalloc(&d_results, N * sizeof(int));

    cudaMemcpy(d_ptr, &h_ptr, sizeof(int), cudaMemcpyHostToDevice);

    testatomicIncrement<<<1, N>>>(d_ptr, d_results);

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

