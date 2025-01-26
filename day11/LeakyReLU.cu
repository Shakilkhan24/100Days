#include <cuda_runtime.h>


__global__ void leakyreluKernel(float*input,float*output,float slope,int N){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < N)
        output[index] = input[index] < 0 ? input[index]*slope : input[index];
}

void CudaLeakyReLU(float *A,float*B,float slope ,int N){
    int ThreadsPerBlock = 256;
    int BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;
    leakyreluKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(A, B,slope,N);
}