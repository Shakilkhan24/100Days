#include <cuda_runtime.h>


__global__ void reluKernel(float*input,float*output,int N){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < N)
        output[index] = input[index] < 0 ? 0 : input[index];
}

void CudaReLU(float *A,float*B, int N){
    int ThreadsPerBlock = 256;
    int BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;
    reluKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(A, B,N);
}

//=========================

__global__ void reluKernelBackward(float *input, float *grad_input, float *grad_output, int N){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < N)
        grad_input[index] = input[index] < 0 ? 0 : grad_output[index];
}

void CudaReLUBackward(float *A, float *Gi, float *Go, int N){
    int ThreadsPerBlock = 256;
    int BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;
    reluKernelBackward<<<BlocksPerGrid, ThreadsPerBlock>>>(A, Gi, Go, N);
}   