#include <cuda_runtime.h>


__global__ void tanhKernel(float*input,float*output,int N){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < N)
        output[index] = tanhf(input[index]);
}

void CudaTanH(float *A,float*B, int N){
    int ThreadsPerBlock = 256;
    int BlocksPerGrid = (N + ThreadsPerBlock - 1) / ThreadsPerBlock;
    tanhKernel<<<BlocksPerGrid, ThreadsPerBlock>>>(A, B,N);
}
