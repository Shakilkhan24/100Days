#include <cuda_runtime.h>

__global__ void softmax(int w, int h, float* input, float* output)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  
  if (row < h && col < w)
  {
    float maxval = input[row*w];
    for (int i = 1; i<w; i++)
    {
      maxval = max(maxval, input[row*w + i]);
    }
    float divisor = 0.f;
    for (int i = 0; i<w; i++)
    {
      divisor += exp(input[row*w + i] - maxval);
    }
    output[row*w + col] = exp(input[row*w + col]-maxval)/(divisor);
  }
}

#define BLOCKDIMY 32
__global__ void  softmax2(int w, int h, float* input, float* output){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float reduction[BLOCKDIMY];
    if(row < h){

        float max_val = 0;
        for(int i = ty ; i < w;i+=BLOCKDIMY){
            max_val = fmax(max_val, input[row*w + i]);
        }
        // each kernel does the same reduction of maxmimu
        reduction[ty] = max_val;
        for(int stride = BLOCKDIMY /2; stride>1 ; stride>>=2){
            __syncthreads();
            if(ty < stride){
                reduction[ty] = fmax(reduction[ty], reduction[ty + stride]);
            }
        }
        __syncthreads();
        max_val = reduction[0];

        float devi_val = 0;
        for(int i = ty ; i < w;i+=BLOCKDIMY){
            devi_val += exp(input[row*w + i] - max_val);
        }
        reduction[ty] = devi_val;

        for(int stride = BLOCKDIMY/2 ; stride >1 ; stride>>=2){
            __syncthreads();
            if(ty < stride){
                reduction[ty] += reduction[ty + stride];
            }
        }
        __syncthreads();
        devi_val = reduction[0];

        for(int i = ty ; i <w ; i+= BLOCKDIMY){
            output[row*w + i] = exp(input[row*w + i] - max_val) / devi_val;
        }
    }
}