#include <cuda_runtime.h>

__global__ void softmax(int w, int h, float *input, float *output)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < h && col < w)
    {
        float maxval = input[row * w];
        for (int i = 1; i < w; i++)
        {
            maxval = max(maxval, input[row * w + i]);
        }
        float divisor = 0.f;
        for (int i = 0; i < w; i++)
        {
            divisor += exp(input[row * w + i] - maxval);
        }
        output[row * w + col] = exp(input[row * w + col] - maxval) / (divisor);
    }
}

#define BLOCKDIMY 32
__global__ void softmax2(int w, int h, float *input, float *output)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float reduction[BLOCKDIMY];
    if (row < h)
    {

        float max_val = 0;
        for (int i = ty; i < w; i += BLOCKDIMY)
        {
            max_val = fmax(max_val, input[row * w + i]);
        }
        // each kernel does the same reduction of maxmimu
        reduction[ty] = max_val;
        for (int stride = BLOCKDIMY / 2; stride > 1; stride >>= 2)
        {
            __syncthreads();
            if (ty < stride)
            {
                reduction[ty] = fmax(reduction[ty], reduction[ty + stride]);
            }
        }
        __syncthreads();
        max_val = reduction[0];

        float devi_val = 0;
        for (int i = ty; i < w; i += BLOCKDIMY)
        {
            devi_val += exp(input[row * w + i] - max_val);
        }
        reduction[ty] = devi_val;

        for (int stride = BLOCKDIMY / 2; stride > 1; stride >>= 2)
        {
            __syncthreads();
            if (ty < stride)
            {
                reduction[ty] += reduction[ty + stride];
            }
        }
        __syncthreads();
        devi_val = reduction[0];

        for (int i = ty; i < w; i += BLOCKDIMY)
        {
            output[row * w + i] = exp(input[row * w + i] - max_val) / devi_val;
        }
    }
}

#define BLOCKDIMY 32
__global__ void softmaxReduction(int w, int h, float *input, float *output)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y;

    __shared__ float reduction[BLOCKDIMY];
    if (row < h)
    {

        float max_val = 0;
        for (int i = ty; i < w/4; i += BLOCKDIMY)
        {
            float4 val = reinterpret_cast<float4*>(&input[row * w + i*4])[0];
            max_val = fmax(max_val, val.x);
            max_val = fmax(max_val, val.y);
            max_val = fmax(max_val, val.z);
            max_val = fmax(max_val, val.w);
        }

        for (int i = 16; i > 0; i /= 2)
        {
            max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, i, 32));
        }
        if (ty % 32 == 0)
        {
            reduction[wrap_id] = max_val;
        }
        __syncthreads() if (warp_id == 0)
        {
            maxval = ty < BLOCK_DIM_Y / 32 ? reduction[ty] : 0;

            for (int i = 16; i > 0; i /= 2)
            {
                maxval = fmaxf(maxval, __shfl_xor_sync(MASK, maxval, i, 32));
            }
        }

        __syncthreads();
        max_val = reduction[0];

        float devi_val = 0;
        for (int i = ty; i < w; i += BLOCKDIMY)
        {
            devi_val += exp(input[row * w + i] - max_val);
        }
        reduction[ty] = devi_val;

        for (int stride = BLOCKDIMY / 2; stride > 1; stride >>= 2)
        {
            __syncthreads();
            if (ty < stride)
            {
                reduction[ty] += reduction[ty + stride];
            }
        }
        __syncthreads();
        devi_val = reduction[0];

        for (int i = ty; i < w; i += BLOCKDIMY)
        {
            output[row * w + i] = exp(input[row * w + i] - max_val) / devi_val;
        }
    }
}

template <typename scalar_t>
__global__ void softmax_kernel10(scalar_t* __restrict__ a, scalar_t* __restrict__ b, int w, int h)
{
  int row = blockIdx.x;
  int ty = threadIdx.y;
  int warp_id = ty/32;
  int lane_id = ty%32;
  
  __shared__ float reduction[BLOCK_DIM_Y/32]; 
  float4 reg_array[CEILING((WIDTH/4),BLOCK_DIM_Y)];

  int reg_array_idx = 0;
  if (row < h)
  {
    float maxval = 0;
#pragma unroll URF
    for (int i = ty; i<WIDTH/4; i+=BLOCK_DIM_Y)
    {
      float4 val = reinterpret_cast<float4*>(&a[row*WIDTH + i*4])[0];
      maxval = fmaxf(maxval, val.x);
      maxval = fmaxf(maxval, val.y);
      maxval = fmaxf(maxval, val.z);
      maxval = fmaxf(maxval, val.w);
      reg_array[reg_array_idx] = val;
      reg_array_idx+=1;
    }
    maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 16, 32));
    maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 8, 32));
    maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 4, 32));
    maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 2, 32));
    maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 1, 32));

    if (lane_id == 0)
    {
      reduction[warp_id] = maxval;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        maxval = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 16, 32));
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 8, 32));
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 4, 32));
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 2, 32));
        maxval = fmaxf(maxval, __shfl_xor_sync(0xffffffff, maxval, 1, 32));
    }
    if (ty == 0)
    {
        reduction[0] = maxval;
    }
    __syncthreads();
    maxval = reduction[0];
    float divisor = 0.f;
    reg_array_idx=0;
#pragma unroll URF
    for (int i = ty; i<WIDTH/4; i+=BLOCK_DIM_Y)
    {
        float4 val = reg_array[reg_array_idx];
        val.x = __expf(val.x - maxval);
        val.y = __expf(val.y - maxval);
        val.z = __expf(val.z - maxval);
        val.w = __expf(val.w - maxval);
        divisor += val.x;
        divisor += val.y;
        divisor += val.z;
        divisor += val.w;
        reg_array[reg_array_idx] = val;
      reg_array_idx+=1;
    }

    divisor += __shfl_xor_sync(0xffffffff, divisor, 16, 32);
    divisor += __shfl_xor_sync(0xffffffff, divisor, 8, 32);
    divisor += __shfl_xor_sync(0xffffffff, divisor, 4, 32);
    divisor += __shfl_xor_sync(0xffffffff, divisor, 2, 32);
    divisor += __shfl_xor_sync(0xffffffff, divisor, 1, 32);

    if (lane_id == 0)
    {
      reduction[warp_id] = divisor;
    }

    __syncthreads();
    if (warp_id == 0)
    {
        divisor = ty < BLOCK_DIM_Y/32 ? reduction[ty] : 0;
        divisor += __shfl_xor_sync(0xffffffff, divisor, 16, 32);
        divisor += __shfl_xor_sync(0xffffffff, divisor, 8, 32);
        divisor += __shfl_xor_sync(0xffffffff, divisor, 4);
        divisor += __shfl_xor_sync(0xffffffff, divisor, 2);
        divisor += __shfl_xor_sync(0xffffffff, divisor, 1);
    }

    if (ty == 0)
    {
        reduction[0] = divisor;
    }

    __syncthreads();
    divisor = reduction[0];

    reg_array_idx = 0;
#pragma unroll URF
    for (int i = ty; i<WIDTH/4; i+=BLOCK_DIM_Y)
    {
        float4 val = reg_array[reg_array_idx];
        val.x = val.x/divisor;
        val.y = val.y/divisor;
        val.z = val.z/divisor;
        val.w = val.w/divisor;
        reinterpret_cast<float4*>(&b[row*WIDTH + i*4])[0] = val;
      reg_array_idx+=1;
    }

  }
}