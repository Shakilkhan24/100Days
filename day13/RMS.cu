#include <cuda_runtime.h>

// Define the CEILING macro
#define CEILING(x, y) (((x) + (y) - 1) / (y))

// WIll do it for vecotrs only
#define blockdimy 128
__global__ void RMSKernel1_V1(float *input, float *output, int w, int h)
{

    int row = blockIdx.x;
    int ty = threadIdx.y;
    int wrap_id = ty / 32;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    __shared__ float reduction[blockdimy / 32];
    float4 reg_array[CEILING((w / 4), blockdimy)];
    int reg_array_index = 0;

    if (row < h)
    {
        float rms_sum = 0;

#pragma unroll
        for (int i = ty; i < w / 4; i += blockdimy)
        {
            float4 val = reinterpret_cast<float4 *>(&input[row * w + i * 4])[0];
            rms_sum += val.x * val.x;
            rms_sum += val.y * val.y;
            rms_sum += val.z * val.z;
            rms_sum += val.w * val.w;

            reg_array[reg_array_index] = val;
            reg_array_index += 1;
        }
        for (int offset = 16; offset > 0; offset /= 2)
        {
            rms_sum += __shfl_down_sync(0xFFFFFFFF, rms_sum, offset);
        }
        if (ty % 32 == 0)
        {
            reduction[ty / 32] = rms_sum;
        }
        __syncthreads();
        if (ty < (blockdimy / 32))
        {
            rms_sum = reduction[ty];
        }
        __syncthreads();
        if (tid == 0)
        {
            float block_rms = 0;
            for (int i = 0; i < blockdimy / 32; i++)
            {
                block_rms += reduction[i];
            }
            output[row] = sqrt(block_rms / w);
        }
    }
}
