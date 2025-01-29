#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define FILTER_SIZE 5

__global__ void convolutionkernel(float *input, float *output, float *kernel,
                                  int kernel_size, int w, int h)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int radius = kernel_size / 2;

    __shared__ float sharedInput[(BLOCK_SIZE + radius * radius) * (BLOCK_SIZE + radius * radius)]; 
    __shared__ float sharedFilter[FILTER_SIZE * FILTER_SIZE];

    if (tx < kernel_size && ty < kernel_size)
        sharedFilter[tx * kernel_size + ty] = kernel[tx + ty * kernel_size];

    int inputX = bx * BLOCK_SIZE + tx - radius; // Global X with halo offset
    int inputY = by * BLOCK_SIZE + ty - radius; // Global Y with halo offset

    inputX = max(0, min(w - 1, inputX));
    inputY = max(0, min(h - 1, inputY));

    sharedInput[ty * (BLOCK_SIZE + 4) + tx] = input[inputY * w + inputX];
    __syncthreads();

    if (tx < BLOCK_SIZE && ty < BLOCK_SIZE)
    {
        float sum = 0.0f;
        for (int ky = 0; ky < kernel_size; ky++)
        {
            for (int kx = 0; kx < kernel_size; kx++)
            {
                int sx = tx + kx;
                int sy = ty + ky;
                sum += sharedInput[sy * (BLOCK_SIZE + 4) + sx] * sharedFilter[ky * kernel_size + kx];
            }
        }
        output[(by * BLOCK_SIZE + ty) * w + (bx * BLOCK_SIZE + tx)] = sum;
    }
}