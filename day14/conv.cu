#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 32
#define FILTER_SIZE 5

__global__ void convolutionkernel(float *input, float *output, float *kernel,
                                  int kernel_size, int w, int h)
{
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int radius = kernel_size / 2;

    extern __shared__ float sharedInput[]; // Use dynamic shared memory
    __shared__ float sharedFilter[FILTER_SIZE * FILTER_SIZE];

    if (tx < kernel_size && ty < kernel_size)
        sharedFilter[tx * kernel_size + ty] = kernel[tx + ty * kernel_size];

    int inputX = bx * BLOCK_SIZE + tx - radius;
    int inputY = by * BLOCK_SIZE + ty - radius;

    inputX = max(0, min(w - 1, inputX));
    inputY = max(0, min(h - 1, inputY));

    sharedInput[ty * (BLOCK_SIZE + radius ) + tx] = input[inputY * w + inputX];
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
                sum += sharedInput[sy * (BLOCK_SIZE + radius ) + sx] * sharedFilter[ky * kernel_size + kx];
            }
        }
        output[(by * BLOCK_SIZE + ty) * w + (bx * BLOCK_SIZE + tx)] = sum;
    }
}
int main()
{

    int w = 10;                    
    int h = 10;                     
    int kernel_size = FILTER_SIZE; 

    float *input = new float[w * h];
    float *output = new float[w * h];
    float *kernel = new float[kernel_size * kernel_size];

    for (int i = 0; i < w * h; ++i)
    {
        input[i] = 1; // Values from 1 to 25
    }

    for (int i = 0; i < kernel_size * kernel_size; ++i)
    {
        kernel[i] = 1.0f; 
    }

    float *d_input, *d_output, *d_kernel;
    cudaMalloc((void **)&d_input, w * h * sizeof(float));
    cudaMalloc((void **)&d_output, w * h * sizeof(float));
    cudaMalloc((void **)&d_kernel, kernel_size * kernel_size * sizeof(float));

    cudaMemcpy(d_input, input, w * h * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((w + BLOCK_SIZE - 1) / BLOCK_SIZE, (h + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int sharedMemorySize = (BLOCK_SIZE + kernel_size - 1) * (BLOCK_SIZE + kernel_size - 1) * sizeof(float);
    convolutionkernel<<<grid, block,sharedMemorySize>>>(d_input, d_output, d_kernel, kernel_size, w, h);

    cudaError_t err = cudaGetLastError();   
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    cudaMemcpy(output, d_output, w * h * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Convolution Result (output):" << std::endl;
    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            std::cout << output[i * w + j] << "  ";
        }
        std::cout << std::endl;
    }

    delete[] input;
    delete[] output;
    delete[] kernel;
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return 0;
}