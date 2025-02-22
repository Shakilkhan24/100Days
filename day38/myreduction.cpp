#include <hip/hip_runtime.h>
#include <iostream>
#include <float.h> // for FLT_MAX

#ifndef WARP_SIZE
#define WARP_SIZE 64
#endif

#define HIPCHECK(error)                                            \
    {                                                              \
        if ((error) != hipSuccess)                                 \
        {                                                          \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at line " << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }


template <typename scalar_t>
__global__ void reduce_max_1d(const scalar_t *__restrict__ input,
                              scalar_t *__restrict__ output,
                              int n)
{
    extern __shared__ float sdata[];
    const uint32_t tid = threadIdx.x;
    const uint32_t i = blockIdx.x * (blockDim.x * 2) + tid;
    const uint32_t lane    = tid % WARP_SIZE;
    const uint32_t warp_id = tid / WARP_SIZE;
    float max_val = -FLT_MAX;
    if (i < n)
        max_val = input[i];
    if (i + blockDim.x < n)
        max_val = fmaxf(max_val, input[i + blockDim.x]);

    for (uint32_t offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        max_val = fmaxf(max_val, __shfl_down(max_val, offset, WARP_SIZE));
    }


    if (lane == 0)
    {
        sdata[warp_id] = max_val;
    }
    __syncthreads();


    const uint32_t numWarps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    if (tid < numWarps)
    {
        max_val = sdata[lane];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        {
            max_val = fmaxf(max_val, __shfl_down(max_val, offset, WARP_SIZE));
        }
        if (lane == 0)
            sdata[tid] = max_val;
    }
    __syncthreads();

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}


int main()
{
    const int n = 102400;                  
    std::vector<float> h_input(n, 1.0f); 
    h_input[500] = 133.0f;               

    float *d_input;
    float *d_output;
    HIPCHECK(hipMalloc(&d_input, n * sizeof(float)));
    HIPCHECK(hipMalloc(&d_output, sizeof(float))); 

    HIPCHECK(hipMemcpy(d_input, h_input.data(), n * sizeof(float), hipMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);
    const size_t sharedMemSize = ((threadsPerBlock + WARP_SIZE - 1) / WARP_SIZE) * sizeof(float);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce_max_1d<float>), dim3(blocks), dim3(threadsPerBlock), sharedMemSize, 0, d_input, d_output, n);

    float h_output;
    HIPCHECK(hipMemcpy(&h_output, d_output, sizeof(float), hipMemcpyDeviceToHost));


    HIPCHECK(hipFree(d_input));
    HIPCHECK(hipFree(d_output));

    std::cout << "Maximum value: " << h_output << std::endl;

    return 0;
}
