#include <iostream>
#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>

__global__ void kernel_noise_image(float *X, const float *e, const float *alpha_hat, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float sqrt_alphas[2];

    if (threadIdx.x == 0) {
        sqrt_alphas[0] = sqrtf(*alpha_hat);
        sqrt_alphas[1] = sqrtf(1.0f - *alpha_hat);
    }
    
    __syncthreads();
    
    if (idx < N) {
        e[idx] = hiprand_normal(&state[idx]);

        X[idx] = sqrt_alphas[0] * X[idx] + sqrt_alphas[1] * e[idx];
    }
}

torch::Tensor noiseImage(torch::Tensor X, int t, torch::Tensor alpha_hat)
{
    torch::Tensor alpha_at_t = alpha_hat.index({t});
    
    float *d_X, *d_e, *d_alpha_hat;
    int N = X.numel();

    hipMalloc(&d_X, N * sizeof(float));
    hipMalloc(&d_e, N * sizeof(float));
    hipMalloc(&d_alpha_hat, sizeof(float));

    hipMemcpy(d_X, X.data_ptr<float>(), N * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_alpha_hat, alpha_at_t.data_ptr<float>(), sizeof(float), hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    kernel_noise_image<<<numBlocks, blockSize>>>(d_X, d_e, d_alpha_hat, N);

    hipDeviceSynchronize();

    hipMemcpy(X.data_ptr<float>(), d_X, N * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_X);
    hipFree(d_e);
    hipFree(d_alpha_hat);

    return X;
}

int main()
{
    torch::Tensor X = torch::rand({1, 3, 64, 64}, torch::kFloat32);
    torch::Tensor alpha_hat = torch::rand({1000}, torch::kFloat32);

    int t = 500;

    X = noiseImage(X, t, alpha_hat);

    std::cout << "Noisy image tensor shape: " << X.sizes() << std::endl;

    return 0;
}
