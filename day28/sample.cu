#include <cuda_runtime.h>
#include <torch/extension.h>
#include <float.h>

__global__ void update_x_kernel(
    float *x, const float *noise, const float *predicted_noise,
    float sqrt_alpha, float sqrt_alpha_hat, float beta, float alpha,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
    {
        x[idx] = (1.0f / sqrt_alpha) *(x[idx] - ((1 - alpha) / sqrt_alpha_hat) * predicted_noise[idx]) +sqrt(beta) * noise[idx];
    }
}

torch::Tensor update_x(torch::Tensor x, torch::Tensor noise, torch::Tensor predicted_noise,
                       torch::Tensor sqrt_alpha, torch::Tensor sqrt_alpha_hat,
                       torch::Tensor beta, torch::Tensor alpha)
{
    int numel = x.numel();
    float sqrt_alpha_val = sqrt_alpha.item<float>();
    float sqrt_alpha_hat_val = sqrt_alpha_hat.item<float>();
    float beta_val = beta.item<float>();
    float alpha_val = alpha.item<float>();
    
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    update_x_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), noise.data_ptr<float>(), predicted_noise.data_ptr<float>(),
        sqrt_alpha_val, sqrt_alpha_hat_val, beta_val, alpha_val, numel);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();

    return x;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("update_x", &update_x, "CUDA kernel for updating x");
}
