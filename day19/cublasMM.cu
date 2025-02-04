#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cassert>

#define CHECK_CUDA_CALL(x)                                                                  \
    if ((x) != cudaSuccess)                                                                 \
    {                                                                                       \
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaGetLastError()) << std::endl; \
        exit(1);                                                                            \
    }

#define CHECK_CUBLAS_CALL(x)                       \
    if ((x) != CUBLAS_STATUS_SUCCESS)              \
    {                                              \
        std::cerr << "cuBLAS error!" << std::endl; \
        exit(1);                                   \
    } 


__global__ void softmax_kernel(float *matrix, int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows)
    {
        float max_val = -INFINITY;
        for (int i = 0; i < cols; i++)
        {
            max_val = fmaxf(max_val, matrix[row * cols + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < cols; i++)
        {
            matrix[row * cols + i] = expf(matrix[row * cols + i] - max_val);
            sum += matrix[row * cols + i];
        }

        for (int i = 0; i < cols; i++)
        {
            matrix[row * cols + i] /= sum;
        }
    }
}

void self_attention(cublasHandle_t handle, float *d_q, float *d_k, float *d_v,
                    float *d_output, int batch_size, int seq_len, int dim)
{

    float alpha = 1.0f, beta = 0.0f;

    float *d_qkt;
    CHECK_CUDA_CALL(cudaMalloc(&d_qkt, batch_size * seq_len * seq_len * sizeof(float)));
    CHECK_CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                  seq_len, seq_len, dim, &alpha,
                                  d_q, seq_len, d_k, seq_len,
                                  &beta, d_qkt, seq_len));

    softmax_kernel<<<(seq_len + 255) / 256, 256>>>(d_qkt, seq_len, seq_len);
    CHECK_CUDA_CALL(cudaDeviceSynchronize());
    CHECK_CUBLAS_CALL(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  seq_len, dim, seq_len, &alpha,
                                  d_qkt, seq_len, d_v, seq_len,
                                  &beta, d_output, seq_len));
    cudaFree(d_qkt);
}

int main()
{
    cublasHandle_t handle;
    CHECK_CUBLAS_CALL(cublasCreate(&handle));

    int batch_size = 1, seq_len = 4, dim = 8;
    float *d_q, *d_k, *d_v, *d_output;

    CHECK_CUDA_CALL(cudaMalloc(&d_q, batch_size * seq_len * dim * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_k, batch_size * seq_len * dim * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_v, batch_size * seq_len * dim * sizeof(float)));
    CHECK_CUDA_CALL(cudaMalloc(&d_output, batch_size * seq_len * dim * sizeof(float)));

    float h_q[32] = {2,213,41,234,1312,334,123,41};
    float h_k[32] = {3};
    float h_v[32] = {4};

    CHECK_CUDA_CALL(cudaMemcpy(d_q, h_q, 32 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_k, h_k, 32 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_CALL(cudaMemcpy(d_v, h_v, 32 * sizeof(float), cudaMemcpyHostToDevice));

    self_attention(handle, d_q, d_k, d_v, d_output, batch_size, seq_len, dim);

    float h_output[32];
    CHECK_CUDA_CALL(cudaMemcpy(h_output, d_output, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Attention Output: " << std::endl;
    for (int i = 0; i < 32; i++)
    {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_output);
    cublasDestroy(handle);

    return 0;
}
