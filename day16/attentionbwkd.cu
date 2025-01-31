#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define BLOCK_SIZE 16

__global__ void attention_forward(float *Q, float *K, float *V, float *O, float *P, int seq_len, int dim)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    float softmaxscale = 1.0f / sqrtf(dim);

    if (row < seq_len && col < dim)
    {
        float sum = 0.0f;
        for (int k = 0; k < dim; k++)
        {
            sum += Q[row * dim + k] * K[col * dim + k];
        }

        O[row * dim + col] = sum * softmaxscale;

        __syncthreads();

        float max_value = -1e30f;
        for (int k = 0; k < dim; k++)
        {
            max_value = fmaxf(max_value, O[row * dim + k]);
        }
        __syncthreads();

        float denum = 0.0f;
        for (int k = 0; k < dim; k++)
        {
            P[row * dim + k] = expf(O[row * dim + k] - max_value);
            denum += P[row * dim + k];
        }

        __syncthreads();

        for (int k = 0; k < dim; k++)
        {
            O[row * dim + k] = P[row * dim + k] / denum;
        }
        __syncthreads();

        sum = 0.0f;
        for (int k = 0; k < dim; k++)
        {
            sum += O[row * dim + k] * V[col + k * dim];
        }
        O[row * dim + col] = sum;
    }
}

__global__ void attention_backward(float *Q, float *K, float *V, float *P, float *dO,
                                   float *dQ, float *dK, float *dV, float *dS,
                                   int seq_len, int dim)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < seq_len && col < dim)
    {
        // Compute dV
        float sum_dV = 0.0f;
        for (int i = 0; i < seq_len; i++)
        {
            sum_dV += P[i * seq_len + row] * dO[i * dim + col];
        }
        dV[row * dim + col] = sum_dV;
    }

    if (row < seq_len && col < seq_len)
    {
        float sum_dP = 0.0f;
        for (int i = 0; i < dim; i++)
        {
            sum_dP += dO[row * dim + i] * V[col * dim + i];
        }
        
        dS[row * seq_len + col] = P[row * seq_len + col] * (sum_dP - P[row * seq_len + col] * sum_dP);
    }

    if (row < seq_len && col < dim)
    {
        float sum_dQ = 0.0f;
        for (int i = 0; i < seq_len; i++)
        {
            sum_dQ += dS[row * seq_len + i] * K[i * dim + col];
        }
        dQ[row * dim + col] = sum_dQ;
    }

    if (row < dim && col < seq_len)
    {
        float sum_dK = 0.0f;
        for (int i = 0; i < seq_len; i++)
        {
            sum_dK += dS[i * seq_len + col] * Q[i * dim + row];
        }
        dK[row * seq_len + col] = sum_dK;
    }
}
void launch_attention_forward(float *Q, float *K, float *V, float *P, float *O, int seq_len, int dim)
{
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, (dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    attention_forward<<<gridDim, blockDim>>>(Q, K, V, P, O, seq_len, dim);
    cudaDeviceSynchronize();
}

void launch_attention_backward(float *Q, float *K, float *V, float *P, float *dO,
                               float *dQ, float *dK, float *dV, float *dS, int seq_len, int dim)
{
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE, (dim + BLOCK_SIZE - 1) / BLOCK_SIZE);
    attention_backward<<<gridDim, blockDim>>>(Q, K, V, P, dO, dQ, dK, dV, dS, seq_len, dim);
    cudaDeviceSynchronize();
}

int main()
{
    int seq_len = 4, dim = 4;
    size_t size_QKV = seq_len * dim * sizeof(float);
    size_t size_PdS = seq_len * seq_len * sizeof(float);

    std::vector<float> h_Q(seq_len * dim, 1.0f);
    std::vector<float> h_K(seq_len * dim, 1.0f);
    std::vector<float> h_V(seq_len * dim, 1.0f);
    std::vector<float> h_P(seq_len * seq_len, 0.25f);
    std::vector<float> h_dO(seq_len * dim, 1.0f);
    std::vector<float> h_O(seq_len * dim);

    float *d_Q, *d_K, *d_V, *d_P, *d_dO, *d_dQ, *d_dK, *d_dV, *d_dS, *d_O;
    cudaMalloc(&d_Q, size_QKV);
    cudaMalloc(&d_K, size_QKV);
    cudaMalloc(&d_V, size_QKV);
    cudaMalloc(&d_P, size_PdS);
    cudaMalloc(&d_dO, size_QKV);
    cudaMalloc(&d_dQ, size_QKV);
    cudaMalloc(&d_dK, size_QKV);
    cudaMalloc(&d_dV, size_QKV);
    cudaMalloc(&d_dS, size_PdS);
    cudaMalloc(&d_O, size_QKV);

    cudaMemcpy(d_Q, h_Q.data(), size_QKV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), size_QKV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V.data(), size_QKV, cudaMemcpyHostToDevice);
    cudaMemcpy(d_P, h_P.data(), size_PdS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dO, h_dO.data(), size_QKV, cudaMemcpyHostToDevice);

    launch_attention_forward(d_Q, d_K, d_V, d_P, d_O, seq_len, dim);
    cudaDeviceSynchronize();

    std::vector<float> O(seq_len * dim);
    cudaMemcpy(O.data(), d_O, size_QKV, cudaMemcpyDeviceToHost);
     std::cout << "O result:" << std::endl;
    for (int i = 0; i < seq_len; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            std::cout << O[i * dim + j] << " ";
        }
        std::cout << std::endl;
    }
    launch_attention_backward(d_Q, d_K, d_V, d_P, d_dO, d_dQ, d_dK, d_dV, d_dS, seq_len, dim);

    std::vector<float> h_dQ(seq_len * dim);
    cudaMemcpy(h_dQ.data(), d_dQ, size_QKV, cudaMemcpyDeviceToHost);

    std::vector<float> h_dV(seq_len * dim);
    cudaMemcpy(h_dV.data(), d_dV, size_QKV, cudaMemcpyDeviceToHost);

    std::vector<float> h_dK(seq_len * dim);
    cudaMemcpy(h_dK.data(), d_dK, size_QKV, cudaMemcpyDeviceToHost);

    std::cout << "dQ result:" << std::endl;
    for (int i = 0; i < seq_len; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            std::cout << h_dQ[i * dim + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "dK result:" << std::endl;
    for (int i = 0; i < seq_len; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            std::cout << h_dK[i * dim + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "dV result:" << std::endl;
    for (int i = 0; i < seq_len; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            std::cout << h_dV[i * dim + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_P);
    cudaFree(d_dO);
    cudaFree(d_dQ);
    cudaFree(d_dK);
    cudaFree(d_dV);
    cudaFree(d_dS);
    cudaFree(d_O);

    return 0;
}
