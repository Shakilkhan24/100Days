#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

#define DIM 128
#define NHEADS 4
#define QPROJ_DIM 64
#define KVPROJ_DIM 85
#define HEADDIM 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void layernorm_kernel(const float* input, float* output, int D, float epsilon) {
    int idx = blockIdx.x;
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < D; i++){
        mean += input[idx * D + i];
    }
    mean /= D;
    for (int i = 0; i < D; i++){
        float diff = input[idx * D + i] - mean;
        var += diff * diff;
    }
    var /= D;
    float inv_std = rsqrtf(var + epsilon);
    for (int i = 0; i < D; i++){
        output[idx * D + i] = (input[idx * D + i] - mean) * inv_std;
    }
}

__global__ void attention_kernel(const float* Q, const float* K, const float* V, float* output, int S, int d) {
    int q_idx = blockIdx.x;
    extern __shared__ float shared[];
    float score = 0.0f;
    for (int j = threadIdx.x; j < S; j += blockDim.x) {
        float dot = 0.0f;
        for (int k = 0; k < d; k++){
            dot += Q[q_idx * d + k] * K[j * d + k];
        }
        dot /= sqrtf((float)d);
        shared[j] = dot;
    }
    __syncthreads();
    float max_val = -1e20f;
    for (int j = 0; j < S; j++){
        if (shared[j] > max_val) max_val = shared[j];
    }
    float sum_exp = 0.0f;
    for (int j = 0; j < S; j++){
        shared[j] = expf(shared[j] - max_val);
        sum_exp += shared[j];
    }
    for (int k = threadIdx.x; k < d; k += blockDim.x){
        float out_val = 0.0f;
        for (int j = 0; j < S; j++){
            float weight = shared[j] / sum_exp;
            out_val += weight * V[j * d + k];
        }
        output[q_idx * d + k] = out_val;
    }
}

struct WEIGHTS {
    float* W_DQ;
    float* W_UQ;
    float* W_DKV;
    float* W_UKV;
    float* WO;
};

void init_weights(WEIGHTS &w) {
    w.W_DQ = (float*)malloc(DIM * QPROJ_DIM * sizeof(float));
    w.W_UQ = (float*)malloc(QPROJ_DIM * DIM * sizeof(float));
    w.W_DKV = (float*)malloc(DIM * KVPROJ_DIM * sizeof(float));
    w.W_UKV = (float*)malloc(KVPROJ_DIM * DIM * 2 * sizeof(float));
    w.WO = (float*)malloc(DIM * DIM * sizeof(float));
    for (int i = 0; i < DIM * QPROJ_DIM; i++)
        w.W_DQ[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < QPROJ_DIM * DIM; i++)
        w.W_UQ[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < DIM * KVPROJ_DIM; i++)
        w.W_DKV[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < KVPROJ_DIM * DIM * 2; i++)
        w.W_UKV[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
    for (int i = 0; i < DIM * DIM; i++)
        w.WO[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
}

int main(){
    const int S = 64;
    const int B = 1;

    float* x_host = (float*)malloc(B * S * DIM * sizeof(float));
    for (int i = 0; i < B * S * DIM; i++) {
        x_host[i] = ((float)rand()/RAND_MAX - 0.5f);
    }

    WEIGHTS weights;
    init_weights(weights);

    float* x_dev;
    cudaMalloc(&x_dev, B * S * DIM * sizeof(float));
    cudaMemcpy(x_dev, x_host, B * S * DIM * sizeof(float), cudaMemcpyHostToDevice);

    float* compressed_q_dev;
    cudaMalloc(&compressed_q_dev, S * QPROJ_DIM * sizeof(float));
    float* Q_dev;
    cudaMalloc(&Q_dev, S * DIM * sizeof(float));

    float* compressed_kv_dev;
    cudaMalloc(&compressed_kv_dev, S * KVPROJ_DIM * sizeof(float));
    float* KV_dev;
    cudaMalloc(&KV_dev, S * DIM * 2 * sizeof(float));

    float* x_out_dev;
    cudaMalloc(&x_out_dev, S * DIM * sizeof(float));

    float* W_DQ_dev;
    cudaMalloc(&W_DQ_dev, DIM * QPROJ_DIM * sizeof(float));
    cudaMemcpy(W_DQ_dev, weights.W_DQ, DIM * QPROJ_DIM * sizeof(float), cudaMemcpyHostToDevice);
    {
        dim3 threads(16, 16);
        dim3 blocks((QPROJ_DIM + threads.x - 1) / threads.x, (S + threads.y - 1) / threads.y);
        matmul_kernel<<<blocks, threads>>>(x_dev, W_DQ_dev, compressed_q_dev, S, QPROJ_DIM, DIM);
        cudaDeviceSynchronize();
    }
    layernorm_kernel<<<S, 1>>>(compressed_q_dev, compressed_q_dev, QPROJ_DIM, 1e-5);
    cudaDeviceSynchronize();
    float* W_UQ_dev;
    cudaMalloc(&W_UQ_dev, QPROJ_DIM * DIM * sizeof(float));
    cudaMemcpy(W_UQ_dev, weights.W_UQ, QPROJ_DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);
    {
        dim3 threads(16, 16);
        dim3 blocks((DIM + threads.x - 1) / threads.x, (S + threads.y - 1) / threads.y);
        matmul_kernel<<<blocks, threads>>>(compressed_q_dev, W_UQ_dev, Q_dev, S, DIM, QPROJ_DIM);
        cudaDeviceSynchronize();
    }

    float* W_DKV_dev;
    cudaMalloc(&W_DKV_dev, DIM * KVPROJ_DIM * sizeof(float));
    cudaMemcpy(W_DKV_dev, weights.W_DKV, DIM * KVPROJ_DIM * sizeof(float), cudaMemcpyHostToDevice);
    {
        dim3 threads(16, 16);
        dim3 blocks((KVPROJ_DIM + threads.x - 1) / threads.x, (S + threads.y - 1) / threads.y);
        matmul_kernel<<<blocks, threads>>>(x_dev, W_DKV_dev, compressed_kv_dev, S, KVPROJ_DIM, DIM);
        cudaDeviceSynchronize();
    }
    layernorm_kernel<<<S, 1>>>(compressed_kv_dev, compressed_kv_dev, KVPROJ_DIM, 1e-5);
    cudaDeviceSynchronize();
    float* W_UKV_dev;
    cudaMalloc(&W_UKV_dev, KVPROJ_DIM * DIM * 2 * sizeof(float));
    cudaMemcpy(W_UKV_dev, weights.W_UKV, KVPROJ_DIM * DIM * 2 * sizeof(float), cudaMemcpyHostToDevice);
    {
        dim3 threads(16, 16);
        dim3 blocks((DIM * 2 + threads.x - 1) / threads.x, (S + threads.y - 1) / threads.y);
        matmul_kernel<<<blocks, threads>>>(compressed_kv_dev, W_UKV_dev, KV_dev, S, DIM * 2, KVPROJ_DIM);
        cudaDeviceSynchronize();
    }
    float* K_dev = KV_dev;
    float* V_dev = KV_dev + S * DIM;

    float* head_outputs_dev;
    cudaMalloc(&head_outputs_dev, NHEADS * S * HEADDIM * sizeof(float));

    for (int h = 0; h < NHEADS; h++) {
        float* Q_head = Q_dev + h * HEADDIM;
        float* K_head = K_dev + h * HEADDIM;
        float* V_head = V_dev + h * HEADDIM;
        float* out_head = head_outputs_dev + h * S * HEADDIM;
        attention_kernel<<<S, 32, S * sizeof(float)>>>(Q_head, K_head, V_head, out_head, S, HEADDIM);
        cudaDeviceSynchronize();
    }

    float* head_outputs_host = (float*)malloc(NHEADS * S * HEADDIM * sizeof(float));
    cudaMemcpy(head_outputs_host, head_outputs_dev, NHEADS * S * HEADDIM * sizeof(float), cudaMemcpyDeviceToHost);
    float* x_out_host = (float*)malloc(S * DIM * sizeof(float));
    for (int s = 0; s < S; s++){
        for (int h = 0; h < NHEADS; h++){
            for (int i = 0; i < HEADDIM; i++){
                x_out_host[s * DIM + h * HEADDIM + i] = head_outputs_host[h * S * HEADDIM + s * HEADDIM + i];
            }
        }
    }
    cudaMemcpy(x_out_dev, x_out_host, S * DIM * sizeof(float), cudaMemcpyHostToDevice);

    float* WO_dev;
    cudaMalloc(&WO_dev, DIM * DIM * sizeof(float));
    cudaMemcpy(WO_dev, weights.WO, DIM * DIM * sizeof(float), cudaMemcpyHostToDevice);
    float* final_out_dev;
    cudaMalloc(&final_out_dev, S * DIM * sizeof(float));
    {
        dim3 threads(16, 16);
        dim3 blocks((DIM + threads.x - 1) / threads.x, (S + threads.y - 1) / threads.y);
        matmul_kernel<<<blocks, threads>>>(x_out_dev, WO_dev, final_out_dev, S, DIM, DIM);
        cudaDeviceSynchronize();
    }

    float* final_out_host = (float*)malloc(S * DIM * sizeof(float));
    cudaMemcpy(final_out_host, final_out_dev, S * DIM * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Final output (first row):\n";
    for (int i = 0; i < DIM; i++){
        std::cout << final_out_host[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(x_dev);
    cudaFree(compressed_q_dev);
    cudaFree(Q_dev);
    cudaFree(compressed_kv_dev);
    cudaFree(KV_dev);
    cudaFree(x_out_dev);
    cudaFree(W_DQ_dev);
    cudaFree(W_UQ_dev);
    cudaFree(W_DKV_dev);
    cudaFree(W_UKV_dev);
    cudaFree(WO_dev);
    cudaFree(final_out_dev);
    cudaFree(head_outputs_dev);
    free(x_host);
    free(head_outputs_host);
    free(x_out_host);
    free(final_out_host);
    free(weights.W_DQ);
    free(weights.W_UQ);
    free(weights.W_DKV);
    free(weights.W_UKV);
    free(weights.WO);
    return 0;
}
