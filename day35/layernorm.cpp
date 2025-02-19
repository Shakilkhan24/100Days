#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define HIP_CALL(call)                                                          \
    {                                                                           \
        hipError_t err = call;                                                  \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err),   \
                    __FILE__, __LINE__);                                        \
            exit(err);                                                          \
        }                                                                       \
    }

__global__ void layernorm_forward(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  const float* __restrict__ gamma,
                                  const float* __restrict__ beta,
                                  float* __restrict__ mean,
                                  float* __restrict__ variance,
                                  const int D,
                                  const float eps) {
    const int sample = blockIdx.x;
    const int tid = threadIdx.x;
    float sum = 0.0f, sum_sq = 0.0f;

    for (int i = tid; i < D; i += blockDim.x) {
        float val = input[sample * D + i];
        sum += val;
        sum_sq += val * val;
    }

    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];
    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float sample_mean = s_sum[0] / D;
    float sample_variance = s_sum_sq[0] / D - sample_mean * sample_mean;

    if (tid == 0) {
        mean[sample] = sample_mean;
        variance[sample] = sample_variance;
    }
    __syncthreads();

    float inv_std = rsqrtf(sample_variance + eps);
    for (int i = tid; i < D; i += blockDim.x) {
        float val = input[sample * D + i];
        float xhat = (val - sample_mean) * inv_std;
        output[sample * D + i] = xhat * gamma[i] + beta[i];
    }
}

__global__ void layernorm_backward(const float* __restrict__ input,
                                   const float* __restrict__ dout,
                                   const float* __restrict__ gamma,
                                   const float* __restrict__ mean,
                                   const float* __restrict__ variance,
                                   float* __restrict__ dx,
                                   float* __restrict__ dgamma,
                                   float* __restrict__ dbeta,
                                   const int D,
                                   const float eps) {
    const int sample = blockIdx.x;
    const int tid = threadIdx.x;
    float sample_mean = mean[sample];
    float sample_var = variance[sample];
    float inv_std = rsqrtf(sample_var + eps);

    float sum_dY = 0.0f, sum_dY_xhat = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float val = input[sample * D + i];
        float xhat = (val - sample_mean) * inv_std;
        float dy = dout[sample * D + i];
        sum_dY += dy;
        sum_dY_xhat += dy * xhat;
    }

    __shared__ float s_sum_dY[256];
    __shared__ float s_sum_dY_xhat[256];
    s_sum_dY[tid] = sum_dY;
    s_sum_dY_xhat[tid] = sum_dY_xhat;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            s_sum_dY[tid] += s_sum_dY[tid + s];
            s_sum_dY_xhat[tid] += s_sum_dY_xhat[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        sum_dY = s_sum_dY[0];
        sum_dY_xhat = s_sum_dY_xhat[0];
    }
    __syncthreads();

    for (int i = tid; i < D; i += blockDim.x) {
        float val = input[sample * D + i];
        float xhat = (val - sample_mean) * inv_std;
        float dy = dout[sample * D + i];
        float dx_val = inv_std * (dy - (sum_dY / D) - xhat * (sum_dY_xhat / D));
        dx[sample * D + i] = dx_val;

        atomicAdd(&dgamma[i], dy * xhat);
        atomicAdd(&dbeta[i], dy);
    }
}

int main() {
    const int N = 2;
    const int D = 8;
    const float eps = 1e-5f;

    float h_input[N * D];
    float h_gamma[D];
    float h_beta[D];
    float h_out[N * D];
    float h_mean[N];
    float h_variance[N];
    float h_dout[N * D];
    float h_dx[N * D];
    float h_dgamma[D];
    float h_dbeta[D];

    for (int i = 0; i < N * D; ++i) {
        h_input[i] = float(i % 7) - 3.0f;
        h_dout[i]  = 1.0f;
    }
    for (int i = 0; i < D; ++i) {
        h_gamma[i] = 1.0f;
        h_beta[i]  = 0.0f;
        h_dgamma[i] = 0.0f;
        h_dbeta[i]  = 0.0f;
    }

    float *d_input, *d_gamma, *d_beta, *d_out, *d_mean, *d_variance;
    float *d_dout, *d_dx, *d_dgamma, *d_dbeta;
    HIP_CALL(hipMalloc(&d_input,     N * D * sizeof(float)));
    HIP_CALL(hipMalloc(&d_gamma,     D * sizeof(float)));
    HIP_CALL(hipMalloc(&d_beta,      D * sizeof(float)));
    HIP_CALL(hipMalloc(&d_out,       N * D * sizeof(float)));
    HIP_CALL(hipMalloc(&d_mean,      N * sizeof(float)));
    HIP_CALL(hipMalloc(&d_variance,  N * sizeof(float)));
    HIP_CALL(hipMalloc(&d_dout,      N * D * sizeof(float)));
    HIP_CALL(hipMalloc(&d_dx,        N * D * sizeof(float)));
    HIP_CALL(hipMalloc(&d_dgamma,    D * sizeof(float)));
    HIP_CALL(hipMalloc(&d_dbeta,     D * sizeof(float)));

    HIP_CALL(hipMemcpy(d_input,    h_input,    N * D * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_gamma,    h_gamma,    D * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_beta,     h_beta,     D * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(d_dout,     h_dout,     N * D * sizeof(float), hipMemcpyHostToDevice));
    HIP_CALL(hipMemset(d_dgamma, 0, D * sizeof(float)));
    HIP_CALL(hipMemset(d_dbeta,  0, D * sizeof(float)));

    const int blockSize = 256;
    dim3 gridForward(N);
    dim3 gridBackward(N);

    hipLaunchKernelGGL(layernorm_forward, gridForward, blockSize, 0, 0,
                       d_input, d_out, d_gamma, d_beta, d_mean, d_variance, D, eps);
    HIP_CALL(hipGetLastError());

    hipLaunchKernelGGL(layernorm_backward, gridBackward, blockSize, 0, 0,
                       d_input, d_dout, d_gamma, d_mean, d_variance,
                       d_dx, d_dgamma, d_dbeta, D, eps);
    HIP_CALL(hipGetLastError());

    HIP_CALL(hipMemcpy(h_out,       d_out,       N * D * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(h_mean,      d_mean,      N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(h_variance,  d_variance,  N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(h_dx,        d_dx,        N * D * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(h_dgamma,    d_dgamma,    D * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(h_dbeta,     d_dbeta,     D * sizeof(float), hipMemcpyDeviceToHost));

    printf("Forward pass results:\n");
    for (int n = 0; n < N; ++n) {
        printf("Sample %d: mean = %.4f, variance = %.4f\n", n, h_mean[n], h_variance[n]);
        printf("Output: ");
        for (int i = 0; i < D; ++i) {
            printf("%.4f ", h_out[n * D + i]);
        }
        printf("\n");
    }

    printf("\nBackward pass results:\n");
    printf("dx:\n");
    for (int n = 0; n < N; ++n) {
        printf("Sample %d: ", n);
        for (int i = 0; i < D; ++i) {
            printf("%.4f ", h_dx[n * D + i]);
        }
        printf("\n");
    }
    printf("dgamma: ");
    for (int i = 0; i < D; ++i) {
        printf("%.4f ", h_dgamma[i]);
    }
    printf("\n");
    printf("dbeta: ");
    for (int i = 0; i < D; ++i) {
        printf("%.4f ", h_dbeta[i]);
    }
    printf("\n");

    hipFree(d_input);
    hipFree(d_gamma);
    hipFree(d_beta);
    hipFree(d_out);
    hipFree(d_mean);
    hipFree(d_variance);
    hipFree(d_dout);
    hipFree(d_dx);
    hipFree(d_dgamma);
    hipFree(d_dbeta);

    return 0;
}
