#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel: performs one gradient descent update for a linear regression weight vector.
// It uses the previous weight vector w_old (read-only) to compute predictions and then writes
// the updated weights to w. Each thread computes the gradient and update for one weight.
__global__ void gradientDescentKernel(const float *x,     // training features, stored row-major (n x d)
                                      const float *y,     // training labels (n)
                                      const float *w_old, // current weight vector (d)
                                      float *w,           // output: updated weight vector (d)
                                      float alpha,        // learning rate
                                      int n,              // number of training examples
                                      int d)              // number of features
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < d)
    {
        float grad = 0.0f;
        for (int i = 0; i < n; i++)
        {
            float prediction = 0.0f;
            for (int k = 0; k < d; k++)
            {
                prediction += w_old[k] * x[i * d + k];
            }
            grad += (prediction - y[i]) * x[i * d + j];
        }
        grad /= n;
        w[j] = w_old[j] - alpha * grad;
    }
}

int main()
{
    int n = 100;
    int d = 3;

    //  the “true” model: y = 1*x0 + 2*x1 + 3*x2.)
    float *h_x = new float[n * d];
    float *h_y = new float[n];
    float *h_w = new float[d];

    // compute the data
    for (int i = 0; i < n; i++)
    {
        h_x[i * d + 0] = 1.0f;
        h_x[i * d + 1] = 2.0f;
        h_x[i * d + 2] = 3.0f;

        h_y[i] = 14.0f;
    }
    // weights
    for (int j = 0; j < d; j++)
    {
        h_w[j] = 0.0f;
    }

    // Allocate device memory
    float *d_x, *d_y, *d_w, *d_w_old;
    cudaMalloc((void **)&d_x, n * d * sizeof(float));
    cudaMalloc((void **)&d_y, n * sizeof(float));
    cudaMalloc((void **)&d_w, d * sizeof(float));
    cudaMalloc((void **)&d_w_old, d * sizeof(float));
    cudaMemcpy(d_x, h_x, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, d * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 0.01f; // learning rate
    int iterations = 10000;

    int threadsPerBlock = 256;
    int blocksPerGrid = (d + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < iterations; iter++)
    {
        cudaMemcpy(d_w_old, d_w, d * sizeof(float), cudaMemcpyDeviceToDevice);
        gradientDescentKernel<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_w_old, d_w, alpha, n, d);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_w, d_w, d * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Learned weights:" << std::endl;
    for (int j = 0; j < d; j++)
    {
        std::cout << "w[" << j << "] = " << h_w[j] << std::endl;
    }

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_w);
    cudaFree(d_w_old);
    delete[] h_x;
    delete[] h_y;
    delete[] h_w;

    return 0;
}
