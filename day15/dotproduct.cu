#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define sharedSize 32


__global__ void dotProduct(float *Q, float *K, float *O, int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    __shared__ float Qs[sharedSize][sharedSize];
    __shared__ float Ks[sharedSize][sharedSize];
    
    float sum = 0.0f;
    
    int numTiles = (width + sharedSize - 1) / sharedSize;
    
    for (int t = 0; t < numTiles; ++t) {
        int qCol = t * sharedSize + tx;
        if (row < height && qCol < width)
            Qs[ty][tx] = Q[row * width + qCol];
        else
            Qs[ty][tx] = 0.0f;
        
        int kRow = t * sharedSize + ty;
        int kCol = col;
        if (kRow < width && kCol < width)
            Ks[ty][tx] = K[kRow * width + kCol];
        else
            Ks[ty][tx] = 0.0f;
        
        __syncthreads();
        
        for (int k = 0; k < sharedSize; ++k) {
            sum += Qs[ty][k] * Ks[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < height && col < width)
        O[row * width + col] = sum;
}

void dotProductCPU(const std::vector<float>& Q, const std::vector<float>& K, std::vector<float>& O, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0;
            for (int k = 0; k < width; k++) {
                sum += Q[i * width + k] * K[k * width + j];
            }
            O[i * width + j] = sum;
        }
    }
}

int main() {
    int height = 1024, width = 1024;
    size_t size = height * width * sizeof(float);
    
    std::vector<float> h_Q(height * width, 1.0f);
    std::vector<float> h_K(height * width, 1.0f);
    std::vector<float> h_O_cpu(height * width, 0);
    std::vector<float> h_O_gpu(height * width, 0);

    float *d_Q, *d_K, *d_O;
    CHECK_CUDA(cudaMalloc((void**)&d_Q, size));
    CHECK_CUDA(cudaMalloc((void**)&d_K, size));
    CHECK_CUDA(cudaMalloc((void**)&d_O, size));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), size, cudaMemcpyHostToDevice));

    dim3 blockSize(sharedSize, sharedSize);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    auto start_cpu = std::chrono::high_resolution_clock::now();
    dotProductCPU(h_Q, h_K, h_O_cpu, height, width);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_time = end_cpu - start_cpu;
    std::cout << "CPU Time: " << cpu_time.count() << " ms" << std::endl;

    auto start_gpu = std::chrono::high_resolution_clock::now();
    dotProduct<<<gridSize, blockSize>>>(d_Q, d_K, d_O, height, width);
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> gpu_time = end_gpu - start_gpu;
    std::cout << "GPU Time: " << gpu_time.count() << " ms" << std::endl;

    CHECK_CUDA(cudaMemcpy(h_O_gpu.data(), d_O, size, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < height * width; i++) {
        if (fabs(h_O_cpu[i] - h_O_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }

    std::cout << "Results match: " << (correct ? "Yes" : "No") << std::endl;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_O);

    return 0;
}
