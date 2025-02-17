#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <limits>

#define CHECK_HIP_ERROR(error)                                     \
    {                                                              \
        if ((error) != hipSuccess)                                 \
        {                                                          \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at line " << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

#define CHECK_ROCBLAS_ERROR(error)                               \
    {                                                            \
        if ((error) != rocblas_status_success)                   \
        {                                                        \
            std::cerr << "rocBLAS error at line " << __LINE__    \
                      << ": " << rocblas_status_to_string(error) \
                      << std::endl;                              \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    }

#define BLOCK_SIZE 256

__global__ void kernel3_registers(float *a, float *b, float *c, int N, float alpha, float beta)
{
    constexpr int BN = 128;
    constexpr int BM = 128;
    constexpr int BK = 8;

    constexpr int TN = 4;
    constexpr int TM = 4;

    constexpr int nbWaves = BLOCK_SIZE / 32;
    constexpr int WN = 64;
    constexpr int WM = BN * BM / nbWaves / WN;

    constexpr int nbWaveX = BN / WN;
    constexpr int nbWaveY = BM / WM;

    const int waveIndex = threadIdx.x / 32;
    const int waveIdx = waveIndex % nbWaveX;
    const int waveIdy = waveIndex / nbWaveX;
    const int indexInWave = threadIdx.x % 32;

    constexpr int nbThreadXPerWave = 8;
    constexpr int nbThreadYPerWave = 4;

    const int idxInWave = indexInWave % nbThreadXPerWave;
    const int idyInWave = indexInWave / nbThreadXPerWave;

    constexpr int nbIterWaveN = WN / (nbThreadXPerWave * TN);
    constexpr int nbIterWaveM = WM / (nbThreadYPerWave * TM);

    constexpr int SUBWN = WN / nbIterWaveN;
    constexpr int SUBWM = WM / nbIterWaveM;

    int rAIdx = threadIdx.x % BK;
    int rAIdy = threadIdx.x / BK;
    int rBIdx = threadIdx.x % BN;
    int rBIdy = threadIdx.x / BN;

    constexpr int strideReadB = BLOCK_SIZE / BN;
    constexpr int strideReadA = BLOCK_SIZE / BK;
    constexpr int nbReadsB = BN * BK / BLOCK_SIZE;
    constexpr int nbReadsA = BM * BK / BLOCK_SIZE;

    float A_col[nbIterWaveM * TM];
    float B_row[nbIterWaveN * TN];

    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    float c_regs[TM * nbIterWaveM * TN * nbIterWaveN] = {0.0f};

    for (int kId = 0; kId < N; kId += BK)
    {
        for (int i = 0; i < nbReadsB; i++)
        {
            int index_x = BN * blockIdx.x + rBIdx;
            int index_y = rBIdy + i * strideReadB + kId;
            Bs[index_y % BK][index_x % BN] = b[N * index_y + index_x];
        }

        for (int i = 0; i < nbReadsA; i++)
        {
            int index_x = rAIdx + kId;
            int index_y = BM * blockIdx.y + rAIdy + i * strideReadA;
            As[(index_x % BK)][(index_y % BM)] = a[N * index_y + index_x];
        }

        __syncthreads();
        for (int k = 0; k < BK; k += 1)
        {
            for (int iterWave = 0; iterWave < nbIterWaveN; iterWave++)
            {
                for (int i = 0; i < TN; i++)
                {
                    int index = waveIdx * WN + iterWave * SUBWN + TN * idxInWave + i;
                    B_row[iterWave * TN + i] = Bs[k][index];
                }
            }

            for (int iterWave = 0; iterWave < nbIterWaveM; iterWave++)
            {
                for (int i = 0; i < TM; i++)
                {
                    int index = waveIdy * WM + iterWave * SUBWM + TM * idyInWave + i;
                    A_col[iterWave * TM + i] = As[k][index];
                }
            }

            for (int iterWaveM = 0; iterWaveM < nbIterWaveM; iterWaveM++)
            {
                for (int iterWaveN = 0; iterWaveN < nbIterWaveN; iterWaveN++)
                {
                    for (int yt = 0; yt < TM; yt++)
                    {
                        for (int xt = 0; xt < TN; xt++)
                        {
                            const int x = iterWaveN * TN + xt;
                            const int y = iterWaveM * TM + yt;
                            c_regs[y * TN * nbIterWaveN + x] += A_col[y] * B_row[x];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    for (int iterWaveM = 0; iterWaveM < nbIterWaveM; iterWaveM++)
    {
        for (int iterWaveN = 0; iterWaveN < nbIterWaveN; iterWaveN++)
        {
            int xOut = blockIdx.x * BN + waveIdx * WN + iterWaveN * SUBWN + TN * idxInWave;
            int yOut = blockIdx.y * BM + waveIdy * WM + iterWaveM * SUBWM + TM * idyInWave;
            for (int yt = 0; yt < TM; yt++)
            {
                for (int xt = 0; xt < TN; xt++)
                {
                    int indexC = N * (yOut + yt) + xOut + xt;
                    c[indexC] = beta * c[indexC] + alpha * c_regs[TN * nbIterWaveN * (iterWaveM * TM + yt) + (iterWaveN * TN + xt)];
                }
            }
        }
    }
}

int main()
{
    const int N = 2560;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t size = N * N * sizeof(float);
    float *hA = (float *)malloc(size);
    float *hB = (float *)malloc(size);
    float *hC = (float *)malloc(size);
    float *hC_ref = (float *)malloc(size);

    for (int i = 0; i < N * N; i++)
    {
        hA[i] = static_cast<float>(i % 100);
        hB[i] = static_cast<float>((i * 2) % 100);
        hC[i] = 0.0f;
        hC_ref[i] = 0.0f;
    }

    float *dA, *dB, *dC;
    CHECK_HIP_ERROR(hipMalloc(&dA, size));
    CHECK_HIP_ERROR(hipMalloc(&dB, size));
    CHECK_HIP_ERROR(hipMalloc(&dC, size));

    CHECK_HIP_ERROR(hipMemcpy(dA, hA, size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC, size, hipMemcpyHostToDevice));

    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(N / 128, N / 128);

    float min_time_custom = std::numeric_limits<float>::max();
    for(int i = 0; i < 5; i++)
    {
        CHECK_HIP_ERROR(hipMemcpy(dC, hC, size, hipMemcpyHostToDevice));

        auto start_time_custom = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(kernel3_registers, blocksPerGrid, threadsPerBlock, 0, 0, dA, dB, dC, N, alpha, beta);
        CHECK_HIP_ERROR(hipGetLastError());
        CHECK_HIP_ERROR(hipMemcpy(hC, dC, size, hipMemcpyDeviceToHost));
        auto end_time_custom = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float> duration_custom = end_time_custom - start_time_custom;
        if (duration_custom.count() < min_time_custom)
        {
            min_time_custom = duration_custom.count();
        }
    }

    float min_time_rocblas = std::numeric_limits<float>::max();
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    for(int i = 0; i < 5; i++)
    {
        CHECK_HIP_ERROR(hipMemcpy(dC, hC_ref, size, hipMemcpyHostToDevice));

        auto start_time_rocblas = std::chrono::high_resolution_clock::now();
        rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                      N, N, N, &alpha, dA, N, dB, N, &beta, dC, N);
        CHECK_ROCBLAS_ERROR(rocblas_get_matrix(N, N, sizeof(float), dC, N, hC_ref, N));
        auto end_time_rocblas = std::chrono::high_resolution_clock::now();

        std::chrono::duration<float> duration_rocblas = end_time_rocblas - start_time_rocblas;
        if (duration_rocblas.count() < min_time_rocblas)
        {
            min_time_rocblas = duration_rocblas.count();
        }
    }

    std::cout << "Our kernel execution time: " << min_time_custom << " seconds" << std::endl;
    std::cout << "rocBLAS kernel execution time: " << min_time_rocblas << " seconds" << std::endl;

    std::cout << "Ratio of execution times: " << min_time_rocblas / min_time_custom * 100 <<" %"<< std::endl;

    free(hA);
    free(hB);
    free(hC);
    free(hC_ref);
    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));
    rocblas_destroy_handle(handle);

    return 0;
}
