#include <hip/hip_runtime.h>
#include <iostream>

#define CHECK_HIP_ERROR(error)                                     \
    {                                                              \
        if ((error) != hipSuccess)                                 \
        {                                                          \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at line " << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

#define BLOCK_SIZE 256
__global__ void kernel3_registers(float *a, float *b, float *c, int N, float alpha, float beta)
{
    // Block Tile size
    constexpr int BN = 128;
    constexpr int BM = 128;
    // Number of Row or column we read per batch
    constexpr int BK = 8;

    // Thread Tile size
    constexpr int TN = 4;
    constexpr int TM = 4;

    constexpr int nbWaves = BLOCK_SIZE / 32;
    // Wave Tile size 
    constexpr int WN = 64;
    constexpr int WM = BN * BM / nbWaves / WN;

    // Number of wave on X & Y axis in the Block tile
    constexpr int nbWaveX = BN / WN;
    constexpr int nbWaveY = BM / WM;

    const int waveIndex = threadIdx.x / 32;
    const int waveIdx = waveIndex % nbWaveX;
    const int waveIdy = waveIndex / nbWaveX;
    const int indexInWave = threadIdx.x % 32;

    // A wave is a block of 8x4 of the output matrix
    constexpr int nbThreadXPerWave = 8;
    constexpr int nbThreadYPerWave = 4;

    // Thread coordinates in Wave
    const int idxInWave = indexInWave % nbThreadXPerWave;
    const int idyInWave = indexInWave / nbThreadXPerWave;

    constexpr int nbIterWaveN = WN / (nbThreadXPerWave * TN);
    constexpr int nbIterWaveM = WM / (nbThreadYPerWave * TM);

    // Wave Sub-tile size
    constexpr int SUBWN = WN / nbIterWaveN;
    constexpr int SUBWM = WM / nbIterWaveM;

    // Thread mapping to read BKxBN block from A
    int rAIdx = threadIdx.x % BK;
    int rAIdy = threadIdx.x / BK;
    // Thread mapping to read BNxBK block from B
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

    // Iteration over BK blocks.
    for (int kId = 0; kId < N; kId += BK)
    {
        // We populate the Shared Memory with Ks row and columns
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
            // we cache A & B for the entire Wave tile
            for (int iterWave = 0; iterWave < nbIterWaveN; iterWave++)
            {
                for (int i = 0; i < TN; i++)
                {
                    int index = waveIdx * WN +     // waveId
                                iterWave * SUBWN + // wave subtile
                                TN * idxInWave +
                                +i;
                    B_row[iterWave * TN + i] = Bs[k][index];
                }
            }

            for (int iterWave = 0; iterWave < nbIterWaveM; iterWave++)
            {
                for (int i = 0; i < TM; i++)
                {
                    int index = waveIdy * WM +     // waveId
                                iterWave * SUBWM + // wave subtile
                                TM * idyInWave +
                                i;

                    A_col[iterWave * TM + i] = As[k][index];
                }
            }

            // we accumulate to C_regs
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
    const int N = 256; // Must be multiple of 128 for this kernel
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Host memory
    size_t size = N * N * sizeof(float);
    float* hA = (float*)malloc(size);
    float* hB = (float*)malloc(size);
    float* hC = (float*)malloc(size);

    // Initialize data
    for(int i = 0; i < N*N; i++)
    {
        hA[i] = static_cast<float>(i % 100);
        hB[i] = static_cast<float>((i * 2) % 100);
        hC[i] = 0.0f; // Initialize C to zero
    }

    // Device memory
    float *dA, *dB, *dC;
    CHECK_HIP_ERROR(hipMalloc(&dA, size));
    CHECK_HIP_ERROR(hipMalloc(&dB, size));
    CHECK_HIP_ERROR(hipMalloc(&dC, size));

    CHECK_HIP_ERROR(hipMemcpy(dA, hA, size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB, size, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dC, hC, size, hipMemcpyHostToDevice));

    // Kernel launch
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid(N / 128, N / 128);
    hipLaunchKernelGGL(kernel3_registers, blocksPerGrid, threadsPerBlock, 0, 0, dA, dB, dC, N, alpha, beta);
    CHECK_HIP_ERROR(hipGetLastError());

    // Copy back and check a small portion
    CHECK_HIP_ERROR(hipMemcpy(hC, dC, size, hipMemcpyDeviceToHost));

    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            std::cout << hC[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    CHECK_HIP_ERROR(hipFree(dA));
    CHECK_HIP_ERROR(hipFree(dB));
    CHECK_HIP_ERROR(hipFree(dC));
    free(hA);
    free(hB);
    free(hC);

    return 0;
}