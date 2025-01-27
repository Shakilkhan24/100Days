#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <iomanip>
#include <fstream>
__global__ void forward(int batch_size, int n, int out_w,
                        float *input, float *weights, float *biases, float *output)
{
    int collumn = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && collumn < out_w)
    {
        output[row * out_w + collumn] = biases[collumn];
        for (int i = 0; i < n; ++i)
        {
            output[row * out_w + collumn] += input[row * n + i] * weights[i * out_w + collumn];
        }
    }
}

__global__ void backward(int batch_size, int n, int out_w,
                         float *weights, float *biases, float *d_l, float *out_d_l)
{
    int collumn = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < batch_size && collumn < out_w)
    {
        float dl = 0;
        for (int k = 0; k < n; ++k)
        {
            dl += weights[k * out_w + collumn] * d_l[row * n + k];
        }
        out_d_l[row * out_w + collumn] = dl;
    }
}

__global__ void relu(int n, int k, float *input, float *output)
{
    int collumn = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && collumn < k)
    {
        output[row * k + collumn] = output[row * k + collumn] > 0.0f ? output[row * k + collumn] : 0.0f;
    }
}

__global__ void relu_backward(int n, int k, float *input, float *dl, float *dl_out)
{
    int collumn = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && collumn < k)
    {
        dl_out[row * k + collumn] = input[row * k + collumn] > 0.0f ? dl[row * k + collumn] : 0.0f;
    }
}

__global__ void softmax(int w, int h, float *input, float *output)
{
    int collumn = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < w && collumn < h)
    {
        float maxval = input[row * w];
        for (int k = 0; k < w; ++k)
        {
            maxval = fmaxf(maxval, input[row * w + k]);
        }
        __syncthreads();
        float sumval = 0;
        for (int k = 0; k < w; ++k)
        {
            sumval += expf(input[row * w + k] - maxval);
        }
        __syncthreads();

        output[row * w + collumn] = expf(output[row * w + collumn] - maxval) / sumval;
    }
}

__global__ void crossentropy(int w, int h, float *preds, float *real, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < h)
    {
        float loss = 0.0f;
        for (int i = 0; i < w; i++)
        {
            loss -= real[idx * w + i] * log(max(1e-6, preds[idx * w + i]));
        }
        output[idx] = loss;
    }
}

__global__ void crossentropy_backwards(int w, int h, float *preds, float *real, float *output)
{
    int collumn = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < w && collumn < h)
    {
        output[row * w + collumn] = preds[row * w + collumn] - real[row * w + collumn];
    }
}

__global__ void initRandom(int w, int h, float *mat)
{
    int collumn = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < w && collumn < h)
    {
        curandState state;
        curand_init(42, row * w + collumn, 0, &state);
        mat[row * w + collumn] = curand_normal(&state) * sqrtf(2.f / h);
    }
}

__global__ void gradientDescent(int w, int h, int batch_size, float lr,
                                float *weights, float *biases, float *activatoin, float *d_l)
{
    int collumn = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < w && collumn < h)
    {
        float dW = 0;
        float dB = 0;
        for (int i = 0; i < batch_size; ++i)
        {
            float act = activatoin[i * h + row];
            float dl = d_l[i * w + collumn];
            dW += act * dl;
            dB += dl;
        }
        weights[row * w + collumn] -= lr * dW / batch_size;
        biases[collumn] -= lr * dB / batch_size;
    }
}

void print_matrix(int w, int h, float *matrix, std::string title)
{
    float *m_h = new float[w * h];
    cudaMemcpy(m_h, matrix, w * h * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << title << std::endl;
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            std::cout << std::fixed << std::setprecision(3) << m_h[i * w + j] << ", ";
        }
        std::cout << std::endl;
    }
    free(m_h);
}

void initLayer(float *weights, float *biases, int w, int h, int BLOCK_SIZE)
{
    dim3 dimGrid = dim3(ceil(w / (float)BLOCK_SIZE), ceil(h / (float)BLOCK_SIZE), 1);
    dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
    initRandom<<<dimGrid, dimBlock>>>(w, h, weights);

    dimGrid = dim3(ceil(h / (float)BLOCK_SIZE), 1, 1);
    dimBlock = dim3(BLOCK_SIZE, 1, 1);
    initRandom<<<dimGrid, dimBlock>>>(1, h, biases);
}

void read_mnist(const std::string filename, int length, float* x, float* y)
{
  int input_size = 784;
  int labels = 10;

  std::fstream fin;
  fin.open(filename);
  std::string row;
  constexpr char delim = ',';
  for(int i = 0; i<length; i++)
  {
    fin >> row;
    int pos = row.find(delim);
    int label = std::stoi(row.substr(0, pos+1));
    for(int j = 0; j<labels; j++)
    {
      y[labels*i + j] = (j==label);
    }
    row.erase(0, pos+1);
    for(int j = 0; j<input_size; j++)
    {
      pos = row.find(delim);
      if (pos == std::string::npos)
      {
        pos = row.length() - 1;
      }
      x[i*input_size+j] = std::stof(row.substr(0, pos+1)) / 255; //normalize value
      row.erase(0, pos+1);
    }
  }
}
