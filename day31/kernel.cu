#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define WIDTH 512
#define HEIGHT 512
#define TILE_SIZE 16
#define DU 0.16f
#define DV 0.08f
#define F  0.060f
#define K  0.062f
#define DT 1.0f

__global__ void reactionDiffusionStep(const float* u, const float* v,
                                      float* u_new, float* v_new,
                                      int width, int height,
                                      float Du, float Dv,
                                      float F, float K, float dt) {
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    __shared__ float su[TILE_SIZE + 2][TILE_SIZE + 2];
    __shared__ float sv[TILE_SIZE + 2][TILE_SIZE + 2];

    int lx = threadIdx.x + 1;
    int ly = threadIdx.y + 1;

    if (x < width && y < height) {
        su[ly][lx] = u[y * width + x];
        sv[ly][lx] = v[y * width + x];
    } else {
        su[ly][lx] = 0.0f;
        sv[ly][lx] = 0.0f;
    }

    if (threadIdx.x == 0) {
        int x_left = x - 1;
        if (x_left >= 0 && y < height) {
            su[ly][lx - 1] = u[y * width + x_left];
            sv[ly][lx - 1] = v[y * width + x_left];
        } else {
            su[ly][lx - 1] = 0.0f;
            sv[ly][lx - 1] = 0.0f;
        }
    }
    if (threadIdx.x == TILE_SIZE - 1) {
        int x_right = x + 1;
        if (x_right < width && y < height) {
            su[ly][lx + 1] = u[y * width + x_right];
            sv[ly][lx + 1] = v[y * width + x_right];
        } else {
            su[ly][lx + 1] = 0.0f;
            sv[ly][lx + 1] = 0.0f;
        }
    }
    if (threadIdx.y == 0) {
        int y_top = y - 1;
        if (y_top >= 0 && x < width) {
            su[ly - 1][lx] = u[y_top * width + x];
            sv[ly - 1][lx] = v[y_top * width + x];
        } else {
            su[ly - 1][lx] = 0.0f;
            sv[ly - 1][lx] = 0.0f;
        }
    }
    if (threadIdx.y == TILE_SIZE - 1) {
        int y_bottom = y + 1;
        if (y_bottom < height && x < width) {
            su[ly + 1][lx] = u[y_bottom * width + x];
            sv[ly + 1][lx] = v[y_bottom * width + x];
        } else {
            su[ly + 1][lx] = 0.0f;
            sv[ly + 1][lx] = 0.0f;
        }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        int x_left = x - 1, y_top = y - 1;
        if (x_left >= 0 && y_top >= 0) {
            su[ly - 1][lx - 1] = u[y_top * width + x_left];
            sv[ly - 1][lx - 1] = v[y_top * width + x_left];
        } else {
            su[ly - 1][lx - 1] = 0.0f;
            sv[ly - 1][lx - 1] = 0.0f;
        }
    }
    if (threadIdx.x == TILE_SIZE - 1 && threadIdx.y == 0) {
        int x_right = x + 1, y_top = y - 1;
        if (x_right < width && y_top >= 0) {
            su[ly - 1][lx + 1] = u[y_top * width + x_right];
            sv[ly - 1][lx + 1] = v[y_top * width + x_right];
        } else {
            su[ly - 1][lx + 1] = 0.0f;
            sv[ly - 1][lx + 1] = 0.0f;
        }
    }
    if (threadIdx.x == 0 && threadIdx.y == TILE_SIZE - 1) {
        int x_left = x - 1, y_bottom = y + 1;
        if (x_left >= 0 && y_bottom < height) {
            su[ly + 1][lx - 1] = u[y_bottom * width + x_left];
            sv[ly + 1][lx - 1] = v[y_bottom * width + x_left];
        } else {
            su[ly + 1][lx - 1] = 0.0f;
            sv[ly + 1][lx - 1] = 0.0f;
        }
    }
    if (threadIdx.x == TILE_SIZE - 1 && threadIdx.y == TILE_SIZE - 1) {
        int x_right = x + 1, y_bottom = y + 1;
        if (x_right < width && y_bottom < height) {
            su[ly + 1][lx + 1] = u[y_bottom * width + x_right];
            sv[ly + 1][lx + 1] = v[y_bottom * width + x_right];
        } else {
            su[ly + 1][lx + 1] = 0.0f;
            sv[ly + 1][lx + 1] = 0.0f;
        }
    }
    
    __syncthreads();

    if (x < width && y < height) {
        float lap_u = su[ly][lx - 1] + su[ly][lx + 1] +
                      su[ly - 1][lx] + su[ly + 1][lx] -
                      4.0f * su[ly][lx];
        float lap_v = sv[ly][lx - 1] + sv[ly][lx + 1] +
                      sv[ly - 1][lx] + sv[ly + 1][lx] -
                      4.0f * sv[ly][lx];

        float u_val = su[ly][lx];
        float v_val = sv[ly][lx];

        float uvv = u_val * v_val * v_val;

        float u_next = u_val + (Du * lap_u - uvv + F * (1.0f - u_val)) * dt;
        float v_next = v_val + (Dv * lap_v + uvv - (F + K) * v_val) * dt;

        u_next = (u_next < 0.0f) ? 0.0f : ((u_next > 1.0f) ? 1.0f : u_next);
        v_next = (v_next < 0.0f) ? 0.0f : ((v_next > 1.0f) ? 1.0f : v_next);

        u_new[y * width + x] = u_next;
        v_new[y * width + x] = v_next;
    }
}

int main() {
    int width = WIDTH;
    int height = HEIGHT;
    int gridSize = width * height;
    size_t memSize = gridSize * sizeof(float);

    float* h_u = (float*)malloc(memSize);
    float* h_v = (float*)malloc(memSize);

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            h_u[j * width + i] = 1.0f;
            h_v[j * width + i] = 0.0f;
        }
    }
    int seedSize = 20;
    for (int j = height / 2 - seedSize / 2; j < height / 2 + seedSize / 2; j++) {
        for (int i = width / 2 - seedSize / 2; i < width / 2 + seedSize / 2; i++) {
            h_u[j * width + i] = 0.50f;
            h_v[j * width + i] = 0.25f;
        }
    }

    float *d_u, *d_v, *d_u_new, *d_v_new;
    cudaMalloc((void**)&d_u, memSize);
    cudaMalloc((void**)&d_v, memSize);
    cudaMalloc((void**)&d_u_new, memSize);
    cudaMalloc((void**)&d_v_new, memSize);

    cudaMemcpy(d_u, h_u, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, memSize, cudaMemcpyHostToDevice);

    float dt = DT;
    float du = DU;
    float dv = DV;
    float f  = F;
    float k  = K;

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    int numIterations = 1000;
    for (int iter = 0; iter < numIterations; iter++) {
        reactionDiffusionStep<<<gridDim, blockDim>>>(d_u, d_v, d_u_new, d_v_new,
                                                     width, height, du, dv, f, k, dt);
        cudaDeviceSynchronize();

        float *temp = d_u;
        d_u = d_u_new;
        d_u_new = temp;

        temp = d_v;
        d_v = d_v_new;
        d_v_new = temp;
    }

    cudaMemcpy(h_u, d_u, memSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, memSize, cudaMemcpyDeviceToHost);

    int center = (height / 2) * width + (width / 2);
    printf("After %d iterations:\n", numIterations);
    printf("Center u: %f\n", h_u[center]);
    printf("Center v: %f\n", h_v[center]);

    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_u_new);
    cudaFree(d_v_new);
    free(h_u);
    free(h_v);

    return 0;
}
