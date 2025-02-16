#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <cstdlib>

// Macro to check HIP errors.
#define CHECK_HIP_ERROR(error)                                              \
    {                                                                       \
        if ((error) != hipSuccess) {                                        \
            std::cerr << "HIP error: " << hipGetErrorString(error)          \
                      << " at line " << __LINE__ << std::endl;              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

// Macro to check rocBLAS status.
#define CHECK_ROCBLAS_STATUS(status)                                        \
    {                                                                       \
        if ((status) != rocblas_status_success) {                           \
            std::cerr << "rocBLAS error: " << (status)                      \
                      << " at line " << __LINE__ << std::endl;              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    }

int main() {
    // Define matrix dimensions.
    // For simplicity, we'll use square matrices: A (M x K), B (K x N), C (M x N)
    const int N = 1024;  // Number of columns in B and C
    const int M = N;     // Number of rows in A and C
    const int K = N;     // Number of columns in A and rows in B

    // Allocate host memory.
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float* h_a = (float*)malloc(size_A);
    float* h_b = (float*)malloc(size_B);
    float* h_c = (float*)malloc(size_C);

    // Initialize host matrices.
    // Here we fill A and B with random values and initialize C to zero.
    for (int i = 0; i < M * K; i++) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < M * N; i++) {
        h_c[i] = 0.0f;
    }

    // Allocate device memory.
    float *d_a, *d_b, *d_c;
    CHECK_HIP_ERROR(hipMalloc(&d_a, size_A));
    CHECK_HIP_ERROR(hipMalloc(&d_b, size_B));
    CHECK_HIP_ERROR(hipMalloc(&d_c, size_C));

    // Copy matrices from host to device.
    CHECK_HIP_ERROR(hipMemcpy(d_a, h_a, size_A, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, h_b, size_B, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_c, h_c, size_C, hipMemcpyHostToDevice));

    // Create a rocBLAS handle.
    rocblas_handle handle;
    CHECK_ROCBLAS_STATUS(rocblas_create_handle(&handle));

    // Set scalar values.
    float alpha = 1.0f;
    float beta  = 0.0f;

    // Call rocblas_sgemm.
    //
    // This performs: C = alpha * A * B + beta * C.
    // Note: rocBLAS expects matrices in column-major order by default.
    //       If your matrices are in row-major order, consider transposing
    //       the operation or adjusting the leading dimensions.
    CHECK_ROCBLAS_STATUS(rocblas_sgemm(
        handle,
        rocblas_operation_none, // Operation on A: no transpose.
        rocblas_operation_none, // Operation on B: no transpose.
        M,                      // Number of rows in A and C.
        N,                      // Number of columns in B and C.
        K,                      // Number of columns in A and rows in B.
        &alpha,                 // Scalar alpha.
        d_a,                    // Matrix A on the device.
        M,                      // Leading dimension of A.
        d_b,                    // Matrix B on the device.
        K,                      // Leading dimension of B.
        &beta,                  // Scalar beta.
        d_c,                    // Matrix C on the device.
        M                       // Leading dimension of C.
    ));

    // Copy the result matrix from device to host.
    CHECK_HIP_ERROR(hipMemcpy(h_c, d_c, size_C, hipMemcpyDeviceToHost));

    // Optionally, print a few elements from the result matrix.
    std::cout << "Some results from matrix C:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    // Clean up.
    rocblas_destroy_handle(handle);
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
