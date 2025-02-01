#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>


#define n 6

int main(){
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    int j;
    float *x;
    x = (float*)malloc(sizeof(float)*n);
    for(j = 0 ; j<n ;++j){
        x[j] = (float)j;
    }

    printf("x:\n");
    for(j = 0 ; j<n ;++j){
        printf("%f\n",x[j]);
    }

    float *d_x;
    cudaStat = cudaMalloc((void**)&d_x,n*sizeof(float));

    stat = cublasCreate(&handle);
    stat = cublasSetVector(n,sizeof(float),x,1,d_x,1);
    
    int result;

    stat = cublasIsamax(handle,n,d_x,1,&result);
    printf("max: %f\n",fabs(x[result-1]));

    stat = cublasIsamin(handle,n,d_x,1,&result);
    printf("min: %f\n",fabs(x[result-1]));

    cudaFree(d_x);
    cublasDestroy(handle);
    free(x);
    return 0;
}