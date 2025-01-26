#include <torch/extension.h>
#include "ATen/ATen.h"


void CudaLeakyReLU(float *A,float*B,float slope ,int N);
torch::Tensor LeakyReLU(torch::Tensor A, float slope){
    torch::Tensor B = torch::empty_like(A);
    int N = A.numel();
    CudaLeakyReLU(A.data_ptr<float>(),B.data_ptr<float>(),slope,N);
    return B;
}

void CudaReLU(float *A,float*B, int N);
torch::Tensor ReLU(torch::Tensor A){
    torch::Tensor B = torch::empty_like(A);
    int N = A.numel();
    CudaReLU(A.data_ptr<float>(),B.data_ptr<float>(),N);
    return B;
}

void CudaReLUBackward(float *A, float *Gi, float *Go, int N);
torch::Tensor ReLUBackward(torch::Tensor A, torch::Tensor Go){
    torch::Tensor Gi = torch::empty_like(A);
    int N = A.numel();
    CudaReLUBackward(A.data_ptr<float>(),Gi.data_ptr<float>(),Go.data_ptr<float>(),N);
    return Go;
}










void CudaSoftmax(float *input, float *output, int BatchSize, int Dim) ;
torch::Tensor Softmax(torch::Tensor input) {
    int BatchSize = input.size(0);
    int Dim = input.size(1);
    torch::Tensor output = torch::empty_like(input);
    CudaSoftmax(input.data_ptr<float>(), output.data_ptr<float>(), BatchSize, Dim);
    return output;
}

void CudaTanH(float *A,float*B, int N);
torch::Tensor TanH(torch::Tensor A){
    torch::Tensor B = torch::empty_like(A);
    int N = A.numel();
    CudaTanH(A.data_ptr<float>(),B.data_ptr<float>(),N);
    return B;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("LeakyReLU", &LeakyReLU, "LeakyReLU (CUDA)");
    m.def("ReLU", &ReLU, "ReLU (CUDA)");
    m.def("ReLUBackward", &ReLUBackward, "ReLU (CUDA)");
    m.def("Softmax", &Softmax, "Softmax (CUDA)");
    m.def("TanH", &TanH, "TanH (CUDA)");
}