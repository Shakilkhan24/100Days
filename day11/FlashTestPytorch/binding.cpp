#include <torch/extension.h>
#include "ATen/ATen.h"

void CudaFlashAttention(const float *Q,
                        const float *K,
                        const float *V,
                        float *O,
                        float *m,
                        float *l,
                        const int seq_len,
                        const int head_dim,
                        const int batch_size,
                        const int nr_heads);

torch::Tensor FlashAttention(torch::Tensor Q,
                             torch::Tensor K,
                             torch::Tensor V)
{
    int batch_size = Q.size(0);
    int nr_heads = Q.size(1);
    int seq_len = Q.size(2);
    int head_dim = Q.size(3);

    torch::Tensor m = torch::full({batch_size, nr_heads, seq_len},
                                  -std::numeric_limits<float>::infinity(),Q.options());
    torch::Tensor l = torch::zeros({batch_size, nr_heads, seq_len},Q.options());

    torch::Tensor O = torch::zeros_like(Q);
    CudaFlashAttention(Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), m.data_ptr<float>(), l.data_ptr<float>(), seq_len, head_dim, batch_size, nr_heads);
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("FlashAttention", &FlashAttention, "FlashAttention (CUDA)");
}