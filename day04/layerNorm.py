import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(X, Y, gamma, beta, N, eps, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)  # Each row (batch sample) runs independently

    # Offsets for this row
    offset = row * N + tl.arange(0, BLOCK_SIZE)
    mask = offset < X.shape[0] * X.shape[1]

    # Load input data
    x = tl.load(X + offset, mask=mask, other=0.0)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N
    var = tl.sum((x - mean) ** 2, axis=0) / N

    # Normalize
    x_norm = (x - mean) / tl.sqrt(var + eps)

    # Scale and shift
    gamma_val = tl.load(gamma + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    beta_val = tl.load(beta + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    y = x_norm * gamma_val + beta_val

    # Store result
    tl.store(Y + offset, y, mask=mask)

class TritonLayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.gamma = torch.nn.Parameter(torch.ones(normalized_shape))
        self.beta = torch.nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        B, N = x.shape
        y = torch.empty_like(x)
        
        layer_norm_kernel[(B,)](
            x, y, self.gamma, self.beta, N, self.eps,
            BLOCK_SIZE=N
        )
        return y

# Testing the implementation
x = torch.randn(4, 128, device="cuda")  # Batch of 4, feature size 128
layer_norm = TritonLayerNorm(128).cuda()
y = layer_norm(x)
print(y)
