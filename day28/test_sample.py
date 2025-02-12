import torch
from torch.utils.cpp_extension import load
import time

lib = load(
    name="update_x",
    sources=["sample.cu"],
    extra_cuda_cflags=[        "-O3",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


print("Loaded ")

size = 10**6
device = "cuda"

x = torch.randn(size, device=device)
noise = torch.randn(size, device=device)
predicted_noise = torch.randn(size, device=device)
alpha = torch.tensor(0.9, device=device)
beta = torch.tensor(0.1, device=device)
alpha_hat = torch.tensor(0.81, device=device)

sqrt_alpha = torch.sqrt(alpha)
sqrt_alpha_hat = torch.sqrt(1 - alpha_hat)

torch.cuda.synchronize()
start = time.time()
x_cuda = lib.update_x(x.clone(), noise, predicted_noise, sqrt_alpha, sqrt_alpha_hat, beta, alpha)
torch.cuda.synchronize()
time_cuda = time.time() - start

torch.cuda.synchronize()
start = time.time()
x_torch = 1 / sqrt_alpha * (x - ((1 - alpha) / sqrt_alpha_hat) * predicted_noise) + torch.sqrt(beta) * noise
torch.cuda.synchronize()
time_torch = time.time() - start

print(f"CUDA Kernel Time: {time_cuda:.6f}s")
print(f"PyTorch Time: {time_torch:.6f}s")
print(f"Speedup: {time_torch / time_cuda:.2f}x")
