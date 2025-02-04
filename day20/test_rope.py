import torch
import time
from torch.utils.cpp_extension import load

lib = load(
    name="rope",
    sources=["rope.cu"],
    extra_cuda_cflags=[        "-O3",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

def benchmark(func, x, out=None, iters=20):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        if out is not None:
            func(x, out)
        else:
            _ = func(x)
    torch.cuda.synchronize()
    return (time.time() - start) * 1000 / iters  

def naive_rope(x, theta=10000.0):
    dim = x.shape[-1]
    seq_len = x.shape[-2]
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)).cuda()
    freqs = torch.outer(torch.arange(seq_len, device='cuda'), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(x_ * freqs_cis).flatten(1).type_as(x)

sizes = [(4096, 512), (4096, 1024), (8192, 512), (8192, 1024)]
for M, N in sizes:
    print(f"Testing M={M}, N={N}")
    x = torch.randn((M, N), device='cuda', dtype=torch.float32).contiguous()
    out = torch.zeros_like(x)
    
    t_naive = benchmark(naive_rope, x)
    t_cuda = benchmark(lib.rope, x, out)
    
    print(f"Naive: {t_naive:.4f}ms, CUDA f32: {t_cuda:.4f}ms")
    print("-" * 60)
