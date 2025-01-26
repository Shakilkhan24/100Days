import torch
from torch.utils.cpp_extension import load
import time

# Load the custom CUDA extension
sources = ["binding.cpp", "FlashAttention.cu"]
flash_attention = load("flash_attention", sources=sources, verbose=True)
print("Custom CUDA extension loaded.")

def manual_attention(Q, K, V):
    batch_size, num_heads, seq_len, head_dim = Q.shape
    
    attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, heads, seq_len, seq_len]
    scale = 1.0 / (head_dim ** 0.5)
    attn_scores = attn_scores * scale
    attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, V)  # [batch, heads, seq_len, head_dim]
    return output
def test_flash_attention():
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64

    # Create random input tensors
    Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    # Warmup runs
    for _ in range(3):
        _ = flash_attention.FlashAttention(Q, K, V)
        _ = manual_attention(Q, K, V)

    # Benchmark custom FlashAttention
    custom_times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        _ = flash_attention.FlashAttention(Q, K, V)
        torch.cuda.synchronize()
        custom_times.append(time.time() - start)

    # Benchmark manual attention
    manual_times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start = time.time()
        _ = manual_attention(Q, K, V)
        torch.cuda.synchronize()
        manual_times.append(time.time() - start)

    # Get fastest iterations
    fastest_custom = min(custom_times) * 1000  # Convert to milliseconds
    fastest_manual = min(manual_times) * 1000

    # Print performance results
    print("\nPerformance results (fastest iteration):")
    print(f"Custom FlashAttention: {fastest_custom:.2f} ms")
    print(f"Manual PyTorch attention: {fastest_manual:.2f} ms")
    print(f"Speedup factor: {fastest_manual / fastest_custom:.2f}x")

if __name__ == "__main__":
    test_flash_attention()