import torch
import triton
import triton.language as tl
import time
import csv
import matplotlib.pyplot as plt
import pandas as pd

@triton.jit
def square_relu_function(
    x_ptr, N,
    o_ptr,
    BLOCKSIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = pid * BLOCKSIZE + tl.arange(0, BLOCKSIZE)
    mask = offset < N
    x = tl.load(x_ptr + offset, mask=mask)
    o = tl.where(x > 0, x * x, 0)
    tl.store(o_ptr + offset, o, mask=mask)

def square_relu(x: torch.Tensor,block_size: int = 32):
    size = x.size()
    x = x.flatten()
    output = torch.empty_like(x, device=x.device, dtype=x.dtype)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCKSIZE']),)
    square_relu_function[grid](x, n, output, BLOCKSIZE=block_size)
    output
    
def benchmark_square_relu(x: torch.Tensor, block_size: int, num_runs: int = 5):
    size = x.size()
    x = x.flatten()
    output = torch.empty_like(x, device=x.device, dtype=x.dtype)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCKSIZE']),)
    durations = []
    for _ in range(num_runs):
        start = time.time()
        square_relu_function[grid](x, n, output, block_size)
        end = time.time()
        durations.append(end - start)
    return sum(durations) / num_runs

if __name__ == "__main__":
    batch_sizes = [1, 4, 8, 32]
    seq_lens = [1024, 2048, 4098, 16354]
    dims = [512, 758, 1028]
    block_sizes = [32, 64, 128, 256, 512]
    num_runs = 5

    results = []

    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for dim in dims:
                x = torch.randn(batch_size, seq_len, dim).cuda()  # Move tensor to GPU
                for block_size in block_sizes:
                    duration = benchmark_square_relu(x, block_size, num_runs)
                    results.append([batch_size, seq_len, dim, block_size, duration])
                    print(f"Batch size {batch_size}, Seq len {seq_len}, Dim {dim}, Block size {block_size}: {duration:.6f} seconds")

    with open('benchmark_results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Batch Size', 'Seq Len', 'Dim', 'Block Size', 'Duration'])
        csvwriter.writerows(results)

    df = pd.read_csv('benchmark_results.csv')

    avg_durations = df.groupby('Block Size')['Duration'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_durations['Block Size'], avg_durations['Duration'], marker='o')
    plt.xlabel('Block Size')
    plt.ylabel('Average Duration (seconds)')
    plt.title('Average Duration for Each Block Size')
    plt.grid(True)
    plt.savefig('average_duration_per_block_size.png')
    plt.show()