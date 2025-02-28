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

def square_relu(x: torch.Tensor, block_size: int = 32):
    # Default block_size is 32
    x = x.flatten()
    output = torch.empty_like(x, device=x.device, dtype=x.dtype)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCKSIZE']),)
    square_relu_function[grid](x, n, output, BLOCKSIZE=block_size)
    return output

def benchmark_square_relu(x: torch.Tensor, block_size: int, num_runs: int = 5):
    x = x.flatten()
    output = torch.empty_like(x, device=x.device, dtype=x.dtype)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCKSIZE']),)
    durations = []
    for _ in range(num_runs):
        start = time.time()
        square_relu_function[grid](x, n, output, BLOCKSIZE=block_size)
        torch.cuda.synchronize()  # Ensure all GPU work is done before timing end
        end = time.time()
        durations.append(end - start)
    return sum(durations) / num_runs

if __name__ == "__main__":
    # Define various input sizes and block sizes to test.
    batch_sizes = [1, 4, 8, 32]
    seq_lens = [1024, 2048, 4098, 16354]
    dims = [512, 758, 1028]
    block_sizes = [32, 64, 128, 256, 512]
    num_runs = 5

    results = []
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            for dim in dims:
                # Create a random input tensor on the GPU.
                x = torch.randn(batch_size, seq_len, dim).cuda()
                for block_size in block_sizes:
                    duration = benchmark_square_relu(x, block_size, num_runs)
                    results.append([batch_size, seq_len, dim, block_size, duration])
                    print(f"Batch size {batch_size}, Seq len {seq_len}, Dim {dim}, Block size {block_size}: {duration:.6f} seconds")

    # Save results to CSV.
    with open('benchmark_results.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Batch Size', 'Seq Len', 'Dim', 'Block Size', 'Duration'])
        csvwriter.writerows(results)

    # Load results with pandas.
    df = pd.read_csv('benchmark_results.csv')

    # Compute total number of elements for each test case.
    df['Total Elements'] = df['Batch Size'] * df['Seq Len'] * df['Dim']

    # Plot: x-axis = Total Elements, y-axis = Duration,
    # with one line per Block Size.
    plt.figure(figsize=(10, 6))
    for block in sorted(df['Block Size'].unique()):
        subset = df[df['Block Size'] == block].sort_values('Total Elements')
        plt.plot(subset['Total Elements'], subset['Duration'], marker='o', label=f'Block Size {block}')

    plt.xlabel('Total Elements (Batch Size * Seq Len * Dim)')
    plt.ylabel('Duration (seconds)')
    plt.title('Kernel Duration vs Total Elements for Different Block Sizes')
    plt.grid(True)
    plt.legend()
    plt.savefig('duration_vs_total_elements.png')
    plt.show()
