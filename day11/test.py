import torch
from torch.utils.cpp_extension import load


def speed_test():
    size = [1, 1000000]  # Batch size 1, dimension 1M
    A_large = torch.rand(size=size, device='cuda', dtype=torch.float32)
    num_runs = 100  # Number of iterations for timing
    
    def time_function(func, *args,**kwargs):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        for _ in range(10):
            _ = func(*args,**kwargs)
        
        start_event.record()
        for _ in range(num_runs):
            _ = func(*args,**kwargs)
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event) / num_runs

    print("\n" + "="*50)
    print("Speed Test (1M elements, average of 100 runs)")
    print("="*50)

    custom_relu_time = time_function(functions.ReLU, A_large)
    custom_softmax_time = time_function(functions.Softmax, A_large)
    custom_leaky_time = time_function(functions.LeakyReLU, A_large, -0.1)
    custom_tanh_time = time_function(functions.TanH, A_large)

    torch_relu_time = time_function(torch.nn.functional.relu, A_large)
    torch_softmax_time = time_function(torch.nn.functional.softmax, A_large, dim=1)
    torch_leaky_time = time_function(torch.nn.functional.leaky_relu, A_large, -0.1)
    torch_tanh_time = time_function(torch.tanh, A_large)

    print(f"{'Operation':<15} | {'Custom CUDA (ms)':<15} | {'PyTorch (ms)':<15} | Speedup")
    print("-"*60)
    print(f"{'ReLU':<15} | {custom_relu_time:15.3f} | {torch_relu_time:15.3f} | {torch_relu_time/custom_relu_time:5.1f}x")
    print(f"{'Softmax':<15} | {custom_softmax_time:15.3f} | {torch_softmax_time:15.3f} | {torch_softmax_time/custom_softmax_time:5.1f}x")
    print(f"{'LeakyReLU':<15} | {custom_leaky_time:15.3f} | {torch_leaky_time:15.3f} | {torch_leaky_time/custom_leaky_time:5.1f}x")
    print(f"{'TanH':<15} | {custom_tanh_time:15.3f} | {torch_tanh_time:15.3f} | {torch_tanh_time/custom_tanh_time:5.1f}x")

sources = ["binding.cpp", "ReLU.cu", "SoftMax.cu", "LeakyReLU.cu", "TanH.cu"]
functions = load("functions", sources=sources, verbose=True)

size = [1, 5]
A = torch.rand(size=size, device='cuda', dtype=torch.float32)

print("=" * 50)
print("Input Tensor A:\n", A)
print("=" * 50)

print("Custom CUDA Kernel Results:")
relu_result = functions.ReLU(A)
softmax_result = functions.Softmax(A)
leaky_relu_result = functions.LeakyReLU(A, -0.1)
tanh_result = functions.TanH(A)

print("ReLU:      ", relu_result)
print("SoftMax:   ", softmax_result)
print("LeakyReLU: ", leaky_relu_result)
print("TanH:      ", tanh_result)

print("Sum of SoftMax elements (Custom CUDA):", torch.sum(softmax_result, dim=1))
print("=" * 50)

print("PyTorch Built-in Results:")
relu_builtin = torch.nn.functional.relu(A)
softmax_builtin = torch.nn.functional.softmax(A, dim=1)  # Softmax along the correct dimension
leaky_relu_builtin = torch.nn.functional.leaky_relu(A, negative_slope=-0.1)
tanh_builtin = torch.tanh(A)

print("ReLU:      ", relu_builtin)
print("SoftMax:   ", softmax_builtin)
print("LeakyReLU: ", leaky_relu_builtin)
print("TanH:      ", tanh_builtin)

print("Sum of SoftMax elements (PyTorch):", torch.sum(softmax_builtin, dim=1))
print("=" * 50)

speed_test()