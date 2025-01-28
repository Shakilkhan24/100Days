import torch
from torch.utils.cpp_extension import load
import time
from liger_kernel.ops import rms_norm
def rms_norm(tensor):
    return tensor / torch.sqrt(torch.mean(tensor ** 2))

sources = ["binding.cpp", "RMS.cu","RMSBetter.cu"]
RMS = load("RMS", sources=sources, verbose=True)
print("Custom CUDA extension loaded.")

tensor_sizes = [(1024,1024),(2048,2048),(4096, 4096),(8192,8192)]  



for tesnor_size in tensor_sizes:
    print("="*50)
    print("Input Size : " , tesnor_size);
    print("="*50)
    input_tensor = torch.randn(tesnor_size, device='cuda')

    start_time = time.time()
    result_pytorch = rms_norm(input_tensor)
    pytorch_time = time.time() - start_time
    print(f"PyTorch RMS time: {pytorch_time:.6f} seconds")

    start_time = time.time()
    result_custom = RMS.RMSV1(input_tensor)  # Replace with your custom operation
    custom_time = time.time() - start_time
    print(f"Custom kernel 1 time: {custom_time:.6f} seconds")


    start_time = time.time()
    result_custom = RMS.RMSV2(input_tensor)  
    custom_time = time.time() - start_time
    print(f"Custom kernel 2 time: {custom_time:.6f} seconds")


    start_time = time.time()
    result_custom = rms_norm(input_tensor)  
    custom_time = time.time() - start_time
    print(f"Liger kernel time: {custom_time:.6f} seconds")
