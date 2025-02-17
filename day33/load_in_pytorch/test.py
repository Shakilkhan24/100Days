import torch
import ctypes
import time

# Load the shared object
lib = ctypes.CDLL('./kernel.so')

# Specify the argument types for the host wrapper function.
lib.launch_kernel_addition.argtypes = [
    ctypes.c_void_p,  # pointer to A
    ctypes.c_void_p,  # pointer to B
    ctypes.c_void_p,  # pointer to C
    ctypes.c_size_t,  # N
    ctypes.c_int,     # grid_x
    ctypes.c_int,     # grid_y
    ctypes.c_int,     # grid_z
    ctypes.c_int,     # block_x
    ctypes.c_int,     # block_y
    ctypes.c_int      # block_z
]
lib.launch_kernel_addition.restype = None

N = 1000

# Create input tensors on the ROCm device.
A = torch.randn(N, device='cuda', dtype=torch.float32)
B = torch.randn(N, device='cuda', dtype=torch.float32)
C = torch.empty(N, device='cuda', dtype=torch.float32)

# Get pointers to the tensor data.
a_ptr = A.data_ptr()
b_ptr = B.data_ptr()
c_ptr = C.data_ptr()

# Define block and grid sizes.
block_size = 256
grid_size = (N + block_size - 1) // block_size

def measure_amd_kernel_time():
    start_amd = time.time()
    lib.launch_kernel_addition(
        ctypes.c_void_p(a_ptr),
        ctypes.c_void_p(b_ptr),
        ctypes.c_void_p(c_ptr),
        ctypes.c_size_t(N),
        ctypes.c_int(grid_size),  # grid_x
        ctypes.c_int(1),          # grid_y
        ctypes.c_int(1),          # grid_z
        ctypes.c_int(block_size), # block_x
        ctypes.c_int(1),          # block_y
        ctypes.c_int(1)           # block_z
    )
    torch.cuda.synchronize()  # Ensure the kernel has finished executing
    end_amd = time.time()
    return end_amd - start_amd

def measure_pytorch_time():
    start_pytorch = time.time()
    c_pytorch = A + B
    end_pytorch = time.time()
    return end_pytorch - start_pytorch

# Run the measurements 5 times and get the lowest time
amd_times = [measure_amd_kernel_time() for _ in range(5)]
pytorch_times = [measure_pytorch_time() for _ in range(5)]

min_amd_time = min(amd_times)
min_pytorch_time = min(pytorch_times)

# Verify the result.
if torch.allclose(C, A + B):
    print("Success!")
else:
    print("Error in computation.")

print(f"Lowest AMD kernel execution time: {min_amd_time} seconds")
print(f"Lowest Pytorch computation time: {min_pytorch_time} seconds")
