## HIP API

- Device Management:  
  - `hipSetDevice()`, `hipGetDevice()`, `hipGetDeviceProperties()`

- Memory Management:  
  - `hipMalloc()`, `hipMemcpy()`, `hipMemcpyAsync()`, `hipFree()`

- Streams:  
  - `hipStreamCreate()`, `hipSynchronize()`, `hipStreamSynchronize()`, `hipStreamFree()`

- Events:  
  - `hipEventCreate()`, `hipEventRecord()`, `hipStreamWaitEvent()`, `hipEventElapsedTime()`

- Device Kernels:  
  - `__global__`, `__device__`, `hipLaunchKernelGGL()`

- Identical to CUDA constructs:  
  - `threadIdx`, `blockIdx`, `blockDim`, `__shared__`

- Error Handling:  
  - `hipGetLastError()`, `hipGetErrorString()`

---

### Kernel Execution Model

- Kernels run on a 3D grid.  
- Usually, mapping to 1D is sufficient, but 2D and 3D are possible.  
- Each dimension of the grid is partitioned into blocks.  
- Each block consists of threads.

Threads can access:  
- `blockIdx.x`, `blockIdx.y`, `blockIdx.z`  
- `threadIdx.x`, `threadIdx.y`, `threadIdx.z`  
- `blockDim.x`, `blockDim.y`, `blockDim.z`  
- `gridDim.x`, `gridDim.y`, `gridDim.z`  

```cpp
dim3 threads(256, 1, 1);                   // 3D block dimensions
dim3 blocks((N + 256 - 1) / 256, 1, 1);    // 3D grid dimensions

hipLaunchKernelGGL(
    myKernel,   // Kernel name (__global__ void function)
    blocks,     // Grid dimensions
    threads,    // Block dimensions
    0,          // Dynamic LDS
    0,          // Stream (0 = default)
    N, a        // Kernel arguments
);
```

**Good Practice**: Use multiples of 64 threads per block, such as 256, to align with wavefronts.

---

### Device Memory

Equivalent to CUDA, but prefixed with `hip`:
- `cudaMemcpy` -> `hipMemcpy`
- `cudaMalloc` -> `hipMalloc`
- `cudaFree` -> `hipFree`

---

### Error Checking

```cpp
hipError_t status1 = hipMalloc(...);
hipError_t status2 = hipMemcpy(...);
hipError_t status3 = hipGetLastError();
hipError_t status4 = hipPeekAtLastError();
```

Use a helper macro to handle errors:

```cpp
#define HIP_CHECK(command) {                      \
    hipError_t status = command;                  \
    if (status != hipSuccess) {                   \
        std::cerr << "Error: HIP reports "        \
                  << hipGetErrorString(status)    \
                  << std::endl;                   \
        std::abort();                             \
    }                                             \
}
```

---

## Device Management

Multiple GPUs. Multiple host threads/ MPI ranks.

```cpp
// Host can query the number of devices visible to the system
int numDevices = 0;
hipGetDeviceCount(&numDevices);

// Host tells the runtime to issue instructions to a particular device
int deviceId = 0;
hipSetDevice(deviceId);

// Host can query what device is currently selected
hipGetDevice(&deviceId);
```

### Device Properties

```cpp
hipDeviceProp_t props;
hipGetDeviceProperties(&props, deviceId);
```

### Blocking vs Nonblocking API

- Kernel is non-blocking.
- `hipMemcpy` is blocking.
- `hipMemcpyAsync()` is non-blocking.

---

## Streams

- A stream in HIP is a queue of tasks (kernels, memcpys, events).
- Tasks enqueued in a stream complete in order on that stream.
- Tasks being executed in different streams are allowed to overlap and share device resources.

Create:
```cpp
hipStream_t stream;
hipStreamCreate(&stream);

hipStreamDestroy(stream);
```

Example:

```cpp
hipLaunchKernelGGL(myKernel1, dim3(1), dim3(256), 0, 0, 256, d_a1);
hipLaunchKernelGGL(myKernel2, dim3(1), dim3(256), 0, 0, 256, d_a2);
hipLaunchKernelGGL(myKernel3, dim3(1), dim3(256), 0, 0, 256, d_a3);
hipLaunchKernelGGL(myKernel4, dim3(1), dim3(256), 0, 0, 256, d_a4);
```

We will have:

[NULLSTREAM] : [myKernel1][myKernel2][myKernel3][myKernel4]

With streams we can do this:

```cpp
hipLaunchKernelGGL(myKernel1, dim3(1), dim3(256), 0, stream1, 256, d_a1);
hipLaunchKernelGGL(myKernel2, dim3(1), dim3(256), 0, stream2, 256, d_a2);
hipLaunchKernelGGL(myKernel3, dim3(1), dim3(256), 0, stream3, 256, d_a3);
hipLaunchKernelGGL(myKernel4, dim3(1), dim3(256), 0, stream4, 256, d_a4);
```

And we will have:
[NULLSTREAM]
[Stream1] : [myKernel1]
[Stream2] : [myKernel2]
[Stream3] : [myKernel3]
[Stream4] : [myKernel4]

Note 1: Check that the kernels modify different parts of memory to avoid data races.
Note 2: With large kernels, overlapping computations may not help performance.

---

## Pinned Memory

Pinned memory is host memory that is allocated in a way that allows the GPU to directly access it without the need for additional copying steps. This is particularly important for optimizing data transfer between the host (CPU) and the device (GPU), as it can significantly improve performance by reducing the overhead associated with memory transfers.

### Key Points about Pinned Memory in HIP:

1. **Allocation**:
   - Pinned memory is allocated using the `hipHostMalloc` function. This function ensures that the allocated memory is page-locked, meaning it is non-swappable and can be directly accessed by the GPU.
   ```cpp
   double *h_a = NULL;
   hipHostMalloc(&h_a, Nbytes);
   ```

2. **Deallocation**:
   - Pinned memory should be freed using the `hipHostFree` function to release the allocated resources.
   ```cpp
   hipHostFree(h_a);
   ```

3. **Performance Benefits**:
   - When host memory is pinned, the bandwidth for data transfers between the host and device (Host<->Device memcpy) increases significantly. This is because the GPU can access the pinned memory directly, bypassing the need for intermediate copies and reducing latency.

4. **Best Practices**:
   - It is a good practice to allocate host memory that is frequently transferred to/from the device as pinned memory. This can help optimize performance-critical applications by ensuring that data transfers are as efficient as possible.

### Example Usage:

Here is a simple example demonstrating the allocation and deallocation of pinned memory in HIP:

```cpp
#include <hip/hip_runtime.h>

int main() {
    const size_t Nbytes = 1024 * 1024; // 1 MB
    double *h_a = NULL;

    // Allocate pinned host memory
    hipHostMalloc(&h_a, Nbytes);

    // Use the pinned memory (e.g., initialize it)
    for (size_t i = 0; i < Nbytes / sizeof(double); ++i) {
        h_a[i] = static_cast<double>(i);
    }

    // Free the pinned host memory
    hipHostFree(h_a);

    return 0;
}
```

In this example, `hipHostMalloc` is used to allocate pinned memory, and `hipHostFree` is used to free it. The pinned memory can then be used for data transfers between the host and device, leveraging the performance benefits of direct GPU access.



## More on streams:

```cpp
//////////////////////////////////////////////////////////////////
hipMemcpy(d_a1, h_a1, Nbytes, hipMemcpyHostToDevice));
hipMemcpy(d_a2, h_a2, Nbytes, hipMemcpyHostToDevice));
hipMemcpy(d_a3, h_a3, Nbytes, hipMemcpyHostToDevice));

hipLaunchKernelGGL(myKernel1, blocks, threads, 0, 0, N, d_a1);
hipLaunchKernelGGL(myKernel2, blocks, threads, 0, 0, N, d_a2);
hipLaunchKernelGGL(myKernel3, blocks, threads, 0, 0, N, d_a3);

hipMemcpy(h_a1, d_a1, Nbytes, hipMemcpyDeviceToHost);
hipMemcpy(h_a2, d_a2, Nbytes, hipMemcpyDeviceToHost);
hipMemcpy(h_a3, d_a3, Nbytes, hipMemcpyDeviceToHost);
// Result:
// NULLSTREAM : [HtoD1][HtoD2][HtoD3][K1][K2][K3][DtoH1][DtoH2][DtoH3]
///////////////////////////////////////////////////////////////////

hipMemcpyAsync(d_a1, h_a1, Nbytes, hipMemcpyHostToDevice, stream1);
hipMemcpyAsync(d_a2, h_a2, Nbytes, hipMemcpyHostToDevice, stream2);
hipMemcpyAsync(d_a3, h_a3, Nbytes, hipMemcpyHostToDevice, stream3);

hipLaunchKernelGGL(myKernel1, blocks, threads, 0, stream1, N, d_a1);
hipLaunchKernelGGL(myKernel2, blocks, threads, 0, stream2, N, d_a2);
hipLaunchKernelGGL(myKernel3, blocks, threads, 0, stream3, N, d_a3);

hipMemcpyAsync(h_a1, d_a1, Nbytes, hipMemcpyDeviceToHost, stream1);
hipMemcpyAsync(h_a2, d_a2, Nbytes, hipMemcpyDeviceToHost, stream2);
hipMemcpyAsync(h_a3, d_a3, Nbytes, hipMemcpyDeviceToHost, stream3);

// NULLSTEAM : nothing
// STREAM 1  : [HToD1][K1][DToH1]
// STREAM 2  :        [HToD2][K2][DToH2]
// STREAM 3  :                [HToD2][K2][DToH2]

//
////////////////////////////////////////////////////////////////////


```cpp

hipDeviceSynchronize(); // heavy sync point . Blocks host untill all work in all devices reported complete
hipStreamSynchronize(stream); // blocks host until all work in a stream has reported complete
```
## Events

`hipEvent_t` object is created on the device . We create with this `hipEventCreate(&event)`

We can queue an event into a stream:

`hipEventRecord(event,stream);`

-> the event recors what work is currently enqueued in the stream
-> When the stream's eecution reaches the event , the event is considered `complete`

We need to destroy event objects


```cpp
//Queue local compute kernel
hipLaunchKernelGGL(myKernel, blocks, threads, 0, computeStream, N, d_a);

//Copy halo data to host
hipMemcpyAsync(h_commBuffer, d_commBuffer, Nbytes, hipMemcpyDeviceToHost, dataStream);
hipStreamSynchronize(dataStream); //Wait for data to arrive

//Exchange data with MPI
MPI_Data_Exchange(h_commBuffer);

//Send new data back to device
hipMemcpyAsync(d_commBuffer, h_commBuffer, Nbytes, hipMemcpyHostToDevice, dataStream);
```


