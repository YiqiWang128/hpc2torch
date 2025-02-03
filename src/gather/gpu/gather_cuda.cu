#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>

template <typename T>
__global__ void gather_kernel(const T* __restrict__ data,
                              const int64_t* __restrict__ indices,
                              T* __restrict__ output,
                              int pre_size, int axis_size, int post_size,
                              int indices_size,
                              int total_output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output)
        return;
    int post = idx % post_size;
    int tmp = idx / post_size;
    int indices_idx = tmp % indices_size;
    int pre = tmp / indices_size;
    int64_t index_val = indices[indices_idx];
    if (index_val < 0 || index_val >= axis_size)
        return;
    int data_idx = pre * (axis_size * post_size) + index_val * post_size + post;
    output[idx] = data[data_idx];
}

extern "C" 
void gather_cuda(void const* data, void const* indices,
                 void* output,
                 int pre_size, int axis_size, int post_size,
                 int indices_size, int elem_size)
{
    int total_output = pre_size * indices_size * post_size;
    int threads = 256;
    int blocks = (total_output + threads - 1) / threads;

    if (elem_size == 2) {
        gather_kernel<half><<<blocks, threads>>>(
            reinterpret_cast<const half*>(data),
            reinterpret_cast<const int64_t*>(indices),
            reinterpret_cast<half*>(output),
            pre_size, axis_size, post_size,
            indices_size, total_output);
    } else if (elem_size == 4) {
        gather_kernel<float><<<blocks, threads>>>(
            reinterpret_cast<const float*>(data),
            reinterpret_cast<const int64_t*>(indices),
            reinterpret_cast<float*>(output),
            pre_size, axis_size, post_size,
            indices_size, total_output);
    } else {
        printf("Unsupported element size: %d\n", elem_size);
        return;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in gather_cuda: %s\n", cudaGetErrorString(err));
    }
}
