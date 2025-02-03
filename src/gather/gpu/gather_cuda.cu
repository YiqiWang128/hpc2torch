#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>

constexpr int BLOCKDIM = 128;
constexpr int THRESHOLD = 1000;

template <typename T>
__global__ void gather_kernel(T const *data, int64_t const* indices,
                              T *output,
                              int pre_size, int axis_size, int post_size,
                              int indices_size, int total_output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output)
        return;

    int post = idx % post_size;
    int tmp = idx / post_size;
    int indices_idx = tmp % indices_size;
    int pre_idx = tmp / indices_size;

    int64_t index_val = indices[indices_idx];
    if (index_val < 0 || index_val >= axis_size)
        return;
    int data_idx = pre_idx * (axis_size * post_size) + index_val * post_size + post;
    output[idx] = data[data_idx];
}

template <typename T>
__global__ void optimized_gather_kernel_2d(const T* __restrict__ input,
                                           const int64_t* __restrict__ indices,
                                           T* __restrict__ output,
                                           int stride, int indexSize)
{
    int tid = threadIdx.x; 
    int idx = blockIdx.y;
    if (tid >= stride)
        return;

    int64_t row = indices[idx];
    int over = (stride + BLOCKDIM - 1) / BLOCKDIM;
    for (int i = 0; i < over; i++){
        int col = tid + i * BLOCKDIM;
        if (col >= stride)
            break;
        int out_index = idx * stride + col;
        int in_index = row * stride + col;
        output[out_index] = input[in_index];
    }
}

extern "C" 
void gather_cuda(void const* data, void const* indices,
                 void* output,
                 int pre_size, int axis_size, int post_size,
                 int indices_size, int elem_size)
{
    int total_output = pre_size * indices_size * post_size;

    if (pre_size == 1 && (axis_size * post_size) > THRESHOLD) {
        dim3 blockDim(BLOCKDIM);
        dim3 gridDim(1, indices_size);
        if (elem_size == 2) {
            optimized_gather_kernel_2d<half><<<gridDim, blockDim>>>(
                reinterpret_cast<const half*>(data),
                reinterpret_cast<const int64_t*>(indices),
                reinterpret_cast<half*>(output),
                post_size, indices_size);
        }
        else if (elem_size == 4) {
            optimized_gather_kernel_2d<float><<<gridDim, blockDim>>>(
                reinterpret_cast<const float*>(data),
                reinterpret_cast<const int64_t*>(indices),
                reinterpret_cast<float*>(output),
                post_size, indices_size);
        }
        else {
            printf("Unsupported element size: %d\n", elem_size);
            return;
        }
    }
    else {
        int threads = 256;
        int blocks = (total_output + threads - 1) / threads;
        if (elem_size == 2) {
            gather_kernel<half><<<blocks, threads>>>(
                reinterpret_cast<const half*>(data),
                reinterpret_cast<const int64_t*>(indices),
                reinterpret_cast<half*>(output),
                pre_size, axis_size, post_size,
                indices_size, total_output);
        }
        else if (elem_size == 4) {
            gather_kernel<float><<<blocks, threads>>>(
                reinterpret_cast<const float*>(data),
                reinterpret_cast<const int64_t*>(indices),
                reinterpret_cast<float*>(output),
                pre_size, axis_size, post_size,
                indices_size, total_output);
        }
        else {
            printf("Unsupported element size: %d\n", elem_size);
            return;
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in gather_cuda: %s\n", cudaGetErrorString(err));
    }
}
