import torch
import ctypes
import numpy as np
import argparse
import os
import performance

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)


lib.gather_cuda.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

def gather_torch(rank, axis, inputTensor, indexTensor):
    indices = [slice(None)] * rank
    indices[axis] = indexTensor
    return inputTensor[tuple(indices)]


def test(inputShape, indexShape, axis, test_dtype, device):
    print(f"Testing Gather on {device} with input shape:{inputShape} , index shape:{indexShape}, axis:{axis} ,dtype:{test_dtype}")
    inputTensor = torch.rand(inputShape, device=device, dtype=test_dtype)

    index_np = np.random.randint(0, inputShape[axis], size=indexShape).astype(np.int64)
    indexTensor = torch.from_numpy(index_np).to(device)
    
    rank = len(inputShape)
    
    torch_gather_time = performance.CudaProfile((gather_torch, (rank, axis, inputTensor, indexTensor)))


    inputShape = inputTensor.shape

    pre_size = int(np.prod(inputShape[:axis])) if axis > 0 else 1
    post_size = int(np.prod(inputShape[axis+1:])) if axis < len(inputShape)-1 else 1
    indices_size = int(indexTensor.numel())
    
    outShape = list(indexTensor.shape) + list(inputShape[:axis]) + list(inputShape[axis+1:])
    outTensor = torch.zeros(outShape, device=inputTensor.device, dtype=inputTensor.dtype)

    input_ptr = ctypes.cast(inputTensor.data_ptr(), ctypes.c_void_p)
    index_ptr = ctypes.cast(indexTensor.data_ptr(), ctypes.c_void_p)
    output_ptr = ctypes.cast(outTensor.data_ptr(), ctypes.c_void_p)

    
    def gather_cuda(rank, axis, inputTensor, indexTensor):
        lib.gather_cuda(input_ptr, index_ptr, output_ptr,
                pre_size, inputShape[axis], post_size,
                indexTensor.numel(), inputTensor.element_size())
        return outTensor


    custom_gather_time = performance.CudaProfile((gather_cuda, (rank, axis, inputTensor, indexTensor)))
    performance.logBenchmark(torch_gather_time, custom_gather_time)


    ref_output = gather_torch(rank, axis, inputTensor, indexTensor)
    custom_output = gather_cuda(rank, axis, inputTensor, indexTensor)
    
    tmpa = ref_output.to('cpu').numpy().flatten()
    tmpb = custom_output.to('cpu').numpy().flatten()
    atol = np.max(np.abs(tmpa - tmpb))
    rtol = atol / (np.max(np.abs(tmpb)) + 1e-8)
    print("absolute error: %.4e" % atol)
    print("relative error: %.4e" % rtol)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Gather on different devices.")
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mlu'], required=True, help="Device to run the tests on.")
    args = parser.parse_args()

    
    test_cases = [
        # inputShape , indexShape, axis, test_dtype, device
        ((3, 2), (2, 2), 0, torch.float32, "cuda"),
        ((12000, 2000), (2, 2), 1, torch.float32, "cuda"),
        ((3, 2), (1, 2), 1, torch.float32, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float32, "cuda"),

        ((3, 2), (2, 2), 0, torch.float16, "cuda"),
        ((3, 2), (1, 2), 1, torch.float16, "cuda"),
        ((50257, 768), (16, 1024), 0, torch.float16, "cuda"),
        ((3, 2, 4, 10), (1, 2, 3, 4), 2, torch.float16, "cuda"),
        ((1, 2, 1, 10, 20), (5, 4, 3, 2, 1), 3, torch.float32, "cuda"),

        ((5, 15, 30), (5, 6, 7, 8, 9), 2, torch.float16, "cuda"),
        ((8, 45), (10, 12, 15), 1, torch.float32, "cuda"),
    ]
    filtered_test_cases = [
        (inputShape, indexShape, axis, test_dtype, device)
        for inputShape, indexShape, axis, test_dtype, device in test_cases
        if device == args.device
    ]
    
    if args.device == 'mlu':
        import torch_mlu
    
    for inputShape, indexShape, axis, test_dtype, device in filtered_test_cases:
        test(inputShape, indexShape, axis, test_dtype, device)
