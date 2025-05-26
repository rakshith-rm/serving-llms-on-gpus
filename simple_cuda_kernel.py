# NOTE: The kernel slows done the base model. Implementation is just to demo custom CUDA kernel integration in PyTorch

import torch
import torch.nn as nn
import time
import numpy as np
import cupy as cp

# Your SimpleNet model (same as before)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(20, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# CuPy kernel (same as from previous message)
matmul_kernel_code = r'''
extern "C" __global__
void matmul_kernel(const float* a, const float* b, float* out, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float tmp = 0.0f;
        for (int k = 0; k < K; ++k) {
            tmp += a[row * K + k] * b[k * N + col];
        }
        out[row * N + col] = tmp;
    }
}
'''

matmul_kernel = cp.RawKernel(matmul_kernel_code, 'matmul_kernel')

def custom_linear_cuda(input_tensor, weight, bias):
    input_np = input_tensor.detach().cpu().numpy().astype(np.float32)
    weight_np = weight.detach().cpu().numpy().astype(np.float32).T  # transpose here

    M, K = input_np.shape
    K2, N = weight_np.shape
    assert K == K2

    d_input = cp.asarray(input_np)
    d_weight = cp.asarray(weight_np)
    d_out = cp.zeros((M, N), dtype=cp.float32)

    threads_per_block = (16, 16)
    blocks_per_grid_x = (M + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (N + threads_per_block[1] - 1) // threads_per_block[1]

    matmul_kernel((blocks_per_grid_x, blocks_per_grid_y), threads_per_block,
                  (d_input, d_weight, d_out, M, N, K))

    out_tensor = torch.tensor(cp.asnumpy(d_out), device=input_tensor.device)
    if bias is not None:
        out_tensor += bias
    return out_tensor


def custom_inference(input_tensor, model):
    # Forward using custom CUDA kernel for each linear layer

    x = custom_linear_cuda(input_tensor, model.fc1.weight, model.fc1.bias)
    x = torch.relu(x)
    x = custom_linear_cuda(x, model.fc2.weight, model.fc2.bias)
    return x

def benchmark_inference(model, input_data, runs=100):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(input_data)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            output = model(input_data)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        end = time.time()

    avg_ms = (end - start) / runs * 1000
    return avg_ms, output

def benchmark_custom_inference(input_data, model, runs=100):
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = custom_inference(input_data, model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            output = custom_inference(input_data, model)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        end = time.time()

    avg_ms = (end - start) / runs * 1000
    return avg_ms, output

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 64
    input_data = torch.randn(batch_size, 20, device=device)

    model = SimpleNet().to(device)

    native_time, native_out = benchmark_inference(model, input_data)
    print(f"Native PyTorch Inference: {native_time:.3f} ms")

    custom_time, custom_out = benchmark_custom_inference(input_data, model)
    print(f"Custom CUDA Kernel Inference: {custom_time:.3f} ms")

    diff = torch.mean(torch.abs(native_out - custom_out)).item()
    print(f"Mean absolute difference in outputs: {diff:.6f}")

if __name__ == "__main__":
    main()
