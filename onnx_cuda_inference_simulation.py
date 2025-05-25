import time
import numpy as np
import cupy as cp
import onnxruntime as ort
from transformers import BertTokenizer

# === Simple Setup ===
sentence = "This is a simple BERT inference example."
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# === Tokenization ===
print("Tokenizing input...")
inputs = tokenizer(sentence, return_tensors="np", padding=True, truncation=True, max_length=128)
input_ids = inputs["input_ids"].astype(np.int64)
attention_mask = inputs["attention_mask"].astype(np.int64)

print(f"Input shapes: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")

# === Method 1: ONNX Runtime (simulating TensorRT) ===
class ONNXInference:
    def __init__(self):
        # Use ONNX Runtime with CUDA provider as TensorRT alternative
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession("bert_corrected.onnx", providers=providers)
        print("ONNX Runtime session created with CUDA provider")
    
    def infer(self, input_ids, attention_mask):
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        outputs = self.session.run(None, inputs)
        return outputs[0]  # last_hidden_state

# === Method 2: ONNX Runtime + Custom CUDA Kernel ===
def simple_cuda_softmax(x):
    """Simple CUDA softmax using CuPy"""
    x_gpu = cp.asarray(x, dtype=cp.float32)
    # Numerical stability
    x_max = cp.max(x_gpu, axis=-1, keepdims=True)
    x_shifted = x_gpu - x_max
    exp_x = cp.exp(x_shifted)
    sum_exp = cp.sum(exp_x, axis=-1, keepdims=True)
    softmax_result = exp_x / sum_exp
    return cp.asnumpy(softmax_result)

class ONNXWithCustomKernel:
    def __init__(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession("bert_corrected.onnx", providers=providers)
        print("ONNX Runtime + Custom CUDA kernel session created")
    
    def infer_with_kernel(self, input_ids, attention_mask):
        # Run ONNX inference
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        outputs = self.session.run(None, inputs)
        bert_output = outputs[0]  # last_hidden_state
        
        # Apply custom CUDA softmax to first token
        first_token = bert_output[0, 0, :]  # Shape: [768]
        softmax_result = simple_cuda_softmax(first_token)
        
        return bert_output, softmax_result

# === Initialize Models ===
print("\n" + "="*50)
print("INITIALIZING MODELS")
print("="*50)

try:
    onnx_model = ONNXInference()
    onnx_custom_model = ONNXWithCustomKernel()
    print("Both models initialized successfully!")
except Exception as e:
    print(f"Model initialization failed: {e}")
    print("Make sure bert_corrected.onnx exists in the current directory")
    exit(1)

# === Warm up ===
print("\nWarming up...")
for i in range(3):
    _ = onnx_model.infer(input_ids, attention_mask)
    _ = onnx_custom_model.infer_with_kernel(input_ids, attention_mask)
print("Warmup completed!")

# === Benchmark Method 1: ONNX Only ===
print("\nBenchmarking ONNX Runtime inference...")
onnx_times = []
for i in range(20):
    start_time = time.perf_counter()
    output1 = onnx_model.infer(input_ids, attention_mask)
    end_time = time.perf_counter()
    onnx_times.append((end_time - start_time) * 1000)

avg_onnx_time = np.mean(onnx_times)
std_onnx_time = np.std(onnx_times)

# === Benchmark Method 2: ONNX + Custom CUDA Kernel ===
print("Benchmarking ONNX Runtime + Custom CUDA Kernel...")
custom_times = []
for i in range(20):
    start_time = time.perf_counter()
    output2, softmax_result = onnx_custom_model.infer_with_kernel(input_ids, attention_mask)
    end_time = time.perf_counter()
    custom_times.append((end_time - start_time) * 1000)

avg_custom_time = np.mean(custom_times)
std_custom_time = np.std(custom_times)

# === Results ===
print("\n" + "="*60)
print("PERFORMANCE COMPARISON RESULTS")
print("="*60)

print(f"\n📊 ONNX Runtime Only:")
print(f"   Average time: {avg_onnx_time:.2f} ± {std_onnx_time:.2f} ms")
print(f"   Min time: {min(onnx_times):.2f} ms")
print(f"   Max time: {max(onnx_times):.2f} ms")

print(f"\n🚀 ONNX Runtime + Custom CUDA Kernel:")
print(f"   Average time: {avg_custom_time:.2f} ± {std_custom_time:.2f} ms")
print(f"   Min time: {min(custom_times):.2f} ms")
print(f"   Max time: {max(custom_times):.2f} ms")

kernel_overhead = avg_custom_time - avg_onnx_time
print(f"\n⚡ Custom Kernel Overhead: {kernel_overhead:.2f} ms")
print(f"📈 Relative Overhead: {(kernel_overhead/avg_onnx_time)*100:.1f}%")

# === Verify Results ===
print(f"\n✅ Output Verification:")
print(f"   BERT output shape: {output1.shape}")
print(f"   BERT output range: [{np.min(output1):.4f}, {np.max(output1):.4f}]")
print(f"   Softmax result shape: {softmax_result.shape}")
print(f"   Softmax sum: {np.sum(softmax_result):.6f} (should be ~1.0)")

print(f"\n🎯 Goal Achieved: Compared inference times successfully!")
print("="*60)
