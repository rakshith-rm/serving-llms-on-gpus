import torch
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # This initializes CUDA
from transformers import BertTokenizer, BertForSequenceClassification
import time
import onnx
import onnxruntime as ort

# Initialize CUDA
cuda.init()

MODEL_NAME = "bert-base-uncased"
ONNX_PATH = "bert.onnx"
TRT_PATH = "bert.trt"

def export_to_onnx(model, tokenizer):
    model.eval()
    dummy_input = tokenizer("Hello, I'm a language model", return_tensors="pt")
    
    # Export the model
    torch.onnx.export(
        model,                     # model being run
        (dummy_input["input_ids"], dummy_input["attention_mask"]),  # model input
        "model.onnx",              # where to save the model
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=14,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input_ids', 'attention_mask'],   # the model's input names
        output_names=['output'],   # the model's output names
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'output': {0: 'batch_size'}
        }
    )
    print("Model has been converted to ONNX format")


def run_pytorch_inference(model, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.time()
    print("PyTorch inference time:", round(end - start, 4), "s")
    return outputs.logits

def run_onnx_inference(session, tokenizer, sentence):
    inputs = tokenizer(sentence, return_tensors="np", padding=True, truncation=True)
    
    # Only pass inputs the ONNX model expects
    ort_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"]
    }

    # Run inference
    start = time.time()
    ort_outs = session.run(None, ort_inputs)
    end = time.time()
    logits = ort_outs[0]

    print("ONNX inference time:", round(end - start, 4), "s")
    print("ONNX Output logits:", logits)



TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_trt_engine(onnx_model_path="bert.onnx", engine_path="bert.trt"):
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    # Set optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    for input_name in ["input_ids", "attention_mask"]:
        profile.set_shape(input_name, (1, 8), (1, 16), (1, 32))  # min, opt, max
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_path}")
    return engine_path


def load_trt_engine(trt_path=TRT_PATH):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(trt_path, "rb") as f:
        engine_data = f.read()
    return runtime.deserialize_cuda_engine(engine_data)


import numpy as np
import time


def run_trt_inference(engine, tokenizer, sentence):
    # Create execution context
    context = engine.create_execution_context()
    
    # Prepare input
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].numpy().astype(np.int64)
    attention_mask = inputs["attention_mask"].numpy().astype(np.int64)
    
    # Set input shapes for dynamic shapes
    context.set_input_shape("input_ids", input_ids.shape)
    context.set_input_shape("attention_mask", attention_mask.shape)
    
    # Allocate GPU memory for inputs and outputs
    d_input_ids = cuda.mem_alloc(input_ids.nbytes)
    d_attention_mask = cuda.mem_alloc(attention_mask.nbytes)
    
    # Get output shape after setting input shapes
    output_shape = context.get_tensor_shape("logits")
    output = np.empty(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output.nbytes)
    
    # Create CUDA stream
    stream = cuda.Stream()
    
    # Copy input data to GPU
    cuda.memcpy_htod_async(d_input_ids, input_ids, stream)
    cuda.memcpy_htod_async(d_attention_mask, attention_mask, stream)
    
    # Set tensor addresses
    context.set_tensor_address("input_ids", int(d_input_ids))
    context.set_tensor_address("attention_mask", int(d_attention_mask))
    context.set_tensor_address("logits", int(d_output))
    
    # Run inference
    start_time = time.time()
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    end_time = time.time()
    
    # Copy output back to CPU
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    
    # Process output
    logits = output[0] if len(output.shape) > 1 else output
    prediction = np.argmax(logits)
    
    print(f"Input: {sentence}")
    print(f"TensorRT inference time: {(end_time - start_time)*1000:.2f} ms")
    print(f"Prediction: {prediction}")
    print(f"Logits: {logits}")
    
    # Clean up GPU memory
    d_input_ids.free()
    d_attention_mask.free()
    d_output.free()
    
    return prediction, logits



def main():
    sentence = "The quick brown fox jumps over the lazy dog"

    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # 1. PyTorch
    print("\n--- PyTorch Inference ---")
    run_pytorch_inference(model, tokenizer, sentence)

    # 2. Export to ONNX and load ONNX
    print("\n--- ONNX Inference ---")
    export_to_onnx(model, tokenizer)
    ort_session = ort.InferenceSession(ONNX_PATH)
    run_onnx_inference(ort_session, tokenizer, sentence)

    # 3. TensorRT
    print("\n--- TensorRT Inference ---")
    build_trt_engine()
    trt_engine = load_trt_engine()
    run_trt_inference(trt_engine, tokenizer, sentence)

if __name__ == "__main__":
    main()
