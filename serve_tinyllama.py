import ray
from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import ray.train.torch # Import for get_device

app = FastAPI()

# Define the request model for your API endpoint
class TextRequest(BaseModel):
    prompt: str

@serve.deployment(ray_actor_options={"num_gpus": 1}) # Allocate 1 GPU per replica
@serve.ingress(app) # Expose the FastAPI app through this deployment
class TinyLLaMA:
    def __init__(self):
        # Use ray.train.torch.get_device() to ensure proper GPU assignment
        # when deploying across multiple GPUs or even just one managed by Ray.
        self.device = ray.train.torch.get_device()
        print(f"Using device: {self.device}")

        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True, # Good practice for larger models
        ).to(self.device) # Move the model to the assigned device
        self.model.eval() # Set model to evaluation mode

        # Set pad_token_id if it's not already set, crucial for generation
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


    # Define your FastAPI endpoint method
    @app.post("/generate")
    async def generate_text_endpoint(self, request: TextRequest): # Renamed to avoid confusion with internal method
        user_prompt = request.prompt

        # 1. Format the prompt using the model's chat template
        messages = [
            {"role": "user", "content": user_prompt},
        ]
        
        # apply_chat_template with return_tensors="pt" returns a tensor directly
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # DEBUG: Print type and content to verify what apply_chat_template returns
        print(f"Type of input_ids: {type(input_ids)}")
        print(f"Shape of input_ids: {input_ids.shape}")

        # 2. Move input_ids to the device
        input_ids = input_ids.to(self.device)

        # 3. Prepare arguments for model.generate
        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": 100,
            "do_sample": False, # For deterministic output
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        with torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)
        
        # 4. Decode only the newly generated part (after the input prompt)
        # outputs[0] is the full sequence (input + generated).
        # input_ids.shape[-1] gives the length of the input sequence.
        generated_token_ids = outputs[0][input_ids.shape[-1]:]
        generated_text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        
        # Some models might generate leading/trailing whitespace or special characters,
        # .strip() helps clean that up.
        return {"generated_text": generated_text.strip()}

# This is the entry point for Ray Serve.
entrypoint_for_serve = TinyLLaMA.bind()

if __name__ == "__main__":
    ray.init() # Initialize Ray
    serve.start(detached=False) # Start the Serve runtime in the foreground
    
    # Run the deployment as an application.
    serve.run(entrypoint_for_serve)

    print("Ray Serve application started on port 8000 (default Serve port) or the port specified in Ray Dashboard.")
    print("You can send requests to http://0.0.0.0:8000/generate.")
    input("Press Enter to shut down the server...\n")
    serve.shutdown()
    ray.shutdown()
