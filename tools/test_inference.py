from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "../models/tinyllama_cpu"

# Load tokenizer + model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")

# Input prompt
prompt = "Write a short email to my manager explaining that I will be working from home tomorrow."

inputs = tokenizer(prompt, return_tensors="pt")

# Generate
with torch.no_grad():
    output = model.generate(
        inputs["input_ids"],
        max_length=200,
        temperature=0.7,
        top_p=0.9
    )

# Decode and print
print("----- Generated Text -----")
print(tokenizer.decode(output[0], skip_special_tokens=True))
