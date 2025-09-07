"""
Generate text using base model + LoRA adapter
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel


def load_model_with_adapter():
    device = torch.device("cpu")  # force CPU (safe for Mac)
    base_model_path = "../models/tinyllama_cpu"
    adapter_path = "../adapters/email_tone"

    print(f"[info] Loading tokenizer from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[info] Loading base model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map=None,
    ).to(device)

    print(f"[info] Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()  # optional: merge LoRA weights into base

    return model, tokenizer


def generate_text(prompt: str):
    model, tokenizer = load_model_with_adapter()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # CPU
    )

    print("[info] Generating text...")
    output = pipe(
        prompt,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    return output[0]["generated_text"]


if __name__ == "__main__":
    test_prompt = "Write a polite email to a manager requesting leave for 2 days."
    result = generate_text(test_prompt)
    print("\n=== Generated Text ===\n")
    print(result)
