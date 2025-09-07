"""
Compare base TinyLlama vs LoRA adapter outputs on multiple prompts.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# -----------------------------
# Prompts to evaluate
# -----------------------------
PROMPTS = [
    "Write a polite email to a manager requesting leave for 2 days.",
    "Draft a professional email to a client apologizing for a delay in delivery.",
    "Write a formal resignation email with a notice period of 1 month.",
    "Compose an email requesting feedback on a recent project.",
    "Write a polite reminder email about an overdue invoice."
]

def generate(model, tokenizer, prompt, device, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    device = torch.device("cpu")
    print(f"[info] Using device: {device}")

    base_model_path = "../models/tinyllama_cpu"
    adapter_path = "../adapters/email_tone"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    print("[info] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map=None
    ).to(device)

    # Load model + adapter
    print("[info] Loading model + adapter...")
    adapter_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map=None
    ).to(device)
    adapter_model = PeftModel.from_pretrained(adapter_model, adapter_path)

    # Evaluate on prompts
    for prompt in PROMPTS:
        print("\n" + "="*80)
        print(f"Prompt: {prompt}\n")

        base_out = generate(base_model, tokenizer, prompt, device)
        print("[Base Model Output]")
        print(base_out)

        adapter_out = generate(adapter_model, tokenizer, prompt, device)
        print("\n[Adapter Model Output]")
        print(adapter_out)


if __name__ == "__main__":
    main()
