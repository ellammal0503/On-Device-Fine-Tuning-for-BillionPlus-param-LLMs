"""
Generate text interactively using the fine-tuned LoRA adapter.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    model_path = "../models/tinyllama_cpu"
    adapter_path = "../adapters/email_tone"

    print(f"[info] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print(f"[info] Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None
    )

    print(f"[info] Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    device = torch.device("cpu")  # force CPU for stability
    model = model.to(device)

    print("[info] Interactive mode started (type 'exit' to quit)\n")

    while True:
        user_input = input(">> Prompt: ")
        if user_input.strip().lower() in {"exit", "quit"}:
            print("[info] Exiting chat.")
            break

        inputs = tokenizer(user_input, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n=== Response ===\n{response}\n")

if __name__ == "__main__":
    main()
