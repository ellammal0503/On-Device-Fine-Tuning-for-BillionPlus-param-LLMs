"""
Evaluate Base vs Adapter models on prompts and save results + metrics.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate
from sentence_transformers import SentenceTransformer, util
import json

# -----------------------------
# Prompts
# -----------------------------
PROMPTS = [
    "Write a polite email to a manager requesting leave for 2 days.",
    "Draft a professional email to a client apologizing for a delay in delivery.",
    "Write a formal resignation email with a notice period of 1 month.",
    "Compose an email requesting feedback on a recent project.",
    "Write a polite reminder email about an overdue invoice."
]

# -----------------------------
# Generation helper
# -----------------------------
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

    # Load adapter model
    print("[info] Loading model + adapter...")
    adapter_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map=None
    ).to(device)
    adapter_model = PeftModel.from_pretrained(adapter_model, adapter_path)

    # Metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")

    results = []

    for prompt in PROMPTS:
        print("\n" + "="*80)
        print(f"Prompt: {prompt}\n")

        base_out = generate(base_model, tokenizer, prompt, device)
        adapter_out = generate(adapter_model, tokenizer, prompt, device)

        # ---- Metrics ----
        rouge_score = rouge.compute(predictions=[adapter_out], references=[base_out])
        bleu_score = bleu.compute(
            predictions=[adapter_out],
            references=[[base_out]]
        )
        emb1 = bert_model.encode(base_out, convert_to_tensor=True)
        emb2 = bert_model.encode(adapter_out, convert_to_tensor=True)
        bert_sim = util.cos_sim(emb1, emb2).item()

        print("[Base Output]\n", base_out)
        print("\n[Adapter Output]\n", adapter_out)
        print("\n--- Metrics ---")
        print(f"ROUGE-L: {rouge_score['rougeL']:.4f}")
        print(f"BLEU: {bleu_score['bleu']:.4f}")
        print(f"BERT Similarity: {bert_sim:.4f}")

        results.append({
            "prompt": prompt,
            "base_output": base_out,
            "adapter_output": adapter_out,
            "rougeL": rouge_score["rougeL"],
            "bleu": bleu_score["bleu"],
            "bert_similarity": bert_sim
        })

    # Save results to JSON
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to eval_results.json")


if __name__ == "__main__":
    main()
