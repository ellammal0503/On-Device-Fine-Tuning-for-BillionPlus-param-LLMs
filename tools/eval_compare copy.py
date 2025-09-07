import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

device = torch.device("cpu")
print(f"[info] Using device: {device}")

# Paths
model_path = "../models/tinyllama_cpu"
adapter_path = "../adapters/email_tone"

# --- Load tokenizer ---
print("[info] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Load base model ---
print("[info] Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
base_model = base_model.to(device)

# --- Load adapter model ---
print("[info] Loading model + adapter...")
adapter_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
adapter_model = PeftModel.from_pretrained(adapter_model, adapter_path)
adapter_model = adapter_model.to(device)

# --- Prompts for evaluation ---
prompts = [
    "Write a polite email to a manager requesting leave for 2 days.",
    "Draft a professional email to a client apologizing for a delay in delivery.",
    "Write a formal resignation email with a notice period of 1 month.",
    "Compose an email requesting feedback on a recent project.",
    "Write a polite reminder email about an overdue invoice."
]

# --- References (ideal outputs) ---
references = [
    "Dear Manager, I would like to request leave for 2 days due to personal reasons. I will be away on [dates] and will resume work on [date]. Thank you for your understanding.",
    "Dear Client, I sincerely apologize for the delay in delivery. We experienced unexpected issues but your order will be delivered by [date]. Thank you for your patience.",
    "Dear [Manager], Please accept this letter as my formal resignation, effective one month from today. I am grateful for the opportunities I’ve had here.",
    "Dear [Name], I hope you are doing well. I would appreciate your feedback on our recent project, particularly on quality, timeliness, and execution.",
    "Dear [Client], This is a polite reminder that invoice #[number], due on [date], is now overdue. Kindly arrange payment at your earliest convenience."
]

def generate_text(model, prompt, max_new_tokens=80):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Run evaluation ---
base_outputs = []
adapter_outputs = []

print("\n" + "="*80)
for i, prompt in enumerate(prompts):
    print(f"Prompt: {prompt}\n")

    base_out = generate_text(base_model, prompt)
    adapter_out = generate_text(adapter_model, prompt)

    print("[Base Model Output]")
    print(base_out + "\n")
    print("[Adapter Model Output]")
    print(adapter_out + "\n")

    base_outputs.append(base_out)
    adapter_outputs.append(adapter_out)

    print("="*80)

# --- Automatic Metrics ---
print("\n=== Automatic Evaluation Metrics ===\n")

# BLEU
base_bleu = bleu.compute(predictions=base_outputs, references=[[r] for r in references])
adapter_bleu = bleu.compute(predictions=adapter_outputs, references=[[r] for r in references])

print(f"Base BLEU: {base_bleu['bleu']:.4f}")
print(f"Adapter BLEU: {adapter_bleu['bleu']:.4f}")

# ROUGE
base_rouge = rouge.compute(predictions=base_outputs, references=references)
adapter_rouge = rouge.compute(predictions=adapter_outputs, references=references)

print("\nBase ROUGE:", {k: round(v, 4) for k, v in base_rouge.items()})
print("Adapter ROUGE:", {k: round(v, 4) for k, v in adapter_rouge.items()})
