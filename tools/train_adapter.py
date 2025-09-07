"""
Train LoRA adapters on a prepared dataset.
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

# Disable MPS so PyTorch will only use CPU
torch.backends.mps.is_available = lambda : False
torch.backends.mps.is_built = lambda : False


def train_adapter():
    # Force CPU (MPS gives meta tensor error)
    device = torch.device("cpu")
    print(f"[info] Using device: {device}")

    # --- Load tokenizer ---
    model_path = "../models/tinyllama_cpu"
    print(f"[info] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load model ---
    print(f"[info] Loading model from {model_path} (slow on CPU)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=None,   # no auto device placement
    )
    model = model.to(device)

    # --- Load dataset ---
    dataset_path = "prepared_dataset"
    print(f"[info] Loading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)

    # Ensure labels are set
    print("[info] Preparing dataset (add labels)")
    def add_labels(examples):
        examples["labels"] = list(examples["input_ids"])  # force copy
        return examples

    dataset = dataset.map(add_labels, batched=True)

    # --- Data collator ---
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # --- Attach LoRA ---
    print("[info] Attaching LoRA adapters (PEFT)")
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    #model.gradient_checkpointing_enable()
    # model.gradient_checkpointing_enable()   # REMOVE for now
    model.config.use_cache = False


    # --- Training args (tiny run for smoke test) ---
    training_args = TrainingArguments(
        output_dir="../adapters/email_tone",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_dir="./logs",
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_steps=5,  # short test run
    )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Train ---
    print("[info] Starting training...")
    trainer.train()
    print("[info] Training complete!")

    # --- Save adapter ---
    output_dir = "../adapters/email_tone"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[info] Saving adapter to {output_dir}")
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    train_adapter()
