"""
Train a LoRA adapter on-device or on edge hardware.
"""

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

def train_adapter(model_path: str, dataset_path: str, output_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    dataset = load_from_disk(dataset_path)

    args = TrainingArguments(
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_steps=50,
        output_dir=output_path,
        learning_rate=2e-4,
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset["train"],
        args=args,
    )
    trainer.train()

    model.save_pretrained(output_path)

if __name__ == "__main__":
    #train_adapter("../models/llama2_nf4", "prepared_dataset", "../adapters/email_tone")
    train_adapter("../models/tinyllama_cpu", "prepared_dataset", "../adapters/email_tone")

