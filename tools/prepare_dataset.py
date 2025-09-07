"""
Prepare user dataset for on-device fine-tuning.
Converts text files into tokenized datasets and saves to disk.
"""

from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_dataset(file_path: str, tokenizer_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = load_dataset("text", data_files={"train": file_path})
    dataset = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length"),
        batched=True
    )

    return dataset

import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python prepare_dataset.py <path_to_text_file>")
    file_path = sys.argv[1]

    ds = prepare_dataset(file_path)

    # ✅ Save to disk so train_adapter.py can use it
    ds.save_to_disk("prepared_dataset")

    print("✅ Dataset prepared and saved to 'prepared_dataset'")
