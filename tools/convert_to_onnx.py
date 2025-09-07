"""
Convert quantized LLM + adapter into ONNX format for ExecuTorch/ONNX Runtime Mobile.
"""

from transformers import AutoModelForCausalLM
import torch

def convert_to_onnx(model_path: str, output_path: str):
    model = AutoModelForCausalLM.from_pretrained(model_path)

    dummy_input = torch.randint(0, 100, (1, 32))  # fake tokenized input
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}},
        opset_version=14,
    )

    print(f"Model exported to {output_path}")

if __name__ == "__main__":
    convert_to_onnx("../adapters/email_tone", "../models/email_tone.onnx")
