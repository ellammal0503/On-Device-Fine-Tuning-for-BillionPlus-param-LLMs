from transformers import AutoModelForCausalLM, AutoTokenizer

def quantize_model(model_name, save_path):
    print(f"Loading model: {model_name}")

    # Load model on CPU with memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="float32",   # use float32 on CPU
        low_cpu_mem_usage=True,
        device_map="cpu"
    )

    # Save model + tokenizer
    model.save_pretrained(save_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)

    print(f"✅ Model saved to {save_path} (CPU friendly, no NF4 quantization)")

if __name__ == "__main__":
    quantize_model("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "../models/tinyllama_cpu")
