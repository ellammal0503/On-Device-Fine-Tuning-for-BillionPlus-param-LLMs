# Models Directory

This folder is a placeholder for the **quantized base LLMs** used by PocketLoRA.

---

## Expected Files
- `base_model.nf4` → Main NF4-quantized checkpoint.
- (Optional) `base_model.onnx` → ONNX-exported version for ExecuTorch/ONNX Runtime Mobile.

---

## Notes
- ⚠️ **Do not commit large model files** to this repository.
- Instead, download and quantize models locally, then place them here.
- Example (LLaMA-2 1.3B):
  ```bash
  python tools/quantize_nf4.py \
    --model /path/to/llama-2-1b \
    --output models/base_model.nf4
