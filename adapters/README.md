
---

## **`adapters/README.md`**
```markdown
# Adapters Directory

This folder stores **LoRA/QLoRA adapters** trained on-device or via scripts.

---

## Expected Files
- `email_tone.bin` → Adapter for polite email drafting.
- `summarizer.bin` → Adapter for summarization tasks.
- `code_helper.bin` → Adapter for code explanation.

---

## Notes
- Adapters are **small (20–150 MB)** and safe to check in if needed.
- Exported adapters contain **only weight deltas**, no raw training text.
- Example training workflow:
  ```bash
  python tools/train_adapter.py \
    --dataset /path/to/user_data.txt \
    --model models/base_model.nf4 \
    --output adapters/email_tone.bin
