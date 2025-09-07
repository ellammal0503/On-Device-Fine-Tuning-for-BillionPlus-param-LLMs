# Technical Stack

Our solution is built entirely on open-source technologies. Below is the stack we used:

## Core ML Frameworks
- [PyTorch](https://pytorch.org/) – Deep learning framework for model training & fine-tuning.  
- [ExecuTorch](https://pytorch.org/executorch/) – PyTorch runtime for efficient on-device inference & training.  
- [ONNX Runtime Training](https://onnxruntime.ai/) – Lightweight runtime with mobile training support.  
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) – Model loading, tokenization, and inference utilities.  
- [PEFT (LoRA/QLoRA)](https://huggingface.co/docs/peft) – Parameter-efficient fine-tuning methods.  

## Mobile & App Development
- [Android Jetpack Compose](https://developer.android.com/jetpack/compose) – Modern Android UI toolkit.  
- [Kotlin](https://kotlinlang.org/) – Programming language for Android app implementation.  

## Supporting Tools
- [Framer Motion](https://www.framer.com/motion/) – For UI animations and diagrams (optional).  
- [Matplotlib](https://matplotlib.org/) – For evaluation metric visualizations (loss curves, memory usage, etc.).  

---

## Why this Stack?
- Optimized for **on-device resource constraints** (memory, power, thermals).  
- Fully **open-source and community-supported**.  
- Cross-platform compatibility (works on Galaxy S23–S25, future edge devices).  

Draft a polite follow-up email
4. Response will include fine-tuned style.

---

## 5. Managing Adapters

- Enable/disable adapters.
- Export adapters (binary only, no raw text).
- Delete adapters securely from device.

---

## 6. Troubleshooting

- **Model not loading?**
- Ensure quantized `.nf4` model exists in `/app/models`.
- **Training stopped early?**
- Device overheated → retry later.
- **High battery drain?**
- Train only while charging.