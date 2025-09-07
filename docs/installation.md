# Installation & Setup Guide

Follow these steps to set up the project and build the PocketLoRA app.

---

## 📦 1. Clone Repository

```bash
git clone https://github.com/ellammal0503/On-Device-Fine-Tuning-for-BillionPlus-param-LLMs.git
cd On-Device-Fine-Tuning-for-BillionPlus-param-LLMs

Python Environment (for training tools)
Create a virtual environment and install dependencies:
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install -r tools/requirements.txt
This installs libraries for:
Model quantization (bitsandbytes, transformers, peft)
Training adapters (PyTorch, LoRA frameworks)
📱 3. Android App Build
Open the app/ folder in Android Studio or use Gradle CLI:
cd app
./gradlew assembleDebug
adb install build/outputs/apk/debug/app-debug.apk
⚙️ 4. Models & Adapters
Place quantized base model (.nf4 or .onnx) inside models/
(not included in repo due to size).
Adapters are stored in adapters/. Example:
models/
└── base_model.nf4   # Quantized base model (ignored by Git)

adapters/
├── email_tone.bin
├── summarizer.bin
└── README.md
🚀 5. Run App
Launch the app on your Android device.
Use the Training Screen to fine-tune new adapters.
Manage them in Adapters Screen.
Test in Inference Screen.
