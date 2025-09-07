# Android App (PocketLoRA)

This folder contains the **Android application source code** for PocketLoRA.

---

## Structure

app/
│
├── src/ # Main application source
│ ├── main/
│ │ ├── java/com/pocketlora/ # Kotlin/Jetpack Compose code
│ │ │ ├── ui/ # UI screens (training, adapters, inference)
│ │ │ ├── manager/ # Adapter lifecycle & storage manager
│ │ │ ├── scheduler/ # Thermal & power-aware scheduler
│ │ │ └── inference/ # Runtime integration (ExecuTorch/ONNX)
│ │ └── res/ # Layouts, drawables, themes
│ │
│ ├── androidTest/ # Instrumentation tests
│ └── test/ # Unit tests
│
├── build.gradle # Gradle build config for the app
├── settings.gradle # Project-level Gradle settings
└── AndroidManifest.xml # App permissions, entry points, intents



---

## Key Components

- **UI (Jetpack Compose)**  
  Screens for:
  - Training progress  
  - Adapter selection & management  
  - Inference chat window  

- **Adapter Manager**  
  Handles loading, attaching, and exporting adapters.

- **Scheduler**  
  Ensures training only runs under safe conditions (charging, Wi-Fi, screen off).

- **Inference Runtime**  
  Connects with ExecuTorch / ONNX Runtime Mobile for LLM inference.

---

## Build Instructions

1. Open this folder in **Android Studio**.  
2. Sync Gradle dependencies.  
3. Build and install:  
   ```bash
   ./gradlew assembleDebug
   adb install build/outputs/apk/debug/app-debug.apk





# 📱 PocketLoRA – On-Device Fine-Tuning for Billion+ Parameter LLMs

PocketLoRA is a lightweight framework that enables **on-device fine-tuning** of billion+ parameter Large Language Models (LLMs) on smartphones (Galaxy S23–S25 class).  
It allows users to adapt pre-trained LLMs to their **personal data** with **low-power training**, **hot-swappable adapters**, and **strict on-device privacy**.

---

## 🚩 Problem Statement

Traditional LLM fine-tuning requires **expensive GPUs** and **cloud infrastructure**.  
Users cannot personalize models without exposing sensitive data to servers.  

**Challenge:** How can we enable **efficient fine-tuning of billion+ parameter LLMs** directly on smartphones, within **tight compute, memory, and power budgets**?

---

## 💡 Our Solution

PocketLoRA introduces an **efficient, privacy-preserving on-device fine-tuning framework**:

- **LoRA/QLoRA adapters** → train only a few million parameters, not the full model.  
- **NF4 quantization** → compresses models for mobile memory budgets.  
- **Thermal & power-aware scheduler** → trains only under safe conditions.  
- **Hot-swap adapters** → instantly enable/disable task-specific adapters.  
- **ExecuTorch / ONNX Mobile runtime** → optimized inference on-device.  

---

## 🌟 Features

- 🔋 **Low-Power Fine-Tuning** → runs safely under charging/Wi-Fi/cool temperature  
- 🔌 **Hot-Swap Adapters** → small files, enable/disable at runtime  
- 🔒 **Privacy by Design** → all training is fully on-device  
- ⚡ **Optimized Runtime** → fast inference with ExecuTorch / ONNX  
- 📱 **User-Friendly UI** → training, adapter management, and inference screens  

See full [features.md](docs/features.md).

---

## 🛠️ Tech Stack

- **Frameworks**: PyTorch, Hugging Face Transformers, PEFT, bitsandbytes  
- **Quantization**: NF4 / QLoRA  
- **Runtime**: ExecuTorch / ONNX Runtime Mobile  
- **App**: Kotlin, Jetpack Compose, Android Studio  
- **Scheduling**: Power & thermal-aware background job manager  

See [technical_stack.md](docs/technical_stack.md).

---

## 📂 Repository Structure

app/ # Android app (UI, adapter manager, inference)
tools/ # Python training scripts (LoRA/quantization)
models/ # Base quantized model (ignored in Git)
adapters/ # Trained LoRA adapters
docs/ # Technical documentation & diagrams
images/ # UI screenshots and diagrams



---

## ⚡ Quickstart

### 1. Clone Repo
```bash
git clone https://github.com/<your-username>/pocketlora.git
cd pocketlora

2. Python Environment (training tools)
python3 -m venv .venv
source .venv/bin/activate
pip install -r tools/requirements.txt
3. Build Android App
cd app
./gradlew assembleDebug
adb install build/outputs/apk/debug/app-debug.apk
4. Add Models & Adapters
Place quantized model in models/ (not tracked by Git)
Place/collect trained adapters in adapters/
5. Run App
Select dataset → Train adapter → Manage in Adapters screen → Run inference
📸 Screenshots
Training	Adapters	Inference
📜 License
This project is licensed under the MIT License