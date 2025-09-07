# User Guide – PocketLoRA

PocketLoRA allows you to fine-tune large language models (LLMs) **directly on your smartphone**.  
This guide walks through the typical user workflow step by step.

---

## 🚀 Demo Workflow

### 1️⃣ Select Dataset
- Open the **Training Screen**.
- Pick a dataset file (e.g., `user_emails.txt` or `custom_notes.json`).
- Hit **Train Adapter** to start.

📸 Screenshot:  
![Training UI](images/training_ui.png)

---

### 2️⃣ Train Adapter
- The model trains using **LoRA fine-tuning**.  
- Progress bar shows training steps and estimated time.
- Training only runs if device is **charging + Wi-Fi on + cool temperature**.

📸 Screenshot:  
![Training Progress](images/training_ui.png)  
*(Progress bar visualized)*

---

### 3️⃣ Manage Adapters
- Switch to the **Adapters Screen**.
- View the list of trained adapters (e.g., `EmailTone`, `Summarizer`).  
- Options: **Enable**, **Disable**, **Export**, **Delete**.

📸 Screenshot:  
![Adapters UI](images/adapters_ui.png)

---

### 4️⃣ Run Inference
- Go to the **Inference Screen**.  
- Type a prompt (e.g., *"Write a professional reply to this email"*).  
- The active adapter personalizes the response.  

📸 Screenshot:  
![Inference UI](images/inference_ui.png)

---

## 📦 Folder Structure

adapters/ # Saved LoRA adapters
models/ # Quantized base models (not checked in)
app/ # Android app source
tools/ # Preprocessing & training scripts
images/ # UI screenshots for documentation

---

---

## 🎯 Example Use Cases
- **Personal Email Assistant** → trains on your past emails to mimic your writing tone.  
- **Meeting Summarizer** → adapts to summarize long notes into bullet points.  
- **Creative Writing** → adapts LLM to your unique style.  

---

## 🔒 Privacy Guarantee
- Training happens **100% on-device**.  
- **No cloud upload** of personal data.  
- You remain in control of your adapters.  

---
