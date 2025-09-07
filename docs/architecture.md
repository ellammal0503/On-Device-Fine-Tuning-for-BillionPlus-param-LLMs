# Technical Architecture

This document describes the system-level architecture for **PocketLoRA** — a mobile-first, on-device adapter fine-tuning framework for 1–7B LLMs on Galaxy S23–S25 class devices.

---

## High-level system flow

![System Flow](../screenshots/system_flow.png)

---

## Component overview

### 1. Data Ingestion & Preprocessor
- Sources: user messages, notes, email drafts, typed corrections (user opt-in).
- Actions: tokenization, deduplication, simple normalization, optional anonymization (PII masking).
- Output: locally-stored fine-tuning corpus with metadata (timestamp, source, label).

### 2. Local Dataset Store
- Stores tokenized examples in an indexed format for streaming to the trainer.
- Uses an encrypted SQLite or flat files under app-private storage.
- Supports lightweight holdout split for evaluation (PPL, A/B human eval).

### 3. Adapter Trainer (LoRA/QLoRA)
- Loads base model in **NF4 (4-bit)** quantized format with memory-mapping.
- Initializes LoRA adapters (targetable modules, rank r configurable).
- Training optimizations:
  - Micro-batch (bs = 1–4), gradient accumulation.
  - Mixed-precision accumulators (16-bit), Adafactor / Lion optimizer.
  - Gradient checkpointing and parameter freezing.
  - Early-stop and scheduler hooks for thermal/battery constraints.
- Output: adapter files (small binary blobs), training logs, evaluation metrics.

### 4. Adapter Store
- Adapters are kept separate from base weights; labeled by purpose (email-tone, code-assist).
- Stored encrypted and user-visible (UI: enable/disable, export, delete).
- Export format contains metadata but **no user text**.

### 5. Inference Runtime
- Loads base model (memory-mapped) and merges adapters dynamically at runtime.
- ExecuTorch / ONNX Runtime Mobile are primary runtime options.
- Inference supports hot-swapping adapters per task and fallback to baseline.

### 6. Scheduler / Device Controls
- Scheduler enforces policies:
  - Train only when (charging AND on Wi-Fi AND screen-off) OR when explicitly allowed by user.
  - Observe thermal headroom via platform APIs; pause/resume accordingly.
  - Rate limit total training time to protect battery/thermals.
- Privacy Controller ensures dataset and training logs never leave device.

---

## Memory & storage budgeting (guidance)

Target device: Galaxy S23 / S24 / S25 class (8–12 GB RAM typical).

![Memory Budget](../screenshots/memory_budget.png)

- Base model (1–3B) in 4-bit NF4: **~1.5–4 GB** memory-mapped on UFS (not fully resident in RAM).
- Activations & optimizer peaks: **~1.5–3.5 GB** (depends on seq_len & micro-bsz).
- LoRA adapter (r=8–16): **~20–150 MB** per adapter.
- Working RAM goal: **≤ 6 GB** peak on a 12 GB device (leaves headroom for OS & UI).
- Use mmap for base weights and stream batches to avoid full weight residency.

---

## Scheduler: state machine

```mermaid
stateDiagram-v2
  [*] --> Idle
  Idle --> ReadyToTrain : Charging && WiFi && ScreenOff
  ReadyToTrain --> Training : ThermalOK && UserConsent
  Training --> Cooldown : ThermalHigh || UserStops
  Cooldown --> Idle : CooldownComplete
  Training --> Idle : Completed || Error


Idle: normal app usage, only inference allowed.
ReadyToTrain: candidate state when device meets training preconditions.
Training: active adapter updates (short bursts).
Cooldown: pause for thermal recovery, then return to Idle.

# App UI (example mock)
Shows training state (Idle, Training, Cooldown).
Displays list of available adapters.
Provides Start/Stop training controls.

#Integration points & APIs
Model Loader API
load_base_model(path, mmap=True, quant='nf4')
attach_adapter(adapter_blob) -> runtime_handle
Trainer API
train_adapters(dataset, config) -> adapter_blob, metrics
pause_training() / resume_training()
Scheduler API
can_start_training() -> {allowed: bool, reason: str}
Privacy API
export_adapter(adapter_blob, include_metadata=True) — no raw text export.

Fault tolerance & recovery
Checkpoint adapters after every N steps (configurable), flush to durable storage.
If training interrupted (thermals, crash), resume from last adapter checkpoint.
Provide UI log and a diagnostics export (local file) to help reproducibility.

#Directory & deployment mapping
/app
  /models
    base.nf4          # memory-mapped base model file (not checked-in)
  /adapters
    email_tone.bin
    travel_planner.bin
  /data
    corpus.db         # tokenized examples (encrypted)
  /logs
    train.log
  /ui
    MainActivity.kt