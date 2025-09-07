# Approach

## Problem Statement
Efficient framework for the on-device fine-tuning of Billion+ scale Large Language Models on a Galaxy S23-S25 equivalent smartphone/edge device.

## Our Approach
We propose **PocketLoRA**, a mobile-first fine-tuning framework using LoRA/QLoRA adapters with memory-mapped 4-bit quantized weights.  
This enables private, efficient, and scalable adaptation of 1–7B parameter LLMs directly on smartphones.

### Why Unique
- **100% On-Device**: No personal data leaves the device, ensuring privacy.  
- **Portable Adapters**: Lightweight (<150 MB) and swappable for different tasks (e.g., email tone, chat style).  
- **Thermal-Aware Scheduling**: Training only runs during charging + idle conditions to avoid overheating.  
- **Optimized for Mobile Hardware**: Designed to leverage Snapdragon 8 Gen 2/3 and UFS 4.0 storage efficiently.  
