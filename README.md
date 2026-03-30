# 🛡️ FinTune-Sentinel: SEC-Compliant Financial Auditor
> **A high-efficiency 3B parameter LLM optimized for financial reasoning and regulatory safety on commodity hardware.**

---

## 📌 Overview
**FinTune-Sentinel** is a specialized financial intelligence engine designed to solve the "hallucination problem" in automated 10-K analysis. While general LLMs often provide vague or unformatted financial data, Sentinel acts as a **Senior Financial Analyst**—providing deterministic, unit-accurate, and compliant audits.

### Key Innovations:
* **CPU-Optimized Inference:** Custom-compiled `llama.cpp` core achieving **10-15 tokens/sec** on standard CPUs (No GPU required).
* **Policy-Sentinel Layer:** A custom programmatic guardrail that intercepts and audits every response for regulatory compliance.
* **4-Bit Quantization:** Leverages the **GGUF (Q4_K_M)** format to reduce memory footprint by 75% without sacrificing analytical depth.

---

## 🚀 The "Sentinel" Tech Stack
* **Base Model:** Llama 3.2 3B Instruct
* **Fine-Tuning:** QLoRA (4-bit) on specialized FinQA & SEC 10-K datasets.
* **Inference Engine:** `llama-cpp-python` (C++ backend with **AVX2/SIMD** hardware acceleration).
* **Governance:** Regex-based Compliance Auditor (Unit integrity & Trade-advice redaction).
* **UI/UX:** Gradio-based Hybrid Chat & Audit Dashboard.

---

## 📊 Benchmarks: Sentinel-3B vs. Industry Standard (8B)
| Metric | FinGPT-v3 (8B) | **FinTune-Sentinel (3B)** |
| :--- | :--- | :--- |
| **Hardware Requirement** | 8GB+ VRAM (GPU) | **Standard Laptop (CPU)** |
| **YoY Revenue Growth** | Frequent Timeouts | **✅ Precise Calculation** |
| **Solvency Analysis** | Generic/Vague | **✅ Detailed Ratio Audit** |
| **Compliance Status** | Unmonitored | **✅ Real-time Safety Gating** |

---

## 🛡️ The Governance Layer (Policy-Sentinel)
To ensure professional-grade reliability, every response is passed through an automated auditor before reaching the user:
1.  **Numerical Integrity:** Rejects answers containing numbers without proper units ($ or %).
2.  **Liability Mitigation:** Automatically flags and redacts prohibited trade advice (e.g., "BUY", "SELL").
3.  **Deterministic Persona:** Maintains a neutral, analytical tone suitable for professional SEC auditing.

---

## 🛠️ How it stays CPU Efficient
This project achieves high-speed performance on CPUs through three primary optimizations:
1. **Weight Compression:** Quantizing the model to 4-bit lowers RAM bandwidth bottlenecks.
2. **SIMD Vectorization:** Utilizing **AVX2/FMA** instructions to perform multiple mathematical operations in a single clock cycle.
3. **Memory Mapping (mmap):** Efficiently managing I/O so only required model weights are pulled into active memory.

## 🛠️ Performance Architecture

![LLM CPU Optimization Workflow](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*9io6C9f3o5OiwGHhwhGWSw.png)
> **Image Source:** [Lakshmi Devi Prakash / Kompact AI](https://medium.com/@datascientist.lakshmi/kompact-ai-running-llms-on-cpus-without-sacrificing-performance-e1fe70fc2a6a) — *Demonstrating the SIMD vectorization and mmap-based memory efficiency used in FinTune-Sentinel.*

### 🧠 Why this Architecture Matters
By implementing the "Kompact" style of CPU inference, **FinTune-Sentinel** achieves:
1. **Lazy Loading:** Using `mmap` to map the 2.02GB model file without consuming physical RAM until the moment of inference.
2. **Parallel Math:** Leveraging **AVX2/SIMD** instructions to calculate financial ratios 4x faster than standard Python-based LLMs.

The Training Data: Three Pillars of Financial IQ 🧠
To move beyond general chat, I fine-tuned Sentinel-3B using a specialized data stack:

10-K Comprehensive Q&A: Used Yousef Saeedian’s Financial Q&A-10k (7,000+ expert pairs) to teach the model the "geography" of SEC filings.

Expert Theory & Reasoning: Integrated the Finance-Llama2-1k dataset to master complex concepts like yield-to-maturity and currency hedging.

Numerical Precision (FinQA): Leveraged Visalakshi Iyer’s FinQA collection to ensure the model follows a strict mathematical program when calculating YoY growth or debt-to-equity ratios.

## 🚦 Quick Start
### Prerequisites
* Python 3.10+
* C++ Build Tools (CMake)

### Installation
```bash
git clone [https://github.com/Prakruthi19/FinTune-Sentinel.git](https://github.com/Prakruthi19/FinTune-Sentinel.git)
cd FinTune-Sentinel
pip install -r requirements.txt
python app.py
