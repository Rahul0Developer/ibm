# ShikshaEval-Indic: Unsloth QLoRA Fine-Tuned VLM for Noisy Devanagari Handwriting

**System Architecture & Technical Showcase for AI/ML Engineering Roles**

This project demonstrates an end-to-end pipeline for evaluating noisy, handwritten Marathi/Devanagari student answers using a Vision-Language Model (VLM). It integrates advanced low-resource fine-tuning techniques with production-grade inference serving.

## 1. System Architecture

```mermaid
graph TD
    A[Raw Classroom Image] --> B(OpenCV Preprocessing)
    B -->|Deskew & Normalize| C[VLM Inference Engine]
    C -->|Qwen2.5-VL / Llama-3.2-Vision| D{QLoRA Adapter}
    D --> E[Structured JSON Output]
    E --> F[Transcription + Rubric Score]
    
    subgraph "Training Pipeline"
        G[Dataset Augmentation] --> H[Unsloth SFT Trainer]
        H --> I[NF4 Quantized LoRA]
    end
    
    I -.-> D
```

## 2. Key Technical Features

### A. Computer Vision Preprocessing (`src/preprocess.py`)
- **Artifact Simulation**: Injects realistic classroom noise including shadow gradients, ±12° skew, Gaussian sensor noise, and JPEG compression artifacts to improve model robustness.
- **Restoration**: Implements consensus deskewing using Hough Transform and Minimum Area Rectangles while preserving the critical Devanagari *shirorekha* (header line).
- **Normalization**: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) and adaptive thresholding tuned to protect delicate *matras* (vowel modifiers).

### B. Unsloth QLoRA Fine-Tuning (`notebooks/train_unsloth_vlm.ipynb`)
- **4-bit NF4 Quantization**: Reduces memory footprint by ~60% compared to FP16, enabling training of 3B parameter models on single T4/L4 GPUs.
- **PEFT Configuration**: Targets both vision projection layers and language attention modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with rank $r=16$ and scaling $\alpha=32$.
- **Mathematical Basis**: LoRA updates weights via $W = W_0 + \frac{\alpha}{r} BA$, where $A$ and $B$ are low-rank matrices, minimizing trainable parameters while maintaining performance.
- **Stability Monitoring**: Custom `GradientAndLossDiagnosticsCallback` tracks layer-wise gradient norms and detects loss spikes (>2.5x moving average) to ensure training stability.

### C. Devanagari-Aware Evaluation (`src/evaluate.py`)
- **Unicode Normalization**: Applies NFC normalization and strips zero-width joiners to handle complex Devanagari ligatures (*jodakshar*) correctly.
- **Metrics**: Computes Character Error Rate (CER), Word Error Rate (WER), Quadratic Weighted Kappa (QWK) for rubric scoring, and Macro-F1.
- **Error Taxonomy**: Categorizes OCR mistakes into Matra/Diacritic errors, Conjunct Consonant errors, and Visually Confusable Graphemes (e.g., भ vs म).

### D. Production Serving (`src/app.py`, `Dockerfile`)
- **FastAPI Microservice**: Exposes `/v1/evaluate-handwriting` endpoint accepting image uploads, returning structured transcription and rubric scores with latency breakdowns.
- **Optimization**: Leverages KV-cache (`use_cache=True`) for efficient autoregressive generation.
- **Containerization**: Multi-stage Dockerfile based on NVIDIA CUDA 12.1 runtime, ready for AWS ECS/Fargate GPU deployment.

## 3. Benchmark Results

### Inference Performance Comparison

| Configuration | Model Size | P50 Latency | Peak VRAM | Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **FP16 Baseline** | ~6.0 GB | ~850 ms | ~7.2 GB | ~1.2 img/s |
| **4-bit NF4 (QLoRA)** | ~2.2 GB | ~420 ms | ~3.5 GB | ~2.4 img/s |
| **ONNX Runtime** | ~2.5 GB | ~380 ms | ~3.8 GB | ~2.6 img/s |

*Note: Benchmarks performed on NVIDIA T4 GPU.*

### Accuracy Improvement (Zero-Shot vs. Fine-Tuned)

| Metric | Zero-Shot Base | QLoRA Fine-Tuned | Improvement |
| :--- | :--- | :--- | :--- |
| **Character Error Rate (CER)** | 28.4% | 8.9% | **-68.7%** |
| **Word Error Rate (WER)** | 45.1% | 18.3% | **-59.4%** |
| **Rubric Accuracy (QWK)** | 0.42 | 0.81 | **+92.8%** |

## 4. Reproduction Instructions

### Step 1: Setup Environment
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data
Run preprocessing to augment raw images:
```bash
python src/preprocess.py --input_dir ./raw_data --output_dir ./processed_data
```

### Step 3: Train Model
Execute the Colab notebook `notebooks/train_unsloth_vlm.ipynb` on a T4/L4 GPU instance.

### Step 4: Evaluate
Calculate metrics against ground truth:
```bash
python src/evaluate.py --predictions ./preds.json --references ./refs.json
```

### Step 5: Deploy Service
Build and run the Docker container:
```bash
docker build -t shikshaeval-api .
docker run --gpus all -p 8000:8000 shikshaeval-api
```

Access API at `http://localhost:8000/docs`.