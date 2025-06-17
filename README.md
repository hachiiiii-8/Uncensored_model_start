# Project Name

## Overview
This repository focuses on studying the safety alignment of large language models, exploring how shallow alignment and deep alignment can improve model safety. The project includes multiple experimental modules for evaluating and optimizing model performance in various scenarios.

---

## Features
### 1. Safety Evaluation
- **DeepSeekEvaluator**: Uses a local model to score the safety of generated content.
- **ChatgptEvaluator**: Leverages GPT-4 API to evaluate the safety of generated content.

### 2. Data Processing
- Provides tools for automatic dataset downloading and preprocessing.
- Supports multiple safety benchmarks (e.g., `hex-phi`).

### 3. Model Fine-Tuning
- Supports shallow and deep alignment fine-tuning methods.
- Offers quantization options (e.g., 8-bit and FP16) to optimize memory usage.

---

## File Structure
```
├── shallow-vs-deep-alignment
│   ├── finetuning_buckets
│   │   ├── inference
│   │   │   ├── safety_eval
│   │   │   │   ├── chatgpt_judge.py
│   │   │   │   ├── deepseek_judge.py
│   ├── models
│   │   ├── get_model.py
│   ├── eval_safety.py
├── README.md
```

---

## Quick Start

### Environment Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure the following libraries are installed:
   - `transformers`
   - `torch`
   - `huggingface_hub`
   - `numpy`

### Dataset Download
Run the following command to download the required dataset:
```bash
python download_dataset.py
```

### Model Loading
Supported models:
- Llama-2-7b-chat
- Gemma-1.1-7b-it
- Qwen-8B
- Llama-3-8B-Lexi-Uncensored

---

## Usage

### Safety Evaluation

### Utility Evaluation

```
