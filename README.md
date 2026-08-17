---
license: apache-2.0
license_link: LICENSE
library_name: colpali
pipeline_tag: visual-document-retrieval
base_model:
  - Qwen/Qwen3.5-4B
tags:
  - visual-document-retrieval
  - colpali
  - colbert
  - late-interaction
  - multi-vector
  - qwen3_5
  - multilingual
  - safetensors
language:
  - multilingual
---

<div align="center">

# EVIE Preview 4.5B

**Next-Generation Multilingual Visual Document Retrieval with Ultra-Compact Token Embeddings**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Base Model](https://img.shields.io/badge/Base%20Model-Qwen3.5--4B-purple.svg)](https://huggingface.co/Qwen/Qwen3.5-4B)
[![Embedding Dim](https://img.shields.io/badge/Token%20Dim-128%20(Native)-success.svg)](#footprint)
[![ViDoRe V3](https://img.shields.io/badge/ViDoRe%20V3-64.40%20(Rank%20%231)-gold.svg)](#vidore-v3-8-public-domains-ndcg10)
[![Framework](https://img.shields.io/badge/Framework-ColPali--Engine-orange.svg)](https://github.com/illuin-tech/colpali)

[Overview](#overview) • [Benchmark Results](#benchmark-results) • [Quick Start](#quick-start) • [Reproducing](#reproducing) • [Training Data](#training-data) • [Acknowledgements](#acknowledgements)

</div>

---

## Overview

**EVIE Preview 4.5B** is a high-performance multilingual Visual Document Retrieval (VDR) model built upon **Qwen3.5-4B**. It employs ColBERT-style late interaction with native **128-dimensional** token embeddings (4.54B parameters, BF16).

By combining native GatedDeltaNet + full-attention hybrid modeling with compact projection design, EVIE achieves top-tier performance across both ViDoRe V1+V2 and ViDoRe V3 benchmarks—while reducing vector index footprint by **up to 32×** compared to standard dense VLM retrievers.

> **Note**: This is a preview release published while the next iteration of EVIE is training.

### Key Highlights

- **🎯 Ultra-Compact 128D Index**: Native 128-dimensional multi-vector representations drastically shrink downstream storage and index latency without sacrifice in retrieval precision.
- **🏆 SOTA on ViDoRe Benchmarks**: Outperforms larger 8B models on ViDoRe V3 (leading 7 of 8 public domains) and delivers top average accuracy on ViDoRe V1+V2.
- **🌐 Robust Multilingual & Multi-Format**: Strong zero-shot generalization across diverse languages (EN, FR, DE, IT, ES, PT, etc.) and visual formats (charts, tables, scientific reports, financial filings).
- **⚡ Plug-and-Play Integration**: Seamlessly compatible with the `colpali-engine` ecosystem and standard late-interaction scoring pipelines.

---

## Benchmark Results

### Footprint & Efficiency Comparison

| Model | Parameters | Token Embedding Dim | Relative Index Size |
| :--- | :---: | :---: | :---: |
| **EVIE Preview 4.5B** | **4.54B** | **128** | **1.0× (Baseline)** |
| [colqwen3.5-4.5B-v3](https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3) | 4.60B | 320 | 2.5× |
| [jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 3.90B | 2048 (128 multi-vec) | 1.0× ~ 16.0× |
| [nemotron-colembed-vl-4b-v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) | 4.80B | 2560 | 20.0× |
| [llama-nemotron-colembed-vl-3b-v2](https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2) | 4.40B | 3072 | 24.0× |
| [nemotron-colembed-vl-8b-v2](https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2) | 8.77B | 4096 | 32.0× |

---

### ViDoRe V3: 8 Public Domains (nDCG@10)

Evaluated across 8 domains with queries spanning 6 languages (EN, FR, DE, IT, PT, ES):

| Model | **Avg** | CompSci | Energy | Finance (EN) | Finance (FR) | HR | Industrial | Pharma | Physics |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EVIE Preview 4.5B** | **64.40** | **80.33** | **71.45** | **67.68** | **53.01** | 65.65 | **57.48** | **68.66** | **50.96** |
| nemotron-colembed-vl-8b-v2 | 63.54 | 79.30 | 69.82 | 67.29 | 51.54 | **66.32** | 56.03 | 67.19 | 50.84 |
| tomoro-colqwen3-embed-8b | 61.60 | 75.35 | 68.41 | 65.08 | 49.10 | 63.98 | 54.41 | 66.36 | 50.13 |
| nemotron-colembed-vl-4b-v2 | 61.42 | 78.56 | 67.48 | 65.02 | 49.01 | 62.39 | 53.91 | 66.10 | 48.86 |
| tomoro-colqwen3-embed-4b | 60.16 | 75.44 | 66.43 | 63.84 | 46.83 | 60.09 | 53.58 | 65.74 | 49.32 |
| llama-nemotron-colembed-vl-3b-v2 | 59.70 | 77.09 | 64.88 | 64.23 | 44.41 | 62.28 | 51.71 | 66.04 | 46.93 |
| colnomic-embed-multimodal-7b | 57.64 | 76.20 | 63.58 | 56.57 | 45.46 | 58.67 | 50.13 | 62.26 | 48.25 |
| jina-embeddings-v4 | 57.54 | 71.81 | 63.50 | 59.30 | 46.10 | 59.53 | 50.38 | 63.09 | 46.63 |

> EVIE leads in **7 out of 8** public domains.

---

### ViDoRe V1 + V2 (nDCG@5)

| Model | ArxivQA | DocVQA | InfoVQA | ShiftProj | SynAI | SynEnergy | SynGov | SynHealth | Tabfquad | Tatdqa | BioMed | ESGHL | ESG | Econ | **Avg** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EVIE Preview 4.5B** | 91.5 | 62.9 | 93.0 | **94.0** | **100.0** | **99.0** | **98.9** | 98.9 | **97.5** | 81.6 | **71.0** | **80.1** | **66.3** | **68.3** | **85.9** |
| Ops-Colqwen3-4B | 91.8 | 66.5 | 94.0 | 90.8 | 99.6 | 97.3 | 98.0 | 99.6 | 93.6 | 82.4 | 65.5 | 78.6 | 66.0 | 64.5 | 84.9 |
| nemotron-colembed-vl-8b-v2 | 93.1 | 68.1 | 94.6 | 93.3 | 100.0 | 97.9 | 98.9 | 99.6 | 97.7 | 83.4 | 66.2 | 73.2 | 60.6 | 60.8 | 84.8 |
| nemotron-colembed-vl-4b-v2 | 92.0 | 67.4 | 93.3 | 92.3 | 99.3 | 96.2 | 98.0 | 98.5 | 98.1 | 81.2 | 64.3 | 71.4 | 61.5 | 60.8 | 83.9 |
| colqwen3.5-4.5B-v3 | 91.9 | 66.6 | 93.6 | 90.2 | 100.0 | 97.1 | 97.3 | 98.9 | 95.9 | 84.0 | 65.3 | 73.8 | 58.0 | 59.9 | 83.7 |
| llama-nemotron-colembed-vl-3b-v2 | 90.4 | 67.2 | 94.7 | 92.0 | 100.0 | 98.0 | 98.0 | 98.9 | 97.3 | 81.0 | 63.2 | 73.1 | 58.6 | 58.6 | 83.6 |
| tomoro-colqwen3-embed-8b | 91.2 | 66.4 | 94.5 | 87.9 | 99.3 | 96.7 | 97.6 | 99.1 | 94.2 | 80.9 | 65.5 | 76.0 | 60.7 | 59.5 | 83.5 |
| EvoQwen2.5-VL-Retriever-7B-v1 | 91.5 | 65.1 | 94.1 | 88.8 | 99.6 | 96.6 | 96.3 | 98.9 | 93.6 | 82.3 | 65.2 | 77.0 | 59.7 | 59.1 | 83.4 |
| tomoro-colqwen3-embed-4b | 90.6 | 66.3 | 94.3 | 87.4 | 99.3 | 96.9 | 97.2 | 99.6 | 94.3 | 79.9 | 65.4 | 74.6 | 62.4 | 56.3 | 83.2 |
| llama-nemoretriever-colembed-3b-v1 | 88.4 | 66.2 | 94.9 | 90.7 | 99.6 | 96.6 | 97.8 | 99.3 | 95.9 | 80.6 | 62.7 | 75.4 | 57.4 | 57.8 | 83.1 |
| SauerkrautLM-ColQwen3-8b-v0.1 | 93.8 | 64.7 | 94.5 | 90.4 | 98.6 | 96.5 | 96.8 | 99.3 | 92.2 | 84.0 | 63.3 | 70.8 | 57.9 | 58.0 | 82.9 |

*Columns 1–10: ViDoRe V1 (10 tasks). Columns 11–14: ViDoRe V2 (4 tasks). `Avg`: Unweighted mean across all 14 tasks.*

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Python Inference

```python
import torch
from PIL import Image
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

model_id = "EVIE-Preview-4.5B"

# 1. Load model and processor
model = ColQwen3_5.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",  # or "flash_attention_2"
).eval()

# 2. Enable bidirectional attention (Required)
model.enable_bidirectional_attention()

processor = ColQwen3_5Processor.from_pretrained(model_id)

# 3. Prepare inputs
images = [Image.open("document_page.png")]
queries = ["What key insights are presented on this page?"]

image_batch = processor.process_images(images).to(model.device)
query_batch = processor.process_queries(queries).to(model.device)

# 4. Compute embeddings and similarity score
with torch.inference_mode():
    image_embeddings = model(**image_batch)
    model.rope_deltas = None  # Reset RoPE deltas before query pass
    query_embeddings = model(**query_batch)

scores = processor.score(query_embeddings, image_embeddings)
print("Late-interaction score:", scores)
```

> **Important**: Both `model.enable_bidirectional_attention()` and resetting `model.rope_deltas = None` prior to query forward passes are required to replicate the checkpoint's full retrieval performance.

### CLI Scoring Tool

You can also use the included [`infer.py`](infer.py) script directly:

```bash
python infer.py --query "Quarterly revenue report" --image document_page_1.png --image document_page_2.png
```

---

## Reproducing

Run the end-to-end evaluation benchmark across all visible GPUs:

```bash
bash reproduce.sh
```

### Notes

- **Automatic Dataset Download**: On the first run, `reproduce.sh` automatically invokes [`download_data.py`](download_data.py) to fetch the 22 public ViDoRe datasets (~55 GB) from Hugging Face.
- **Custom Dataset Path**: To reuse an existing dataset directory, pass it directly:
  ```bash
  bash reproduce.sh /path/to/vidore
  ```
- **Target Aggregates**:

```text
ViDoRe V1        nDCG@5    91.72  (10 tasks)
ViDoRe V2        nDCG@5    71.44  (4 tasks)
ViDoRe V1+V2     nDCG@5    85.93  (14 tasks)
ViDoRe V3 public nDCG@10   64.40  (8 domains x 6 languages)
```

---

## Training Data

EVIE was trained on approximately **0.8 million high-quality image-query pairs** covering multilingual documents, technical reports, complex financial tables, infographics, and document visual QA.

### Hard Negative Mining & Data Filtering

- **Dynamic Mining & Verification**: Hard negatives are actively mined using intermediate retrievers and re-verified:
  - Candidates that accurately answer the query are promoted to **positives**.
  - Partially relevant or ambiguous candidates are **masked** out of the loss.
  - Only strictly irrelevant pages are retained as true **hard negatives**.
- **Quality Filtering**: Rows containing empty queries, corrupted images, or degraded text are systematically discarded.

---

## Acknowledgements

- Built upon the [ColPali Engine](https://github.com/illuin-tech/colpali) developed by Illuin Technology.
- Powered by the [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) vision-language backbone.
- Evaluated on the [ViDoRe Benchmark](https://huggingface.co/vidore) family.
