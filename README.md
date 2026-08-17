---
language:
- en
- fr
- de
- it
- es
- pt
- zh
- multilingual
license: apache-2.0
library_name: colpali-engine
pipeline_tag: visual-document-retrieval
tags:
- visual-document-retrieval
- vision-language
- colpali
- colbert
- late-interaction
- multi-vector
- qwen3_5
- vidore
- document-retrieval
- multimodal
base_model:
- Qwen/Qwen3.5-4B
datasets:
- vidore/vidore_benchmark
- vidore/vidore_benchmark_v2
inference: false
---

<div align="center">

# EVIE-Preview-4.5B

**Next-Generation Multilingual Visual Document Retrieval with Ultra-Compact Token Embeddings**

<p align="center">
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://huggingface.co/Qwen/Qwen3.5-4B"><img src="https://img.shields.io/badge/Base%20Model-Qwen3.5--4B-purple.svg" alt="Base Model"></a>
  <a href="#model-footprint"><img src="https://img.shields.io/badge/Token%20Dim-128%20(Native)-success.svg" alt="Embedding Dim"></a>
  <a href="#vidore-v3-8-public-domains-ndcg10"><img src="https://img.shields.io/badge/ViDoRe%20V3-64.40%20(Rank%20%231)-gold.svg" alt="ViDoRe V3"></a>
  <a href="https://github.com/illuin-tech/colpali"><img src="https://img.shields.io/badge/Framework-ColPali--Engine-orange.svg" alt="Framework"></a>
  <a href="https://huggingface.co/tencent/EVIE-Preview-4.5B"><img src="https://img.shields.io/badge/Hugging%20Face-Model-FFD21E.svg" alt="Hugging Face"></a>
  <a href="https://github.com/Tencent/EVIE-Preview-4.5B"><img src="https://img.shields.io/badge/GitHub-Source-181717.svg" alt="GitHub"></a>
</p>

[Overview](#overview) • [Architecture](#architecture) • [Benchmark Results](#benchmark-results) • [Quick Start](#quick-start) • [Reproducing](#reproducing) • [Training Details](#training-details) • [Citation](#citation)

</div>

---

## Overview

**EVIE-Preview-4.5B** is a state-of-the-art multilingual Visual Document Retrieval (VDR) model built upon **Qwen3.5-4B**. It employs ColBERT-style late interaction with native **128-dimensional** multi-vector token embeddings (4.54B parameters, BF16).

By combining native GatedDeltaNet linear-attention and full-attention hybrid modeling with a compact visual projection, EVIE achieves top-tier performance across ViDoRe V1+V2 and ViDoRe V3 while generating ultra-compact 128D multi-vectors—cutting vector storage and indexing costs by **8× to 32×** compared to wider 2560D–4096D representations.

> **Note**: This is a preview release. The next iteration of EVIE is pending release.

### Key Highlights

- **🎯 Ultra-Compact 128D Index**: Native 128-dimensional multi-vector representations drastically shrink downstream storage and index latency without sacrifice in retrieval precision.
- **🏆 SOTA on ViDoRe Benchmarks**: Outperforms larger 8B models on ViDoRe V3 (leading **7 of 8** public domains) and delivers top average accuracy on ViDoRe V1+V2 (**85.93** nDCG@5).
- **🌐 Robust Multilingual & Multi-Format**: Strong zero-shot generalization across diverse languages (EN, FR, DE, IT, ES, PT, ZH, etc.) and visual formats (charts, tables, scientific reports, financial filings).
- **⚡ Seamless ColPali Compatibility**: Fully integrated with the standard `colpali-engine` ecosystem and late-interaction scoring pipelines.

---

## Architecture

```text
  Text Query ───────► ColQwen3_5 (BiDir Attn) ─────► Query Token Embeddings (128D)
                                                                 │
                                                       Late Interaction (MaxSim) ──► Relevance Score
                                                                 │
Document Image ─────► ColQwen3_5 (Dynamic Vision) ──► Doc Token Embeddings (128D)
```

1. **Vision-Language Backbone**: Built on `Qwen3.5-4B` with interleaved linear and full attention layers.
2. **Compact Projection**: Projects contextual token states directly into 128-dimensional representations.
3. **Late-Interaction Retrieval**: Calculates similarity via token-level MaxSim operator across query tokens and document visual tokens.

---

## Benchmark Results

### Model Footprint

| Model | Parameters | Token Embedding Dim | Relative Index Size |
| :--- | :---: | :---: | :---: |
| **EVIE-Preview-4.5B** | **4.54B** | **128** | **1.0× (Baseline)** |
| [colqwen3.5-4.5B-v3](https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3) | 4.60B | 320 | 2.5× |
| [jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 3.90B | 2048 (128 multi-vec) | 16.0× |
| [nemotron-colembed-vl-4b-v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) | 4.80B | 2560 | 20.0× |
| [llama-nemotron-colembed-vl-3b-v2](https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2) | 4.40B | 3072 | 24.0× |
| [nemotron-colembed-vl-8b-v2](https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2) | 8.77B | 4096 | 32.0× |

---

### ViDoRe V3: 8 Public Domains (nDCG@10)

Evaluated across 8 domains with queries spanning 6 languages (EN, FR, DE, IT, PT, ES):

| Model | **Avg** | CompSci | Energy | Finance (EN) | Finance (FR) | HR | Industrial | Pharma | Physics |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EVIE-Preview-4.5B** | **64.40** | **80.33** | **71.45** | **67.68** | **53.01** | 65.65 | **57.48** | **68.66** | **50.96** |
| nemotron-colembed-vl-8b-v2 | 63.54 | 79.30 | 69.82 | 67.29 | 51.54 | **66.32** | 56.03 | 67.19 | 50.84 |
| tomoro-colqwen3-embed-8b | 61.60 | 75.35 | 68.41 | 65.08 | 49.10 | 63.98 | 54.41 | 66.36 | 50.13 |
| nemotron-colembed-vl-4b-v2 | 61.42 | 78.56 | 67.48 | 65.02 | 49.01 | 62.39 | 53.91 | 66.10 | 48.86 |
| tomoro-colqwen3-embed-4b | 60.16 | 75.44 | 66.43 | 63.84 | 46.83 | 60.09 | 53.58 | 65.74 | 49.32 |
| llama-nemotron-colembed-vl-3b-v2 | 59.70 | 77.09 | 64.88 | 64.23 | 44.41 | 62.28 | 51.71 | 66.04 | 46.93 |
| colnomic-embed-multimodal-7b | 57.64 | 76.20 | 63.58 | 56.57 | 45.46 | 58.67 | 50.13 | 62.26 | 48.25 |
| jina-embeddings-v4 | 57.54 | 71.81 | 63.50 | 59.30 | 46.10 | 59.53 | 50.38 | 63.09 | 46.63 |

> 🏆 **Result**: EVIE leads in **7 out of 8** public domains on ViDoRe V3.

---

### ViDoRe V1 + V2 (nDCG@5)

| Model | **Avg** | ArxivQA | DocVQA | InfoVQA | ShiftProj | SynAI | SynEnergy | SynGov | SynHealth | Tabfquad | Tatdqa | BioMed | ESGHL | ESG | Econ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EVIE-Preview-4.5B** | **85.9** | 91.5 | 62.9 | 93.0 | **94.0** | **100.0** | **99.0** | **98.9** | 98.9 | **97.5** | 81.6 | **71.0** | **80.1** | **66.3** | **68.3** |
| Ops-Colqwen3-4B | 84.9 | 91.8 | 66.5 | 94.0 | 90.8 | 99.6 | 97.3 | 98.0 | 99.6 | 93.6 | 82.4 | 65.5 | 78.6 | 66.0 | 64.5 |
| nemotron-colembed-vl-8b-v2 | 84.8 | 93.1 | 68.1 | 94.6 | 93.3 | 100.0 | 97.9 | 98.9 | 99.6 | 97.7 | 83.4 | 66.2 | 73.2 | 60.6 | 60.8 |
| nemotron-colembed-vl-4b-v2 | 83.9 | 92.0 | 67.4 | 93.3 | 92.3 | 99.3 | 96.2 | 98.0 | 98.5 | 98.1 | 81.2 | 64.3 | 71.4 | 61.5 | 60.8 |
| colqwen3.5-4.5B-v3 | 83.7 | 91.9 | 66.6 | 93.6 | 90.2 | 100.0 | 97.1 | 97.3 | 98.9 | 95.9 | 84.0 | 65.3 | 73.8 | 58.0 | 59.9 |
| llama-nemotron-colembed-vl-3b-v2 | 83.6 | 90.4 | 67.2 | 94.7 | 92.0 | 100.0 | 98.0 | 98.0 | 98.9 | 97.3 | 81.0 | 63.2 | 73.1 | 58.6 | 58.6 |
| tomoro-colqwen3-embed-8b | 83.5 | 91.2 | 66.4 | 94.5 | 87.9 | 99.3 | 96.7 | 97.6 | 99.1 | 94.2 | 80.9 | 65.5 | 76.0 | 60.7 | 59.5 |
| EvoQwen2.5-VL-Retriever-7B-v1 | 83.4 | 91.5 | 65.1 | 94.1 | 88.8 | 99.6 | 96.6 | 96.3 | 98.9 | 93.6 | 82.3 | 65.2 | 77.0 | 59.7 | 59.1 |
| tomoro-colqwen3-embed-4b | 83.2 | 90.6 | 66.3 | 94.3 | 87.4 | 99.3 | 96.9 | 97.2 | 99.6 | 94.3 | 79.9 | 65.4 | 74.6 | 62.4 | 56.3 |
| llama-nemoretriever-colembed-3b-v1 | 83.1 | 88.4 | 66.2 | 94.9 | 90.7 | 99.6 | 96.6 | 97.8 | 99.3 | 95.9 | 80.6 | 62.7 | 75.4 | 57.4 | 57.8 |
| SauerkrautLM-ColQwen3-8b-v0.1 | 82.9 | 93.8 | 64.7 | 94.5 | 90.4 | 98.6 | 96.5 | 96.8 | 99.3 | 92.2 | 84.0 | 63.3 | 70.8 | 57.9 | 58.0 |

*`Avg`: Unweighted mean across all 14 tasks. Tasks 1–10: ViDoRe V1. Tasks 11–14: ViDoRe V2.*

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

# Use the Hugging Face model repository or local directory
model_id = "tencent/EVIE-Preview-4.5B"

# 1. Load model and enable bidirectional attention
model = ColQwen3_5.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
).eval()
model.enable_bidirectional_attention()

# 2. Load processor
processor = ColQwen3_5Processor.from_pretrained(model_id)

# 3. Prepare inputs
images = [Image.open("document_page.png")]
queries = ["What key insights are presented on this page?"]

image_batch = processor.process_images(images).to(model.device)
query_batch = processor.process_queries(queries).to(model.device)

# 4. Generate multi-vector embeddings and score
with torch.inference_mode():
    image_embeddings = model(**image_batch)
    model.rope_deltas = None  # required before query forward
    query_embeddings = model(**query_batch)

scores = processor.score(query_embeddings, image_embeddings)
print("Late-interaction retrieval scores:", scores)
```

> ⚠️ **Important**: Both `model.enable_bidirectional_attention()` and resetting `model.rope_deltas = None` prior to query forward passes are required to replicate the checkpoint's full retrieval performance.

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

## Training Details

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

---

## Citation

```bibtex
@misc{tencent2026evie,
  title        = {EVIE-Preview-4.5B},
  author       = {{Tencent}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/tencent/EVIE-Preview-4.5B}},
  note         = {Multilingual visual document retrieval with compact multi-vector embeddings}
}
```
