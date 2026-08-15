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

# EVIE Preview 4.5B

Multilingual visual document retrieval built on Qwen3.5-4B, using ColBERT-style late interaction with native **128-dimensional** token embeddings. 4.54B parameters, BF16, Apache-2.0.

It takes the highest average on both ViDoRe boards in the comparison below, with the narrowest token embeddings in the table and roughly half the parameters of the runner-up on ViDoRe V3.

> **Preview release.** This is an early checkpoint, published while the next EVIE model is already training. Expect it to be superseded.

## Footprint

| Model | Parameters | Token embedding dimension |
| --- | ---: | ---: |
| **EVIE Preview 4.5B** | **4.54B** | **128** |
| [nemotron-colembed-vl-8b-v2](https://huggingface.co/nvidia/nemotron-colembed-vl-8b-v2) | 8.77B | 4096 |
| [nemotron-colembed-vl-4b-v2](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2) | 4.8B | 2560 |
| [llama-nemotron-colembed-vl-3b-v2](https://huggingface.co/nvidia/llama-nemotron-colembed-vl-3b-v2) | 4.4B | 3072 |
| [colqwen3.5-4.5B-v3](https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3) | 4.6B | 320 |
| [jina-embeddings-v4](https://huggingface.co/jinaai/jina-embeddings-v4) | 3.9B | 2048 (128 multi-vector) |

Narrower token vectors shrink a stored multi-vector index proportionally, at equal token counts and precision. Nothing in this table scores higher on either board; the closest model on ViDoRe V3 is 1.9x the size and stores 32x wider token vectors.

## ViDoRe V1 + V2 (nDCG@5)

| Model | ArxivQA | DocVQA | InfoVQA | ShiftProj | SynAI | SynEnergy | SynGov | SynHealth | Tabfquad | Tatdqa | BioMed | ESGHL | ESG | Econ | **Avg** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **EVIE Preview 4.5B** | 91.5 | 62.9 | 93.0 | 94.0 | 100.0 | 99.0 | 98.9 | 98.9 | 97.5 | 81.6 | 71.0 | 80.1 | 66.3 | 68.3 | **85.9** |
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

First ten columns are V1, last four are V2; `Avg` is the unweighted mean of the 14 tasks. The lead comes from the zero-shot V2 sets, where EVIE averages 71.4 against 65.2 for Nemotron ColEmbed 8B v2. On V1 alone that model is ahead, 92.7 to 91.7.

## ViDoRe V3, 8 public domains (nDCG@10)

| Model | **Avg** | CompSci | Energy | FinanceEn | FinanceFr | HR | Industrial | Pharma | Physics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **EVIE Preview 4.5B** | **64.40** | 80.33 | 71.45 | 67.68 | 53.01 | 65.65 | 57.48 | 68.66 | 50.96 |
| nemotron-colembed-vl-8b-v2 | 63.54 | 79.30 | 69.82 | 67.29 | 51.54 | 66.32 | 56.03 | 67.19 | 50.84 |
| tomoro-colqwen3-embed-8b | 61.60 | 75.35 | 68.41 | 65.08 | 49.10 | 63.98 | 54.41 | 66.36 | 50.13 |
| nemotron-colembed-vl-4b-v2 | 61.42 | 78.56 | 67.48 | 65.02 | 49.01 | 62.39 | 53.91 | 66.10 | 48.86 |
| tomoro-colqwen3-embed-4b | 60.16 | 75.44 | 66.43 | 63.84 | 46.83 | 60.09 | 53.58 | 65.74 | 49.32 |
| llama-nemotron-colembed-vl-3b-v2 | 59.70 | 77.09 | 64.88 | 64.23 | 44.41 | 62.28 | 51.71 | 66.04 | 46.93 |
| colnomic-embed-multimodal-7b | 57.64 | 76.20 | 63.58 | 56.57 | 45.46 | 58.67 | 50.13 | 62.26 | 48.25 |
| jina-embeddings-v4 | 57.54 | 71.81 | 63.50 | 59.30 | 46.10 | 59.53 | 50.38 | 63.09 | 46.63 |

Leads 7 of the 8 public domains, trailing only on HR. The two held-out private ViDoRe V3 tasks are not included and no full-benchmark rank is claimed.

## Usage

```bash
pip install -r requirements.txt
```

```python
import torch
from PIL import Image
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

model_id = "EVIE-Preview-4.5B"

model = ColQwen3_5.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
).eval()
model.enable_bidirectional_attention()

processor = ColQwen3_5Processor.from_pretrained(model_id)

images = [Image.open("document_page.png")]
queries = ["What information is shown on this page?"]

image_batch = processor.process_images(images).to(model.device)
query_batch = processor.process_queries(queries).to(model.device)

with torch.inference_mode():
    image_embeddings = model(**image_batch)
    model.rope_deltas = None
    query_embeddings = model(**query_batch)

print(processor.score(query_embeddings, image_embeddings))
```

Calling `enable_bidirectional_attention()` and clearing `rope_deltas` before a query forward pass are both required to reproduce this checkpoint's retrieval behavior. [`infer.py`](infer.py) wraps the same steps as a command line tool.

## Reproducing

```bash
bash reproduce.sh /path/to/vidore
```

Runs on every visible GPU and writes a timestamped log. It prints one line per task and ends with the four aggregates:

```
ViDoRe V1        nDCG@5    91.72  (10 tasks)
ViDoRe V2        nDCG@5    71.44  (4 tasks)
ViDoRe V1+V2     nDCG@5    85.93  (14 tasks)
ViDoRe V3 public nDCG@10   64.40  (8 domains x 6 languages)
```

The expected dataset layout is documented at the top of [`reproduce.py`](reproduce.py).

## Training data

Roughly 0.8 million image-query pairs covering multilingual documents, reports, tables, charts, and document-question answering.

Hard negatives are mined with a trained retriever and then re-judged rather than trusted. A mined page that actually answers the query is promoted to a positive, a partially relevant one is masked out of the loss, and only genuinely irrelevant pages stay as negatives. Rows with an empty query or an unusable image are dropped.

The full data pipeline is documented in the technical report that accompanies the full release.

## Acknowledgements

Trained with the [ColPali engine](https://github.com/illuin-tech/colpali) from Illuin Technology — this release depends directly on its late-interaction training framework and `ColQwen3_5` implementation. Thanks also to [Qwen](https://huggingface.co/Qwen/Qwen3.5-4B) for the backbone and to the [ViDoRe](https://huggingface.co/vidore) authors for the benchmarks.
