#!/usr/bin/env python3
"""Reproduce the ViDoRe V1 / V2 / V3 numbers reported in the model card.

Expects the released ViDoRe datasets on disk, one directory per board:

    <data-root>/eval/<task>/data/*.parquet                  ViDoRe V1, QA format
    <data-root>/eval_v2/<task>/{corpus,queries,qrels}/*.parquet
    <data-root>/eval_v3/<domain>/<language>-{corpus,queries,qrels}/*.parquet

V1 follows the official QA protocol: every page of a dataset is a candidate and
queries are deduplicated. V2 and V3 use the released qrels, including graded
relevance. V3 averages the six query languages within each domain; the six
language corpora are identical, so each domain is encoded once.
"""

import argparse
import glob
import io
import json
import math
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
from colpali_engine.utils.maxsim import maxsim_inbatch

V1 = [
    "arxivqa_test_subsampled",
    "docvqa_test_subsampled",
    "infovqa_test_subsampled",
    "shiftproject_test",
    "syntheticDocQA_artificial_intelligence_test",
    "syntheticDocQA_energy_test",
    "syntheticDocQA_government_reports_test",
    "syntheticDocQA_healthcare_industry_test",
    "tabfquad_test_subsampled",
    "tatdqa_test",
]
V2 = ["biomedical_lectures", "economics_reports", "esg_reports", "esg_reports_human_labeled"]
V3 = ["computer_science", "energy", "finance_en", "finance_fr", "hr", "industrial",
      "pharmaceuticals", "physics"]
V3_LANGS = ["english", "french", "german", "italian", "portuguese", "spanish"]
META = ("image_filename", "query", "corpus-id", "id")


def parquets(directory: Path) -> list[str]:
    found = sorted(glob.glob(str(directory / "*.parquet")))
    test = [f for f in found if Path(f).name.startswith("test-")]
    return test or found


def text_rows(paths):
    for path in paths:
        handle = pq.ParquetFile(path)
        columns = [c for c in handle.schema_arrow.names if c != "image"]
        for group in range(handle.num_row_groups):
            yield from handle.read_row_group(group, columns=columns).to_pylist()


def to_pil(value):
    if isinstance(value, dict):
        value = value["bytes"]
    return Image.open(io.BytesIO(value)).convert("RGB")


def pad_cat(chunks):
    width = max(c.shape[1] for c in chunks)
    return torch.cat([F.pad(c, (0, 0, 0, width - c.shape[1])) for c in chunks], 0)


@torch.inference_mode()
def embed_images(model, processor, paths, batch: int):
    out, metadata, pending = [], [], []

    def flush():
        if pending:
            out.append(model(**processor.process_images(pending).to(model.device)))
            pending.clear()

    for path in paths:
        handle = pq.ParquetFile(path)
        names = handle.schema_arrow.names
        columns = ["image"] + [c for c in META if c in names]
        for group in range(handle.num_row_groups):
            table = handle.read_row_group(group, columns=columns)
            images = table.column("image")
            others = {c: table.column(c) for c in columns[1:]}
            for j in range(table.num_rows):
                metadata.append({c: others[c][j].as_py() for c in others})
                pending.append(to_pil(images[j].as_py()))
                if len(pending) == batch:
                    flush()
    flush()
    return pad_cat(out), metadata


@torch.inference_mode()
def embed_queries(model, processor, texts, batch: int):
    out = []
    for start in range(0, len(texts), batch):
        chunk = [t if t.strip() else " " for t in texts[start : start + batch]]
        model.rope_deltas = None
        out.append(model(**processor.process_queries(chunk).to(model.device)))
    return pad_cat(out)


def maxsim(queries, docs, chunk_q=64, chunk_d=256):
    scores = torch.zeros(queries.shape[0], docs.shape[0], device=queries.device)
    for qi in range(0, queries.shape[0], chunk_q):
        for di in range(0, docs.shape[0], chunk_d):
            scores[qi : qi + chunk_q, di : di + chunk_d] = maxsim_inbatch(
                queries[qi : qi + chunk_q].contiguous(), docs[di : di + chunk_d].contiguous()
            )
    return scores


def ndcg(order, gains: dict, k: int) -> float:
    dcg = sum(
        (2 ** gains[doc] - 1) / math.log2(rank + 2)
        for rank, doc in enumerate(order[:k])
        if gains.get(doc, 0.0) > 0
    )
    ideal = sorted((2 ** g - 1 for g in gains.values() if g > 0), reverse=True)[:k]
    best = sum(g / math.log2(rank + 2) for rank, g in enumerate(ideal))
    return dcg / best if best else 0.0


def text_of(record) -> str:
    value = record.get("text", record.get("query"))
    value = "" if value is None else str(value).strip()
    return "" if value.lower() == "none" else value


def eval_qa(model, processor, directory: Path, batch: int, k: int) -> float:
    """Official ViDoRe QA protocol: full-page corpus, deduplicated queries."""
    docs, metadata = embed_images(model, processor, parquets(directory / "data"), batch)
    names = [str(m.get("image_filename") or i) for i, m in enumerate(metadata)]

    page_of, column_page = {}, []
    for name in names:
        column_page.append(page_of.setdefault(name, len(page_of)))
    gold_of = {text_of(m): names[i] for i, m in enumerate(metadata) if text_of(m)}

    queries = list(dict.fromkeys(text_of(m) for m in metadata if text_of(m)))
    scores = maxsim(embed_queries(model, processor, queries, batch), docs)

    groups = torch.tensor(column_page, device=scores.device).expand(scores.shape[0], -1)
    merged = torch.full((scores.shape[0], len(page_of)), float("-inf"), device=scores.device)
    merged.scatter_reduce_(1, groups, scores, reduce="amax")
    ranking = merged.argsort(dim=1, descending=True).tolist()

    return sum(
        ndcg(ranking[i], {page_of[gold_of[q]]: 1.0}, k) for i, q in enumerate(queries)
    ) / len(queries)


def eval_beir(model, processor, corpus: Path, queries: Path, qrels: Path, batch: int, k: int,
              cache: dict) -> float:
    if str(corpus) not in cache:
        cache.clear()
        docs, metadata = embed_images(model, processor, parquets(corpus), batch)
        index = {str(m.get("corpus-id") or m.get("id")): i for i, m in enumerate(metadata)}
        cache[str(corpus)] = (docs, index)
    docs, index = cache[str(corpus)]

    gains = defaultdict(dict)
    for record in text_rows(parquets(qrels)):
        doc = index.get(str(record.get("corpus-id") or record.get("id")))
        relevance = float(record.get("score") or 0.0)
        if doc is not None and relevance > 0:
            gains[str(record.get("query-id") or record.get("id"))][doc] = relevance

    texts, ids = [], []
    for record in text_rows(parquets(queries)):
        key = str(record.get("id") or record.get("query-id"))
        if text_of(record) and gains.get(key):
            texts.append(text_of(record))
            ids.append(key)

    scores = maxsim(embed_queries(model, processor, texts, batch), docs)
    ranking = scores.argsort(dim=1, descending=True).tolist()
    return sum(ndcg(ranking[i], gains[key], k) for i, key in enumerate(ids)) / len(ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EVIE-Preview-4.5B")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--max-visual-tokens", type=int, default=768)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    root = Path(args.data_root)
    processor = ColQwen3_5Processor.from_pretrained(
        args.model, max_num_visual_tokens=args.max_visual_tokens
    )
    model = ColQwen3_5.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.enable_bidirectional_attention()
    model = model.to("cuda").eval()

    result, cache = {}, {}
    for task in V1:
        result["V1/" + task] = eval_qa(model, processor, root / "eval" / task, args.batch, 5)
        print("V1  %-46s nDCG@5  %6.2f" % (task, 100 * result["V1/" + task]), flush=True)

    for task in V2:
        base = root / "eval_v2" / task
        result["V2/" + task] = eval_beir(
            model, processor, base / "corpus", base / "queries", base / "qrels", args.batch, 5, cache
        )
        print("V2  %-46s nDCG@5  %6.2f" % (task, 100 * result["V2/" + task]), flush=True)

    for domain in V3:
        base = root / "eval_v3" / domain
        for language in V3_LANGS:
            result["V3/%s/%s" % (domain, language)] = eval_beir(
                model, processor, base / (V3_LANGS[0] + "-corpus"),
                base / (language + "-queries"), base / (language + "-qrels"), args.batch, 10, cache
            )
        scores = [result["V3/%s/%s" % (domain, l)] for l in V3_LANGS]
        print("V3  %-46s nDCG@10 %6.2f" % (domain, 100 * sum(scores) / len(scores)), flush=True)

    def mean(prefix):
        picked = [v for k, v in result.items() if k.startswith(prefix)]
        return sum(picked) / len(picked)

    v1, v2, v3 = mean("V1/"), mean("V2/"), mean("V3/")
    combined = (10 * v1 + 4 * v2) / 14
    print("\nViDoRe V1        nDCG@5   %6.2f  (10 tasks)" % (100 * v1))
    print("ViDoRe V2        nDCG@5   %6.2f  (4 tasks)" % (100 * v2))
    print("ViDoRe V1+V2     nDCG@5   %6.2f  (14 tasks)" % (100 * combined))
    print("ViDoRe V3 public nDCG@10  %6.2f  (8 domains x 6 languages)" % (100 * v3))

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"model": args.model, "max_visual_tokens": args.max_visual_tokens,
             "per_task": result,
             "average": {"v1_ndcg@5": v1, "v2_ndcg@5": v2, "v1v2_ndcg@5": combined,
                         "v3_public_ndcg@10": v3}},
            ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
