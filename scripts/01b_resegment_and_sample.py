from __future__ import annotations

import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import nltk
import tiktoken

for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.tokenize import sent_tokenize

# ------------------------- Configuración -------------------------------------

IN_DIR = Path("../data/processed")
OUT_DIR = Path("../data/processed")
DOCS_PATH = IN_DIR / "documents.jsonl"

TARGET_TOKENS = 400
MAX_TOKENS = 600
MIN_TOKENS = 80
OVERLAP_SENTENCES = 1

SEED = 42
SAMPLE_PER_OBJECTIVE_GOOD = 37
INCLUDE_ALL_BAD = True

encoder = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(encoder.encode(text, disallowed_special=()))


# ------------------------- Chunking ------------------------------------------

def chunk_document(text: str, doc_id: int) -> list[dict]:

    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in text.split(". ") if s.strip()]

    if not sentences:
        return []

    sent_tokens = [n_tokens(s) for s in sentences]

    chunks = []
    i = 0
    n = len(sentences)

    while i < n:
        current_sents = []
        current_tokens = 0
        j = i

        while j < n:
            st = sent_tokens[j]
            if st > MAX_TOKENS and not current_sents:
                words = sentences[j].split()
                max_words = int(MAX_TOKENS * 0.75)
                truncated = " ".join(words[:max_words])
                current_sents.append(truncated)
                current_tokens = n_tokens(truncated)
                j += 1
                break

            if current_tokens + st > TARGET_TOKENS and current_sents:
                break

            current_sents.append(sentences[j])
            current_tokens += st
            j += 1

            if current_tokens >= TARGET_TOKENS:
                break

        chunk_text = " ".join(current_sents).strip()
        if current_tokens >= MIN_TOKENS or not chunks:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_c{len(chunks)}",
                "chunk_index": len(chunks),
                "n_tokens": current_tokens,
                "n_sentences": len(current_sents),
                "text": chunk_text,
            })

        if j >= n:
            break
        i = max(j - OVERLAP_SENTENCES, i + 1)

    return chunks


# ------------------------- Pipeline ------------------------------------------

def main() -> None:
    random.seed(SEED)

    # ---------- carga ----------
    docs = []
    with DOCS_PATH.open(encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"[INFO] Cargados {len(docs)} documentos")

    # ---------- chunking de TODO el corpus ----------
    all_chunks = []
    chunks_per_doc = []
    print("[INFO] Resegmentando...")
    for idx, d in enumerate(docs):
        doc_chunks = chunk_document(d["text"], d["doc_id"])
        # propagamos metadata del doc al chunk
        for c in doc_chunks:
            c["source_file"] = d["source_file"]
            c["label"] = d["label"]
            c["objective"] = d["objective"]
        all_chunks.extend(doc_chunks)
        chunks_per_doc.append(len(doc_chunks))

        if (idx + 1) % 100 == 0:
            print(f"       procesados {idx + 1}/{len(docs)} docs")

    # ---------- escritura de chunks completos ----------
    chunks_all_path = OUT_DIR / "chunks_all.jsonl"
    with chunks_all_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[OK] {len(all_chunks)} chunks -> {chunks_all_path}")

    # ---------- muestreo estratificado ----------
    by_strata: dict[tuple, list[dict]] = defaultdict(list)
    for d in docs:
        by_strata[(d["label"], d["objective"])].append(d)

    sampled_docs: list[dict] = []
    for (label, objective), group in by_strata.items():
        if label == "bad" and INCLUDE_ALL_BAD:
            sampled_docs.extend(group)
            continue
        if label == "good":
            k = min(SAMPLE_PER_OBJECTIVE_GOOD, len(group))
            sampled_docs.extend(random.sample(group, k))

    sampled_doc_ids = {d["doc_id"] for d in sampled_docs}
    sampled_chunks = [c for c in all_chunks if c["doc_id"] in sampled_doc_ids]

    # ---------- escritura de muestra ----------
    sample_chunks_path = OUT_DIR / "chunks_sample.jsonl"
    with sample_chunks_path.open("w", encoding="utf-8") as f:
        for c in sampled_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    sample_docs_path = OUT_DIR / "sample_docs.jsonl"
    with sample_docs_path.open("w", encoding="utf-8") as f:
        for d in sampled_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ---------- estadísticas ----------
    chunk_token_lens = [c["n_tokens"] for c in all_chunks]
    sample_token_lens = [c["n_tokens"] for c in sampled_chunks]

    sample_tokens_total = sum(sample_token_lens)
    est_input_cost = sample_tokens_total * 3 / 1_000_000
    est_output_cost = sample_tokens_total * 0.3 * 15 / 1_000_000
    est_total_cost = est_input_cost + est_output_cost

    stats = {
        "total_chunks": len(all_chunks),
        "chunks_per_doc": {
            "min": min(chunks_per_doc),
            "max": max(chunks_per_doc),
            "mean": round(statistics.mean(chunks_per_doc), 1),
            "median": statistics.median(chunks_per_doc),
        },
        "chunk_tokens_all": {
            "min": min(chunk_token_lens),
            "max": max(chunk_token_lens),
            "mean": round(statistics.mean(chunk_token_lens), 1),
            "median": statistics.median(chunk_token_lens),
        },
        "sample": {
            "n_docs": len(sampled_docs),
            "n_chunks": len(sampled_chunks),
            "by_strata": dict(Counter(
                (d["label"], d["objective"]) for d in sampled_docs
            )),
            "total_tokens": sample_tokens_total,
            "estimated_api_cost_usd": {
                "input": round(est_input_cost, 2),
                "output_approx": round(est_output_cost, 2),
                "total_approx": round(est_total_cost, 2),
            },
        },
    }
    stats["sample"]["by_strata"] = {
        f"{k[0]}/{k[1]}": v for k, v in stats["sample"]["by_strata"].items()
    }

    (OUT_DIR / "chunking_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("RESUMEN FASE 1b")
    print("=" * 60)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n[OK] Chunks muestra -> {sample_chunks_path}")
    print(f"[OK] Docs muestra   -> {sample_docs_path}")
    print(f"[OK] Stats          -> {OUT_DIR / 'chunking_stats.json'}")


if __name__ == "__main__":
    main()