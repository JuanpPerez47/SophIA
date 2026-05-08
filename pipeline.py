from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

import nltk

for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.tokenize import sent_tokenize

# ------------------------- Config --------------------------------------------

TARGET_TOKENS = 400
MAX_TOKENS = 600
MIN_TOKENS = 80
OVERLAP_SENTENCES = 1

GOOD_CATEGORIES = [
    "data_acquisition_technology",
    "data_management",
    "operation_maintenance",
    "sustainability",
    "community_participation",
    "local_adaptation",
    "scalability",
]

BAD_CATEGORIES = [
    "inappropriate_technology",
    "non_adaptable_infrastructure",
    "cloud_dependency",
    "high_costs",
    "technical_complexity",
    "centralization",
    "rural_inaccessibility",
]

ALL_CATEGORIES = GOOD_CATEGORIES + BAD_CATEGORIES


# ------------------------- Estructuras de datos ------------------------------

@dataclass
class PracticeFinding:
    """Una práctica detectada en un chunk."""
    chunk_index: int
    type: str
    categories: list[str]
    span: str
    explanation: str
    confidence: float


@dataclass
class ChunkResult:
    chunk_index: int
    text: str
    contains_practice: bool
    summary: str
    practices: list[PracticeFinding] = field(default_factory=list)
    parse_error: str | None = None


@dataclass
class DocumentResult:
    filename: str
    n_chunks: int
    n_chunks_with_practice: int
    n_good_practices: int
    n_bad_practices: int
    overall_classification: str
    good_score: float
    bad_score: float
    category_counts: dict
    chunk_results: list[ChunkResult]
    top_findings: list[PracticeFinding]
    document_summary: str

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ------------------------- Extracción de texto -------------------------------

def extract_text_from_file(path: str | Path, file_bytes: bytes | None = None) -> str:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path, file_bytes)
    elif suffix in (".txt", ".md"):
        if file_bytes is not None:
            return file_bytes.decode("utf-8", errors="replace")
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Formato no soportado: {suffix}. Usa .pdf, .txt o .md")


def _extract_pdf(path: Path, file_bytes: bytes | None) -> str:
    try:
        import pymupdf  # PyMuPDF
    except ImportError:
        raise ImportError("Falta pymupdf. Instala con: pip install pymupdf")

    if file_bytes is not None:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    else:
        doc = pymupdf.open(path)

    texts = []
    for page in doc:
        texts.append(page.get_text())
    doc.close()
    return "\n\n".join(texts)


# ------------------------- Chunking ------------------------------------------

def chunk_text(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    try:
        sentences = sent_tokenize(text)
    except Exception:
        sentences = [s.strip() for s in text.split(". ") if s.strip()]

    if not sentences:
        return []

    sent_lens = [len(s) // 4 for s in sentences]

    chunks = []
    i = 0
    n = len(sentences)
    while i < n:
        current = []
        current_tokens = 0
        j = i
        while j < n:
            st = sent_lens[j]
            if st > MAX_TOKENS and not current:
                truncated_chars = MAX_TOKENS * 4
                current.append(sentences[j][:truncated_chars])
                current_tokens = MAX_TOKENS
                j += 1
                break
            if current_tokens + st > TARGET_TOKENS and current:
                break
            current.append(sentences[j])
            current_tokens += st
            j += 1
            if current_tokens >= TARGET_TOKENS:
                break

        chunk = " ".join(current).strip()
        if current_tokens >= MIN_TOKENS or not chunks:
            chunks.append(chunk)

        if j >= n:
            break
        i = max(j - OVERLAP_SENTENCES, i + 1)

    return chunks


# ------------------------- Agregación a nivel documento ----------------------

def aggregate_results(filename: str, chunk_results: list[ChunkResult]) -> DocumentResult:
    n_chunks = len(chunk_results)
    n_with = sum(1 for c in chunk_results if c.contains_practice)
    n_good = sum(len([p for p in c.practices if p.type == "good"]) for c in chunk_results)
    n_bad = sum(len([p for p in c.practices if p.type == "bad"]) for c in chunk_results)

    cat_counter: Counter = Counter()
    for c in chunk_results:
        for p in c.practices:
            for cat in p.categories:
                cat_counter[cat] += 1

    good_score = round((n_good / max(n_chunks, 1)) * 100, 1)
    bad_score = round((n_bad / max(n_chunks, 1)) * 100, 1)

    if n_good == 0 and n_bad == 0:
        overall = "neutral"
    elif n_good > n_bad * 2:
        overall = "good"
    elif n_bad > n_good * 1.5:
        overall = "bad"
    elif n_good > n_bad:
        overall = "good"
    elif n_bad > n_good:
        overall = "bad"
    else:
        overall = "neutral"

    all_findings = []
    for c in chunk_results:
        all_findings.extend(c.practices)
    top_findings = sorted(all_findings, key=lambda p: -p.confidence)[:5]

    practice_summaries = [c.summary for c in chunk_results if c.contains_practice and c.summary]
    if practice_summaries:
        document_summary = " | ".join(practice_summaries[:3])
        if len(document_summary) > 500:
            document_summary = document_summary[:497] + "..."
    else:
        document_summary = "No specific rural water monitoring practices detected in this document."

    return DocumentResult(
        filename=filename,
        n_chunks=n_chunks,
        n_chunks_with_practice=n_with,
        n_good_practices=n_good,
        n_bad_practices=n_bad,
        overall_classification=overall,
        good_score=good_score,
        bad_score=bad_score,
        category_counts=dict(cat_counter),
        chunk_results=chunk_results,
        top_findings=top_findings,
        document_summary=document_summary,
    )


# ------------------------- Analizador principal ------------------------------

class DocumentAnalyzer:

    def __init__(self, client):
        self.client = client

    def analyze_file(
        self,
        filename: str,
        file_bytes: bytes | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> DocumentResult:

        if progress_callback:
            progress_callback(0, 1, "Extrayendo texto...")

        text = extract_text_from_file(filename, file_bytes)
        if not text.strip():
            return aggregate_results(filename, [])

        if progress_callback:
            progress_callback(0, 1, "Segmentando en chunks...")

        chunks = chunk_text(text)
        n = len(chunks)

        if progress_callback:
            progress_callback(0, n, f"{n} chunks identificados. Analizando...")

        chunk_results: list[ChunkResult] = []
        for i, chunk in enumerate(chunks):
            response = self.client.analyze_text(chunk)
            practices = [
                PracticeFinding(
                    chunk_index=i,
                    type=p.get("type", "good"),
                    categories=p.get("categories", []),
                    span=p.get("span", ""),
                    explanation=p.get("explanation", ""),
                    confidence=float(p.get("confidence", 0.5)),
                )
                for p in response.get("practices", [])
            ]
            chunk_results.append(ChunkResult(
                chunk_index=i,
                text=chunk,
                contains_practice=response.get("contains_practice", False),
                summary=response.get("summary", ""),
                practices=practices,
                parse_error=response.get("_parse_error"),
            ))

            if progress_callback:
                progress_callback(i + 1, n, f"Chunk {i+1}/{n} analizado")

        return aggregate_results(filename, chunk_results)


# ------------------------- CLI -----------------------------------------------

def _cli():
    import argparse
    import json
    import sys

    from local.ollama_client import WaterPracticesClient

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Ruta al PDF o TXT")
    parser.add_argument("--output", type=str, default=None, help="Guardar resultado JSON")
    args = parser.parse_args()

    client = WaterPracticesClient()
    analyzer = DocumentAnalyzer(client)

    def cb(done, total, msg):
        print(f"[{done}/{total}] {msg}", file=sys.stderr)

    result = analyzer.analyze_file(args.file, progress_callback=cb)

    print("\n" + "=" * 60)
    print(f"DOCUMENTO: {result.filename}")
    print("=" * 60)
    print(f"Chunks totales:           {result.n_chunks}")
    print(f"Chunks con práctica:      {result.n_chunks_with_practice}")
    print(f"Prácticas BUENAS:         {result.n_good_practices}")
    print(f"Prácticas MALAS:          {result.n_bad_practices}")
    print(f"Clasificación global:     {result.overall_classification.upper()}")
    print(f"Score buenas:             {result.good_score}%")
    print(f"Score malas:              {result.bad_score}%")
    print(f"\nTop categorías:")
    for cat, count in sorted(result.category_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {count:3d}  {cat}")
    print(f"\nResumen: {result.document_summary[:200]}")

    if args.output:
        Path(args.output).write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"\n[OK] Guardado en {args.output}")


if __name__ == "__main__":
    _cli()