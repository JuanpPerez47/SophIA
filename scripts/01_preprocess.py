from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

# ------------------------- Configuración -------------------------------------

RAW_DIR = Path("../data/raw")
OUT_DIR = Path("../data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DELIMITER = "<|endoftext|>"

FILE_LABELS = {
    "corpus_buenas_practicas_350docs_objGen.txt": ("good", "objGen"),
    "corpus_buenas_practicas_285docs_Obj1.txt":   ("good", "obj1"),
    "corpus_buenas_practicas_184docs_Obj2.txt":   ("good", "obj2"),
    "corpus_buenas_practicas_267docs_Obj3.txt":   ("good", "obj3"),
    "corpus_malas_practicas_3doc_objGen.txt":     ("bad",  "objGen"),
}

MIN_DOC_CHARS = 100
MIN_PARAGRAPH_CHARS = 40

# ------------------------- Utilidades ----------------------------------------

def clean_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = "\n".join(line.strip() for line in t.split("\n"))
    return t.strip()


def split_paragraphs(doc_text: str) -> list[str]:
    chunks = re.split(r"\n\s*\n", doc_text)
    out = []
    for c in chunks:
        c = c.strip()
        if len(c) >= MIN_PARAGRAPH_CHARS:
            out.append(c)
    return out


def detect_lang_simple(text: str) -> str:
    sample = text.lower()[:2000]
    es_markers = sum(sample.count(w) for w in [
        " el ", " la ", " los ", " las ", " de ", " que ", " en ", " un ",
        " una ", " es ", " por ", " para ", " con ", " no ", " se ", " con ",
    ])
    en_markers = sum(sample.count(w) for w in [
        " the ", " of ", " and ", " to ", " in ", " is ", " that ", " for ",
        " with ", " on ", " as ", " by ", " this ", " are ", " be ",
    ])
    if es_markers > en_markers * 1.2:
        return "es"
    if en_markers > es_markers * 1.2:
        return "en"
    return "mixed"


# ------------------------- Pipeline principal --------------------------------

def main() -> None:
    documents: list[dict] = []
    paragraphs: list[dict] = []
    doc_id = 0

    for fname, (label, objective) in FILE_LABELS.items():
        fpath = RAW_DIR / fname
        if not fpath.exists():
            print(f"[WARN] No encontrado: {fpath}")
            continue

        raw = fpath.read_text(encoding="utf-8", errors="replace")
        chunks = [c for c in raw.split(DELIMITER) if c.strip()]
        print(f"[INFO] {fname}: {len(chunks)} documentos crudos")

        kept = 0
        for chunk in chunks:
            text = clean_text(chunk)
            if len(text) < MIN_DOC_CHARS:
                continue

            lang = detect_lang_simple(text)
            paras = split_paragraphs(text)
            if not paras:
                continue

            doc_record = {
                "doc_id": doc_id,
                "source_file": fname,
                "label": label,            # "good" | "bad"
                "objective": objective,    # "objGen" | "obj1" | "obj2" | "obj3"
                "lang": lang,
                "n_chars": len(text),
                "n_paragraphs": len(paras),
                "text": text,
            }
            documents.append(doc_record)

            for p_idx, p in enumerate(paras):
                paragraphs.append({
                    "doc_id": doc_id,
                    "paragraph_id": f"{doc_id}_{p_idx}",
                    "source_file": fname,
                    "label": label,
                    "objective": objective,
                    "lang": detect_lang_simple(p),
                    "n_chars": len(p),
                    "text": p,
                })
            doc_id += 1
            kept += 1
        print(f"       -> {kept} documentos conservados")

    # ---------- escritura ----------
    docs_path = OUT_DIR / "documents.jsonl"
    paras_path = OUT_DIR / "paragraphs.jsonl"

    with docs_path.open("w", encoding="utf-8") as f:
        for d in documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with paras_path.open("w", encoding="utf-8") as f:
        for p in paragraphs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ---------- estadísticas ----------
    char_lens = [d["n_chars"] for d in documents]
    para_lens = [p["n_chars"] for p in paragraphs]
    by_label = Counter(d["label"] for d in documents)
    by_objective = Counter(d["objective"] for d in documents)
    by_lang_doc = Counter(d["lang"] for d in documents)
    by_lang_para = Counter(p["lang"] for p in paragraphs)

    stats = {
        "n_documents": len(documents),
        "n_paragraphs": len(paragraphs),
        "documents_by_label": dict(by_label),
        "documents_by_objective": dict(by_objective),
        "language_distribution_docs": dict(by_lang_doc),
        "language_distribution_paragraphs": dict(by_lang_para),
        "doc_char_stats": {
            "min": min(char_lens) if char_lens else 0,
            "max": max(char_lens) if char_lens else 0,
            "mean": round(statistics.mean(char_lens), 1) if char_lens else 0,
            "median": statistics.median(char_lens) if char_lens else 0,
        },
        "paragraph_char_stats": {
            "min": min(para_lens) if para_lens else 0,
            "max": max(para_lens) if para_lens else 0,
            "mean": round(statistics.mean(para_lens), 1) if para_lens else 0,
            "median": statistics.median(para_lens) if para_lens else 0,
        },
        "approx_total_tokens": sum(char_lens) // 4,
    }

    (OUT_DIR / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- resumen en consola ----------
    print("\n" + "=" * 60)
    print("RESUMEN FASE 1")
    print("=" * 60)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\n[OK] Documentos -> {docs_path}")
    print(f"[OK] Párrafos   -> {paras_path}")
    print(f"[OK] Stats      -> {OUT_DIR / 'stats.json'}")


if __name__ == "__main__":
    main()