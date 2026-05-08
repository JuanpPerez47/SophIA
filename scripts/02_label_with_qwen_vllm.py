from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

# ------------------------- Configuración -------------------------------------

IN_PATH = Path("../data/processed/chunks_sample.jsonl")
OUT_DIR = Path("../data/labeled")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELED_PATH = OUT_DIR / "labeled_chunks.jsonl"
SUBSAMPLE_PATH = OUT_DIR / "subsample_used.jsonl"

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct-AWQ"
MAX_TOKENS = 1024
MAX_CHUNKS_PER_DOC = 80
SEED = 42

# vLLM tuning
GPU_MEMORY_UTILIZATION = 0.92
MAX_MODEL_LEN = 4096
TEMPERATURE = 0.1
TOP_P = 0.9

# ------------------------- Ontología (idéntica al script anterior) -----------

GOOD_CATEGORIES = {
    "data_acquisition_technology": "Low-cost sensors, field validation, offline loggers, solar power, Arduino, ESP32, LoRa, SMS, replaceable components, manual measurement, simple telemetry, low-power consumption, microcontrollers, open hardware, low-cost IoT nodes, Raspberry Pi, ultrasonic sensors.",
    "data_management": "CSV, JSON, SQLite, local data, offline data, automatic validation, acceptable ranges, version control, differential synchronization, local storage, incomplete data handling, interpolation, in-field data, lightweight databases, open formats.",
    "operation_maintenance": "Minimalist dashboards, offline-first apps, illustrated manuals, maintenance protocols, checklists, sensor redundancy, spare parts management, technical training, preventive maintenance, simple indicators, SMS alerts, local maintenance, available spare parts.",
    "sustainability": "Solar panels, MPPT, IP67/IP68, open-source, reusability, energy optimization, low cost, accessibility, replicability, low budget, locally available materials, appropriate technology, sustainability, economic viability.",
    "community_participation": "Farmer training, rural community, participatory monitoring, community leaders, rural schools, participatory design, results dissemination, water councils, local users, technological appropriation, empowerment, local capacity building, environmental promoters.",
    "local_adaptation": "Climate adaptation, local materials, low literacy contexts, rural areas, no connectivity, remote zones, difficult access, context-adapted, rural context, local language.",
    "scalability": "Standardized kits, replicability, modular architecture, progressive piloting, knowledge transfer, scalable, modules, standardization, replicable methodology, low scaling cost, gradual expansion, open source.",
}

BAD_CATEGORIES = {
    "inappropriate_technology": "Mass spectrometers, chromatography, industrial analyzers, high-precision industrial sensors, laboratory equipment, sophisticated hardware, high-end sensors, microscopes, spectrophotometers.",
    "non_adaptable_infrastructure": "Fiber optics, mandatory 4G/5G, constant connectivity, high bandwidth requirements, dedicated servers, datacenters, low-latency requirements, permanent connection.",
    "cloud_dependency": "Mandatory cloud (AWS/Azure/GCP), cloud subscriptions, internet dependency, permanent connection required, mandatory SaaS, mandatory cloud APIs.",
    "high_costs": "High costs, costly investments, expensive infrastructure, expensive maintenance, costly licenses, expensive proprietary software, expensive equipment.",
    "technical_complexity": "Advanced technical requirements, specialized engineering, expert-required, specialized technicians, advanced training, expert knowledge, complex calibration, specialized maintenance, external technical support, exclusive providers, vendor lock-in, no local support.",
    "centralization": "Centralized systems, central control, single platform, technological monopoly, top-down approach, non-participatory, imposed on community, foreign to community, no local adaptation.",
    "rural_inaccessibility": "Urban-only, city-only, not adaptable to rural areas, requires constant electricity, requires power grid, no offline option, doesn't work without internet, requires urban infrastructure, not field-scalable.",
}

SYSTEM_PROMPT = f"""You are an expert annotator for a research project on appropriate technologies for rural water monitoring. Your task is to analyze text chunks from academic papers and identify whether they describe practices (good or bad) related to rural water monitoring systems.

# CATEGORIES OF GOOD PRACTICES
{json.dumps(GOOD_CATEGORIES, indent=2)}

# CATEGORIES OF BAD PRACTICES
{json.dumps(BAD_CATEGORIES, indent=2)}

# YOUR TASK
For each chunk, determine:
1. Does it describe a concrete practice (good or bad) relevant to rural water monitoring? Many chunks will be metadata, references, generic introductions, or unrelated content - those should be marked as no_practice.
2. If yes, classify as "good" or "bad".
3. Assign 1-3 categories from the lists above.
4. If you find a clear pattern not covered by existing categories, propose a new subcategory (snake_case).
5. Extract the exact span (verbatim text) from the chunk that constitutes the practice.
6. Write a brief explanation (1-2 sentences) of why it's a good/bad practice.
7. Provide a confidence score 0.0-1.0.

# OUTPUT FORMAT
Return ONLY valid JSON, no markdown, no preamble. Schema:
{{
  "contains_practice": true/false,
  "practices": [
    {{
      "type": "good" | "bad",
      "categories": ["category_key", ...],
      "subcategory_proposed": "snake_case_name_or_null",
      "span": "verbatim text from chunk",
      "explanation": "brief justification",
      "confidence": 0.0-1.0
    }}
  ],
  "summary": "one-sentence summary of the chunk content"
}}

If contains_practice is false, "practices" must be an empty array. Always include "summary" regardless.

Be conservative: only mark contains_practice=true when the chunk genuinely describes a practice, methodology, or design choice. References, abstracts restating goals, and acknowledgments do NOT count as practices."""


# ------------------------- Subsampling con cap -------------------------------

def apply_per_doc_cap(chunks: list[dict], cap: int) -> list[dict]:
    random.seed(SEED)
    by_doc: dict[int, list[dict]] = defaultdict(list)
    for c in chunks:
        by_doc[c["doc_id"]].append(c)

    out = []
    capped = 0
    for doc_id, doc_chunks in by_doc.items():
        if len(doc_chunks) <= cap:
            out.extend(doc_chunks)
        else:
            step = len(doc_chunks) / cap
            indices = sorted({int(i * step) for i in range(cap)})
            out.extend(doc_chunks[i] for i in indices[:cap])
            capped += 1
    print(f"[INFO] Cap aplicado: {capped} docs reducidos. "
          f"Total chunks: {len(chunks)} -> {len(out)}")
    return out


# ------------------------- Parsing robusto del output ------------------------

def parse_response_text(text: str) -> dict:
    """Parsea JSON tolerando fences, preámbulos, etc."""
    text = text.strip()

    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        else:
            text = text.lstrip("`").lstrip("json").strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        text = text[first_brace:last_brace + 1]

    return json.loads(text)


# ------------------------- vLLM: carga del modelo ----------------------------

def build_llm():
    from vllm import LLM
    print(f"[INFO] Cargando {MODEL_NAME}...")
    t0 = time.time()
    llm = LLM(
        model=MODEL_NAME,
        dtype="auto",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        trust_remote_code=True,
    )
    print(f"[INFO] Modelo cargado en {time.time() - t0:.1f}s")
    return llm


def build_messages(chunk: dict) -> list[dict]:
    user = (
        f"Analyze this chunk (chunk_id={chunk['chunk_id']}, "
        f"from a paper labeled '{chunk['label']}/{chunk['objective']}'):\n\n"
        f"---\n{chunk['text']}\n---"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def run_inference(llm, chunks: list[dict]) -> list[tuple[dict, str]]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    print(f"[INFO] Construyendo {len(chunks)} prompts...")
    all_messages = [build_messages(c) for c in chunks]

    print(f"[INFO] Lanzando inferencia (vLLM hace batching dinámico)...")
    t0 = time.time()
    outputs = llm.chat(all_messages, sampling)
    elapsed = time.time() - t0
    print(f"[INFO] {len(chunks)} chunks procesados en {elapsed:.1f}s "
          f"({len(chunks) / elapsed:.1f} chunks/s)")

    return [(chunks[i], outputs[i].outputs[0].text) for i in range(len(chunks))]


# ------------------------- Modos ---------------------------------------------

def run_test() -> None:
    chunks = [json.loads(l) for l in IN_PATH.read_text(encoding="utf-8").splitlines()]
    chunks = apply_per_doc_cap(chunks, MAX_CHUNKS_PER_DOC)
    random.seed(SEED)
    sample = random.sample(chunks, 10)

    llm = build_llm()
    results = run_inference(llm, sample)

    print("\n" + "=" * 60)
    print("RESULTADOS DE TEST (10 chunks)")
    print("=" * 60)
    for i, (chunk, raw) in enumerate(results, 1):
        print(f"\n--- [{i}/10] {chunk['chunk_id']} "
              f"(label={chunk['label']}, {chunk['n_tokens']} tok) ---")
        try:
            parsed = parse_response_text(raw)
            print(f"contains_practice: {parsed.get('contains_practice')}")
            for p in parsed.get("practices", []):
                print(f"  -> {p['type']} | {p['categories']} | "
                      f"conf={p.get('confidence')}")
                span = p.get("span", "")
                print(f"     span: {span[:120]}{'...' if len(span) > 120 else ''}")
                if p.get("subcategory_proposed"):
                    print(f"     proposed: {p['subcategory_proposed']}")
            print(f"summary: {parsed.get('summary', '')[:150]}")
        except Exception as e:
            print(f"[PARSE ERROR] {e}")
            print(f"Raw output:\n{raw[:500]}")

    print("\n[TEST OK] Si los resultados se ven bien, ejecuta --full")


def run_full() -> None:
    chunks = [json.loads(l) for l in IN_PATH.read_text(encoding="utf-8").splitlines()]
    chunks = apply_per_doc_cap(chunks, MAX_CHUNKS_PER_DOC)

    with SUBSAMPLE_PATH.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"[OK] Submuestra guardada -> {SUBSAMPLE_PATH}")

    llm = build_llm()
    results = run_inference(llm, chunks)

    n_ok = 0
    n_err = 0
    n_with = 0
    n_without = 0
    proposed = defaultdict(int)

    with LABELED_PATH.open("w", encoding="utf-8") as fout:
        for chunk, raw in results:
            record = {"chunk_id": chunk["chunk_id"], "raw_output": raw}
            try:
                parsed = parse_response_text(raw)
                record["status"] = "ok"
                record["labels"] = parsed
                n_ok += 1
                if parsed.get("contains_practice"):
                    n_with += 1
                    for p in parsed.get("practices", []):
                        sub = p.get("subcategory_proposed")
                        if sub and sub not in (None, "null", ""):
                            proposed[sub] += 1
                else:
                    n_without += 1
            except Exception as e:
                record["status"] = "parse_error"
                record["error"] = str(e)
                n_err += 1
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("RESUMEN FASE 2 (LOCAL)")
    print("=" * 60)
    print(f"Total procesados: {n_ok + n_err}")
    print(f"  OK:           {n_ok}")
    print(f"  Parse errors: {n_err}")
    print(f"  Con práctica: {n_with}")
    print(f"  Sin práctica: {n_without}")
    print(f"\nSubcategorías propuestas (top 20):")
    top = sorted(proposed.items(), key=lambda x: -x[1])[:20]
    for name, count in top:
        print(f"  {count:4d}  {name}")

    report = {
        "model": MODEL_NAME,
        "n_ok": n_ok,
        "n_err": n_err,
        "n_with_practice": n_with,
        "n_no_practice": n_without,
        "proposed_subcategories": dict(proposed),
    }
    (OUT_DIR / "labeling_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[OK] Resultados -> {LABELED_PATH}")
    print(f"[OK] Reporte    -> {OUT_DIR / 'labeling_report.json'}")


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", action="store_true", help="10 chunks de prueba")
    g.add_argument("--full", action="store_true", help="Corrida completa")
    args = p.parse_args()

    if args.test:
        run_test()
    elif args.full:
        run_full()


if __name__ == "__main__":
    main()