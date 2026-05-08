from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks_sample.jsonl"
LABELED_PATH = PROJECT_ROOT / "data" / "labeled" / "labeled_chunks.jsonl"
OUT_DIR = PROJECT_ROOT / "data" / "sft"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EVAL_FRACTION = 0.1
NEGATIVE_RATIO = 2.0

# ---------- Normalización de categorías "inventadas" ----------

OFFICIAL_GOOD = {
    "data_acquisition_technology", "data_management", "operation_maintenance",
    "sustainability", "community_participation", "local_adaptation", "scalability",
}
OFFICIAL_BAD = {
    "inappropriate_technology", "non_adaptable_infrastructure", "cloud_dependency",
    "high_costs", "technical_complexity", "centralization", "rural_inaccessibility",
}
OFFICIAL = OFFICIAL_GOOD | OFFICIAL_BAD

CATEGORY_REMAP = {
    "low_cost": "sustainability",
    "low_costs": "sustainability",
    "economic_viabilty": "sustainability",
    "economic_viability": "sustainability",
    "open_source": "sustainability",
    "energy optimization": "sustainability",
    "energy_optimization": "sustainability",
    "environmental_risks": "sustainability",
    "open_formats": "data_management",
    "incomplete_data_handling": "data_management",
    "information_overload": "data_management",
    "governance": "community_participation",
    "policy": "community_participation",
    "policy_change": "community_participation",
    "funding": "community_participation",
    "funding_sources": "community_participation",
    "funding_strategies": "community_participation",
    "capacity_building": "community_participation",
    "farmer_motivation": "community_participation",
    "modular_architecture": "scalability",
    "knowledge_transfer": "scalability",
    "limited_manual_measurements": "operation_maintenance",
    "continual_signaling_interactions": "operation_maintenance",
    "pollution_risk": "inappropriate_technology",
}


def normalize_categories(cats: list[str]) -> list[str]:
    out = []
    for c in cats:
        if c in OFFICIAL:
            out.append(c)
        elif c in CATEGORY_REMAP:
            out.append(CATEGORY_REMAP[c])
    return list(dict.fromkeys(out))


# ---------- Construcción del prompt ----------

SYSTEM_PROMPT_INFERENCE = """You are an expert annotator for rural water monitoring research. Analyze text chunks from academic papers and identify practices (good or bad) related to rural water monitoring.

Categories of GOOD practices: data_acquisition_technology, data_management, operation_maintenance, sustainability, community_participation, local_adaptation, scalability.

Categories of BAD practices: inappropriate_technology, non_adaptable_infrastructure, cloud_dependency, high_costs, technical_complexity, centralization, rural_inaccessibility.

Return ONLY valid JSON with this schema:
{
  "contains_practice": true|false,
  "practices": [
    {
      "type": "good"|"bad",
      "categories": ["category_key", ...],
      "span": "verbatim text from chunk",
      "explanation": "brief justification",
      "confidence": 0.0-1.0
    }
  ],
  "summary": "one-sentence summary"
}

If contains_practice is false, "practices" must be []."""


def build_user_message(chunk_text: str) -> str:
    return f"Analyze this chunk:\n\n---\n{chunk_text}\n---"


def build_assistant_message(labels: dict) -> str:
    contains = bool(labels.get("contains_practice", False))
    practices_clean = []
    if contains:
        for p in labels.get("practices", []):
            cats = normalize_categories(p.get("categories", []))
            if not cats:
                continue
            practices_clean.append({
                "type": p.get("type", "good"),
                "categories": cats,
                "span": p.get("span", "")[:500],   # cap longitud del span
                "explanation": p.get("explanation", "")[:300],
                "confidence": round(float(p.get("confidence", 0.7)), 2),
            })
        if not practices_clean:
            contains = False

    summary = (labels.get("summary") or "").strip()[:300]
    response = {
        "contains_practice": contains,
        "practices": practices_clean,
        "summary": summary,
    }
    return json.dumps(response, ensure_ascii=False)


# ---------- Pipeline principal ----------

def main() -> None:
    random.seed(SEED)

    chunk_text_by_id: dict[str, str] = {}
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            chunk_text_by_id[r["chunk_id"]] = r["text"]
    print(f"[INFO] Cargados {len(chunk_text_by_id)} textos de chunks")

    positives: list[dict] = []
    negatives: list[dict] = []
    parse_errors = 0
    missing_text = 0

    with LABELED_PATH.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                parse_errors += 1
                continue
            chunk_id = rec["chunk_id"]
            text = chunk_text_by_id.get(chunk_id)
            if text is None:
                missing_text += 1
                continue

            labels = rec.get("labels", {})
            assistant_msg = build_assistant_message(labels)
            example = {
                "chunk_id": chunk_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_INFERENCE},
                    {"role": "user", "content": build_user_message(text)},
                    {"role": "assistant", "content": assistant_msg},
                ],
            }
            assistant_parsed = json.loads(assistant_msg)
            if assistant_parsed["contains_practice"]:
                positives.append(example)
            else:
                negatives.append(example)

    print(f"[INFO] Parse errors descartados: {parse_errors}")
    print(f"[INFO] Sin texto disponible:     {missing_text}")
    print(f"[INFO] Positivos:                {len(positives)}")
    print(f"[INFO] Negativos:                {len(negatives)}")

    target_neg = int(len(positives) * NEGATIVE_RATIO)
    if len(negatives) > target_neg:
        negatives_sampled = random.sample(negatives, target_neg)
        print(f"[INFO] Submuestreando negativos: {len(negatives)} -> {target_neg}")
    else:
        negatives_sampled = negatives
        print(f"[INFO] No se submuestrea (negativos <= target)")

    all_examples = positives + negatives_sampled
    random.shuffle(all_examples)

    n_eval = int(len(all_examples) * EVAL_FRACTION)
    eval_set = all_examples[:n_eval]
    train_set = all_examples[n_eval:]

    train_path = OUT_DIR / "train.jsonl"
    eval_path = OUT_DIR / "eval.jsonl"

    with train_path.open("w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with eval_path.open("w", encoding="utf-8") as f:
        for ex in eval_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def stats(examples: list[dict]) -> dict:
        n_pos = sum(
            1 for e in examples
            if json.loads(e["messages"][-1]["content"])["contains_practice"]
        )
        return {"total": len(examples), "with_practice": n_pos, "no_practice": len(examples) - n_pos}

    train_stats = stats(train_set)
    eval_stats = stats(eval_set)

    report = {
        "train": train_stats,
        "eval": eval_stats,
        "total": len(all_examples),
        "negative_ratio_used": NEGATIVE_RATIO,
        "seed": SEED,
    }
    (OUT_DIR / "dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("DATASET SFT CONSTRUIDO")
    print("=" * 60)
    print(f"Train:  {train_stats['total']:5d} | "
          f"con práctica: {train_stats['with_practice']:4d} | "
          f"sin práctica: {train_stats['no_practice']:4d}")
    print(f"Eval:   {eval_stats['total']:5d} | "
          f"con práctica: {eval_stats['with_practice']:4d} | "
          f"sin práctica: {eval_stats['no_practice']:4d}")
    print(f"\n[OK] Train -> {train_path}")
    print(f"[OK] Eval  -> {eval_path}")
    print(f"[OK] Stats -> {OUT_DIR / 'dataset_report.json'}")


if __name__ == "__main__":
    main()