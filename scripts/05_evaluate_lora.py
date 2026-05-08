from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = PROJECT_ROOT / "data" / "sft" / "eval.jsonl"
ADAPTER_PATH = PROJECT_ROOT / "models" / "qwen7b-water-lora"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 1024
TEMPERATURE = 0.0

# ------------------------- Parsing robusto -----------------------------------

def parse_json_response(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        else:
            text = text.lstrip("`").lstrip("json").strip()
    fb = text.find("{")
    lb = text.rfind("}")
    if fb >= 0 and lb > fb:
        text = text[fb:lb + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ------------------------- Métricas ------------------------------------------

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def categories_from_practices(practices: list) -> set[str]:
    cats = set()
    for p in practices or []:
        for c in p.get("categories", []):
            cats.add(c)
    return cats


def types_from_practices(practices: list) -> set[str]:
    return {p.get("type") for p in (practices or []) if p.get("type")}


# ------------------------- Inferencia con vLLM -------------------------------

def run_inference():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"[INFO] Cargando {BASE_MODEL} con LoRA adapter...")
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        dtype="bfloat16",
    )

    lora_req = LoRARequest("water-adapter", 1, str(ADAPTER_PATH))

    eval_examples = [
        json.loads(l) for l in EVAL_PATH.read_text(encoding="utf-8").splitlines()
    ]
    print(f"[INFO] Eval set: {len(eval_examples)} ejemplos")

    prompts = []
    expected_outputs = []
    for ex in eval_examples:
        msgs = ex["messages"]
        system_user = [m for m in msgs if m["role"] in ("system", "user")]
        assistant = next(m for m in msgs if m["role"] == "assistant")
        prompts.append(system_user)
        expected_outputs.append(assistant["content"])

    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        max_tokens=MAX_TOKENS,
    )

    print(f"[INFO] Lanzando inferencia (greedy, batched)...")
    outputs = llm.chat(prompts, sampling, lora_request=lora_req)

    predictions = [out.outputs[0].text for out in outputs]
    return predictions, expected_outputs, eval_examples


# ------------------------- Evaluación principal ------------------------------

def evaluate(predictions: list[str], expected_outputs: list[str], eval_examples: list[dict]):
    n = len(predictions)
    parse_ok_pred = 0
    parse_ok_exp = 0

    tp = fp = tn = fn = 0

    type_agreements = 0
    type_total = 0
    cat_jaccards = []

    qualitative_examples = []

    for i, (pred_text, exp_text, ex) in enumerate(zip(predictions, expected_outputs, eval_examples)):
        pred = parse_json_response(pred_text)
        exp = parse_json_response(exp_text)

        if exp is not None:
            parse_ok_exp += 1
        if pred is not None:
            parse_ok_pred += 1

        if exp is None:
            continue

        exp_has = bool(exp.get("contains_practice", False))

        if pred is None:
            pred_has = False
        else:
            pred_has = bool(pred.get("contains_practice", False))

        if pred_has and exp_has:
            tp += 1
        elif pred_has and not exp_has:
            fp += 1
        elif not pred_has and not exp_has:
            tn += 1
        else:
            fn += 1

        if pred_has and exp_has and pred is not None:
            exp_types = types_from_practices(exp.get("practices", []))
            pred_types = types_from_practices(pred.get("practices", []))
            if exp_types == pred_types:
                type_agreements += 1
            type_total += 1

            exp_cats = categories_from_practices(exp.get("practices", []))
            pred_cats = categories_from_practices(pred.get("practices", []))
            cat_jaccards.append(jaccard(pred_cats, exp_cats))

        if len(qualitative_examples) < 5:
            keep = (
                (i < 2)
                or (exp_has and len(qualitative_examples) < 4)
                or (not exp_has and len(qualitative_examples) >= 4)
            )
            if keep:
                qualitative_examples.append({
                    "chunk_id": ex.get("chunk_id"),
                    "user_text": ex["messages"][1]["content"][:300] + "...",
                    "expected": exp_text,
                    "predicted": pred_text,
                })

    accuracy = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    type_agreement_rate = type_agreements / max(type_total, 1)
    mean_cat_jaccard = sum(cat_jaccards) / max(len(cat_jaccards), 1)

    report = {
        "n_examples": n,
        "json_parse_success_rate": {
            "expected": parse_ok_exp / max(n, 1),
            "predicted": parse_ok_pred / max(n, 1),
        },
        "contains_practice": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "confusion_matrix": {
                "true_positive": tp,
                "false_positive": fp,
                "true_negative": tn,
                "false_negative": fn,
            },
        },
        "when_both_detect_practice": {
            "n_cases": type_total,
            "type_agreement_rate": round(type_agreement_rate, 4),
            "mean_categories_jaccard": round(mean_cat_jaccard, 4),
        },
    }
    return report, qualitative_examples


# ------------------------- Main ----------------------------------------------

def main():
    predictions, expected, eval_examples = run_inference()
    report, qualitative = evaluate(predictions, expected, eval_examples)

    (LOGS_DIR / "eval_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    examples_text = []
    for i, ex in enumerate(qualitative, 1):
        examples_text.append(
            f"\n{'=' * 70}\n"
            f"EJEMPLO {i} (chunk_id={ex['chunk_id']})\n"
            f"{'=' * 70}\n"
            f"USER INPUT (300 chars):\n{ex['user_text']}\n\n"
            f"EXPECTED:\n{ex['expected']}\n\n"
            f"PREDICTED:\n{ex['predicted']}\n"
        )
    (LOGS_DIR / "eval_examples.txt").write_text(
        "".join(examples_text), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("REPORTE DE EVALUACIÓN")
    print("=" * 60)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[OK] Reporte completo -> {LOGS_DIR / 'eval_report.json'}")
    print(f"[OK] Ejemplos          -> {LOGS_DIR / 'eval_examples.txt'}")


if __name__ == "__main__":
    main()