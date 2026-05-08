from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_PATH = PROJECT_ROOT / "models" / "qwen7b-water-lora"
MERGED_PATH = PROJECT_ROOT / "models" / "qwen7b-water-merged"
LOGS_DIR = PROJECT_ROOT / "logs"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO = os.environ.get("HF_REPO")


def step1_merge() -> None:
    if MERGED_PATH.exists() and (MERGED_PATH / "config.json").exists():
        print(f"[INFO] Modelo merged ya existe en {MERGED_PATH}, saltando merge.")
        return

    MERGED_PATH.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Cargando modelo base ({BASE_MODEL}) en BF16...")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"[2/4] Cargando adapter LoRA desde {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))

    print("[3/4] Mergeando adapter con base (puede tardar 1-2 min)...")
    model = model.merge_and_unload()

    print(f"[4/4] Guardando modelo merged en {MERGED_PATH}...")
    model.save_pretrained(
        str(MERGED_PATH),
        safe_serialization=True,
        max_shard_size="5GB",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.save_pretrained(str(MERGED_PATH))

    del model, base
    torch.cuda.empty_cache()

    print(f"[OK] Merge completo. Tamaño en disco:")
    os.system(f"du -sh {MERGED_PATH}")


def step2_create_model_card() -> None:
    eval_report = json.loads((LOGS_DIR / "eval_report.json").read_text())
    train_log = json.loads((LOGS_DIR / "training_log.json").read_text())

    cp = eval_report["contains_practice"]
    when_both = eval_report["when_both_detect_practice"]

    card = f"""---
language:
- en
- es
license: apache-2.0
base_model: {BASE_MODEL}
tags:
- text-classification
- structured-output
- water-monitoring
- rural-development
- lora
- fine-tuned
library_name: transformers
pipeline_tag: text-generation
---

# Qwen 2.5 7B — Rural Water Monitoring Practices Classifier

Fine-tuned version of [Qwen/Qwen2.5-7B-Instruct]({f"https://huggingface.co/{BASE_MODEL}"}) for classifying and extracting good and bad practices in rural water monitoring research papers.

## Model Description

This model analyzes text chunks from academic papers and identifies whether they describe **good** or **bad** practices for rural water monitoring systems, classifying them across 14 predefined categories (7 good + 7 bad).

The model was fine-tuned with LoRA (r=32) on a dataset of 3516 chunks labeled by Qwen 2.5 72B Instruct AWQ.

## Categories

**Good practices**: `data_acquisition_technology`, `data_management`, `operation_maintenance`, `sustainability`, `community_participation`, `local_adaptation`, `scalability`

**Bad practices**: `inappropriate_technology`, `non_adaptable_infrastructure`, `cloud_dependency`, `high_costs`, `technical_complexity`, `centralization`, `rural_inaccessibility`

## Output Format

The model returns structured JSON:

```json
{{
  "contains_practice": true,
  "practices": [
    {{
      "type": "good",
      "categories": ["data_acquisition_technology", "sustainability"],
      "span": "verbatim text from input",
      "explanation": "brief justification",
      "confidence": 0.92
    }}
  ],
  "summary": "one-sentence summary"
}}
```

## Evaluation Results

Evaluated on a held-out set of 390 chunks:

| Metric | Value |
|---|---|
| JSON parse success | {eval_report['json_parse_success_rate']['predicted']:.1%} |
| Accuracy (contains_practice) | {cp['accuracy']:.1%} |
| Precision | {cp['precision']:.1%} |
| Recall | {cp['recall']:.1%} |
| F1 | {cp['f1']:.3f} |
| Type agreement (good/bad) | {when_both['type_agreement_rate']:.1%} |
| Mean categories Jaccard | {when_both['mean_categories_jaccard']:.3f} |

## Training Details

| Parameter | Value |
|---|---|
| Base model | {BASE_MODEL} |
| Method | LoRA (r=32, alpha=64) |
| Trainable params | 80.7M (1.05% of total) |
| Train samples | {train_log['train_samples']} |
| Epochs | 3 |
| Batch size (effective) | 16 |
| Learning rate | 2e-4 (cosine schedule) |
| Final train loss | {train_log['train_loss']:.4f} |
| Final eval loss | {train_log['eval_loss']:.4f} |
| Hardware | NVIDIA A100 SXM4 80GB |
| Training time | ~1 hour |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{HF_REPO}",
    torch_dtype="bfloat16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{HF_REPO}")

system_prompt = '''You are an expert annotator for rural water monitoring research. Analyze text chunks from academic papers and identify practices (good or bad) related to rural water monitoring.

Categories of GOOD practices: data_acquisition_technology, data_management, operation_maintenance, sustainability, community_participation, local_adaptation, scalability.

Categories of BAD practices: inappropriate_technology, non_adaptable_infrastructure, cloud_dependency, high_costs, technical_complexity, centralization, rural_inaccessibility.

Return ONLY valid JSON with this schema:
{{"contains_practice": true|false, "practices": [...], "summary": "..."}}'''

text = "Your chunk text here..."
messages = [
    {{"role": "system", "content": system_prompt}},
    {{"role": "user", "content": f"Analyze this chunk:\\n\\n---\\n{{text}}\\n---"}},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
outputs = model.generate(inputs, max_new_tokens=1024, do_sample=False)
response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(response)
```

## License & Citation

Apache 2.0. Inherits restrictions and rights from the base Qwen 2.5 model.

This model was developed for academic research on appropriate technologies for rural water monitoring.
"""
    (MERGED_PATH / "README.md").write_text(card, encoding="utf-8")
    print("[OK] README.md (model card) escrito")


def step3_push() -> None:
    if not HF_TOKEN:
        print("[ERROR] Falta variable de entorno HF_TOKEN")
        sys.exit(1)
    if not HF_REPO:
        print("[ERROR] Falta variable de entorno HF_REPO")
        sys.exit(1)

    print(f"\n[PUSH] Autenticando en HuggingFace Hub...")
    login(token=HF_TOKEN)

    api = HfApi()
    print(f"[PUSH] Subiendo {MERGED_PATH} a {HF_REPO}...")
    print(f"       (esto puede tardar 5-30 min según ancho de banda)")

    api.upload_folder(
        folder_path=str(MERGED_PATH),
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="Initial upload: Qwen 7B fine-tuned for rural water practices",
    )

    print(f"\n[OK] Modelo subido exitosamente!")
    print(f"     URL: https://huggingface.co/{HF_REPO}")


def main():
    step1_merge()
    step2_create_model_card()
    step3_push()


if __name__ == "__main__":
    main()