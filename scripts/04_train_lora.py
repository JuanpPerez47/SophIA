from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

# ------------------------- Config --------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "sft" / "train.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "sft" / "eval.jsonl"
MODEL_OUT_DIR = PROJECT_ROOT / "models" / "qwen7b-water-lora"
LOGS_DIR = PROJECT_ROOT / "logs"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 2048
SEED = 42

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

NUM_EPOCHS = 3
PER_DEVICE_BATCH = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 2

# ------------------------- Pipeline ------------------------------------------

def main() -> None:
    print("=" * 60)
    print("FASE 4: FINE-TUNING QWEN 2.5 7B CON LoRA")
    print("=" * 60)

    print("\n[1/5] Cargando dataset SFT...")
    raw_dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_PATH),
            "eval": str(EVAL_PATH),
        },
    )
    print(f"      Train: {len(raw_dataset['train'])} ejemplos")
    print(f"      Eval:  {len(raw_dataset['eval'])} ejemplos")

    print(f"\n[2/5] Cargando tokenizer de {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"\n[3/5] Cargando modelo base ({BASE_MODEL})...")
    print("      Esto descarga ~15GB la primera vez (5-10 min)")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
        if torch.cuda.is_available() else "eager",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    print("\n[4/5] Aplicando LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\n[5/5] Lanzando entrenamiento...")

    sft_config = SFTConfig(
        output_dir=str(MODEL_OUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        per_device_eval_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        bf16=True,
        tf32=True,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        dataset_text_field=None,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset["eval"],
    )

    train_result = trainer.train()

    print("\n[OK] Entrenamiento terminado, guardando artefactos...")
    trainer.save_model(str(MODEL_OUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUT_DIR))

    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_dataset["train"])

    eval_metrics = trainer.evaluate()
    metrics.update(eval_metrics)

    (LOGS_DIR / "training_log.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("RESUMEN ENTRENAMIENTO")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\n[OK] Adapter LoRA -> {MODEL_OUT_DIR}")
    print(f"[OK] Logs         -> {LOGS_DIR / 'training_log.json'}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()