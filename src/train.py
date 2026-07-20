"""Stage 4/5 — fine-tune Whisper with the HF Seq2SeqTrainer on Apple MPS.

Run alone (assumes prepared splits via the pipeline), or import train_model().
"""
from __future__ import annotations

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from src.config import Config, resolve_path
from src.metrics import compute_all
from src.preprocess import DataCollatorSpeechSeq2SeqWithPadding


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_model(cfg: Config, splits, processor: WhisperProcessor):
    m = cfg["model"]
    t = cfg["train"]
    device = pick_device()
    print(f"[train] device = {device}")

    model = WhisperForConditionalGeneration.from_pretrained(m["name"])
    # Let generation config drive language/task; clear forced ids to avoid conflicts.
    model.generation_config.language = m["language"]
    model.generation_config.task = m["task"]
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        scores = compute_all(pred_str, label_str, cfg["eval"]["normalize"])
        # Trainer expects flat scalar metrics.
        return {"wer": scores["wer"], "f1": scores["f1"],
                "precision": scores["precision"], "recall": scores["recall"]}

    args = Seq2SeqTrainingArguments(
        output_dir=resolve_path(t["output_dir"]),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=float(t["learning_rate"]),
        warmup_steps=t["warmup_steps"],
        max_steps=t["max_steps"],
        gradient_checkpointing=False,
        fp16=t["fp16"],
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_steps=t["save_steps"],
        logging_steps=t["logging_steps"],
        predict_with_generate=True,
        generation_max_length=t["generation_max_length"],
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_pin_memory=False,   # MPS: pinning is a no-op / can warn
        seed=t["seed"],
    )

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=splits["train"],
        eval_dataset=splits["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(resolve_path(t["output_dir"]))
    processor.save_pretrained(resolve_path(t["output_dir"]))
    return trainer
