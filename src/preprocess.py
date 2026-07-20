"""Stage 2/3 — feature extraction, tokenization, and the batching collator.

- build_processor: loads the WhisperProcessor (feature extractor + tokenizer)
- prepare_dataset:  audio -> log-mel input_features, text -> token label ids
- DataCollatorSpeechSeq2SeqWithPadding: dynamic padding into batches
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import WhisperProcessor

from src.config import Config


def build_processor(cfg: Config) -> WhisperProcessor:
    m = cfg["model"]
    return WhisperProcessor.from_pretrained(
        m["name"], language=m["language"], task=m["task"]
    )


def make_prepare_fn(processor: WhisperProcessor, sampling_rate: int):
    fe = processor.feature_extractor
    tok = processor.tokenizer

    def prepare(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch["audio"]
        batch["input_features"] = fe(
            audio["array"], sampling_rate=sampling_rate
        ).input_features[0]
        batch["labels"] = tok(batch["text"]).input_ids
        return batch

    return prepare


def prepare_dataset(dataset, processor: WhisperProcessor, sampling_rate: int):
    prepare = make_prepare_fn(processor, sampling_rate)
    remove = dataset.column_names
    if isinstance(remove, dict):  # DatasetDict
        remove = next(iter(remove.values()))
    return dataset.map(prepare, remove_columns=remove)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads audio features and label ids independently, masks pad tokens in loss."""

    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so it is ignored by the loss.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If a BOS token was prepended by the tokenizer, drop it (model adds it).
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
