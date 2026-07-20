"""Stage 6 — run the trained model over the held-out test split and report WER + F1."""
from __future__ import annotations

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.config import Config, resolve_path
from src.metrics import compute_all
from src.train import pick_device


@torch.no_grad()
def evaluate_on_test(cfg: Config, test_split, processor: WhisperProcessor,
                     model_dir: str | None = None) -> dict[str, float]:
    device = pick_device()
    model_dir = model_dir or resolve_path(cfg["train"]["output_dir"])
    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    bs = cfg["train"]["per_device_eval_batch_size"]
    max_len = cfg["train"]["generation_max_length"]
    preds: list[str] = []
    refs: list[str] = []

    for start in range(0, len(test_split), bs):
        rows = test_split[start:start + bs]
        feats = torch.tensor(rows["input_features"]).to(device)
        gen = model.generate(input_features=feats, max_length=max_len)
        preds.extend(processor.tokenizer.batch_decode(gen, skip_special_tokens=True))
        labels = rows["labels"]
        for ids in labels:
            clean = [i for i in ids if i >= 0]  # strip any -100
            refs.append(processor.tokenizer.decode(clean, skip_special_tokens=True))

    scores = compute_all(preds, refs, cfg["eval"]["normalize"])
    return scores
