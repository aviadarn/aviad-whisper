"""Stage 6 metrics — WER (standard ASR) + word-level F1 (requested gate).

word-level F1: treat each reference/prediction pair as a multiset of words,
count true positives as the multiset intersection, aggregate micro-averaged
precision / recall / F1 over the corpus.
"""
from __future__ import annotations

import re
from collections import Counter

import jiwer

_PUNCT = re.compile(r"[^\w\s']")


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_f1(predictions: list[str], references: list[str]) -> dict[str, float]:
    tp = fp = fn = 0
    for pred, ref in zip(predictions, references):
        pc = Counter(pred.split())
        rc = Counter(ref.split())
        inter = pc & rc  # multiset intersection
        tp_i = sum(inter.values())
        tp += tp_i
        fp += sum(pc.values()) - tp_i
        fn += sum(rc.values()) - tp_i

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_all(predictions: list[str], references: list[str], do_normalize: bool = True) -> dict[str, float]:
    if do_normalize:
        predictions = [normalize(p) for p in predictions]
        references = [normalize(r) for r in references]
    # jiwer errors on empty refs; guard with a space.
    safe_refs = [r if r else " " for r in references]
    wer = jiwer.wer(safe_refs, predictions)
    out = {"wer": wer}
    out.update(word_f1(predictions, references))
    return out
