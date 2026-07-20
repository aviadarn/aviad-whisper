"""End-to-end: audio -> split -> batch -> train -> validate (WER + F1).

    python -m src.pipeline                 # full run with config.yaml
    python -m src.pipeline --config x.yaml  # alternate config
    python -m src.pipeline --skip-train     # eval an existing checkpoint only
"""
from __future__ import annotations

import argparse
import json

from src.config import load_config, resolve_path
from src.data import load_splits
from src.evaluate_model import evaluate_on_test
from src.preprocess import build_processor, prepare_dataset
from src.train import train_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    sr = cfg["dataset"]["sampling_rate"]

    print("== [1/4] load + split ==")
    splits = load_splits(cfg)
    for name, s in splits.items():
        print(f"   {name:12s} {len(s):5d}")

    print("== [2/4] processor + preprocess (features, labels, batching) ==")
    processor = build_processor(cfg)
    prepared = prepare_dataset(splits, processor, sr)

    if not args.skip_train:
        print("== [3/4] train ==")
        train_model(cfg, prepared, processor)
    else:
        print("== [3/4] train SKIPPED ==")

    print("== [4/4] validate on held-out test ==")
    scores = evaluate_on_test(cfg, prepared["test"], processor)

    print("\n===== RESULTS (test split) =====")
    print(f"  WER        : {scores['wer']:.4f}   (lower is better)")
    print(f"  F1 (word)  : {scores['f1']:.4f}   (higher is better)")
    print(f"  precision  : {scores['precision']:.4f}")
    print(f"  recall     : {scores['recall']:.4f}")

    out = resolve_path("models/test_metrics.json")
    with open(out, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
