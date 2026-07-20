"""Stage 1 — load the dataset and split into train / val / test.

Run alone:  python -m src.data
Returns a DatasetDict with 'train', 'validation', 'test' splits.
"""
from __future__ import annotations

from datasets import Audio, DatasetDict, load_dataset

from src.config import Config, load_config


def load_splits(cfg: Config) -> DatasetDict:
    ds_cfg = cfg["dataset"]
    sp_cfg = cfg["split"]

    # MINDS-14 ships a single 'train' split; we carve val/test out of it.
    full = load_dataset(ds_cfg["name"], ds_cfg["config"], split="train")

    # Keep only the columns we need, standardize names -> audio / text.
    keep = {ds_cfg["audio_column"], ds_cfg["text_column"]}
    drop = [c for c in full.column_names if c not in keep]
    full = full.remove_columns(drop)
    if ds_cfg["text_column"] != "text":
        full = full.rename_column(ds_cfg["text_column"], "text")
    if ds_cfg["audio_column"] != "audio":
        full = full.rename_column(ds_cfg["audio_column"], "audio")

    # Ensure audio is decoded at the model's sampling rate (16 kHz).
    full = full.cast_column("audio", Audio(sampling_rate=ds_cfg["sampling_rate"]))

    seed = sp_cfg["seed"]
    test_size = sp_cfg["test_size"]
    val_size = sp_cfg["val_size"]

    # First peel off the test set, then split the remainder into train/val.
    first = full.train_test_split(test_size=test_size, seed=seed)
    remaining = first["train"]
    test = first["test"]

    val_rel = val_size / (1.0 - test_size)
    second = remaining.train_test_split(test_size=val_rel, seed=seed)

    return DatasetDict(train=second["train"], validation=second["test"], test=test)


if __name__ == "__main__":
    cfg = load_config()
    splits = load_splits(cfg)
    for name, split in splits.items():
        print(f"{name:12s} {len(split):5d} examples")
    print("sample text:", splits["train"][0]["text"])
