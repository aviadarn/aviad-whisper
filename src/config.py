"""Load and access the YAML config. One source of truth for every stage."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "config.yaml")


@dataclass
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)


def load_config(path: str | None = None) -> Config:
    path = path or DEFAULT_CONFIG
    with open(path) as f:
        return Config(yaml.safe_load(f))


def resolve_path(rel: str) -> str:
    """Resolve a config-relative path against the repo root."""
    if os.path.isabs(rel):
        return rel
    return os.path.join(ROOT, rel)
