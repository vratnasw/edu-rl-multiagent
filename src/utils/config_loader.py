"""Minimal YAML config loader.

Returns a plain dict; downstream code is free to wrap in SimpleNamespace or
pydantic models in Phase B."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config/config.yaml from the repo root unless overridden."""
    if path is None:
        repo = Path(__file__).resolve().parents[2]
        path = repo / "config" / "config.yaml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))
