"""
Tiny file-based cache utilities (generic).

Embeddings caching is already implemented in CachedEmbedder.
This module exists for future caching needs (e.g., parsed PDFs, retrieval results).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional


def _key_to_filename(key: str) -> str:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return f"{h}.json"


def cache_get(cache_dir: Path, key: str) -> Optional[Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / _key_to_filename(key)
    if not fp.exists():
        return None
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def cache_set(cache_dir: Path, key: str, value: Any) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = cache_dir / _key_to_filename(key)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
