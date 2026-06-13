"""Two-level cache: in-memory first, JSON/audio files under /tmp second."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from settings import load_environment


load_environment()

CACHE_DIR = Path(os.environ.get("CACHE_DIR", ".cache"))
AUDIO_DIR = CACHE_DIR / "audio"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

_mem: dict[str, Any] = {}


def cache_key(url: str, lang: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:12] + "_" + lang


def _json_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _read_json(key: str) -> dict[str, Any] | None:
    path = _json_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(key: str, payload: dict[str, Any]) -> None:
    _json_path(key).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def is_cached(url: str, lang: str) -> bool:
    key = cache_key(url, lang)
    return key in _mem or _json_path(key).exists()


def save_summary(url: str, text: str) -> None:
    save_text(url, "en", text)


def get_summary(url: str) -> str | None:
    return get_text(url, "en")


def save_translation(url: str, lang: str, text: str) -> None:
    save_text(url, lang, text)


def get_translation(url: str, lang: str) -> str | None:
    return get_text(url, lang)


def save_text(url: str, lang: str, text: str) -> None:
    key = cache_key(url, lang)
    _mem[key] = text
    _write_json(key, {"url": url, "lang": lang, "text": text})


def get_text(url: str, lang: str) -> str | None:
    key = cache_key(url, lang)
    if key in _mem:
        value = _mem[key]
        return value if isinstance(value, str) else None

    payload = _read_json(key)
    if not payload:
        return None

    text = payload.get("text")
    if isinstance(text, str):
        _mem[key] = text
        return text
    return None


def audio_path(url: str, lang: str) -> Path:
    return AUDIO_DIR / f"{cache_key(url, lang)}.mp3"


def save_articles(clusters: list[dict[str, Any]]) -> None:
    _mem["__clusters__"] = clusters
    _write_json("__articles__", {"clusters": clusters})


def get_articles() -> list[dict[str, Any]]:
    clusters = _mem.get("__clusters__")
    if isinstance(clusters, list):
        return clusters

    payload = _read_json("__articles__")
    if not payload:
        return []

    stored = payload.get("clusters")
    if isinstance(stored, list):
        _mem["__clusters__"] = stored
        return stored
    return []


def clear_memory_cache() -> None:
    _mem.clear()
