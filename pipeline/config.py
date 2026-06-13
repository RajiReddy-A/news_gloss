"""Runtime configuration helpers."""

from __future__ import annotations

import os

from settings import load_environment


load_environment()

SUPPORTED_LANGS_DEFAULT = "hi,mr,ta,te,bn,kn,ml"

LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "bn": "Bengali",
    "kn": "Kannada",
    "ml": "Malayalam",
}


def get_supported_langs() -> list[str]:
    raw = os.environ.get("SUPPORTED_LANGS", SUPPORTED_LANGS_DEFAULT)
    langs = [lang.strip() for lang in raw.split(",") if lang.strip()]
    return [lang for lang in langs if lang in LANG_NAMES]


def get_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
