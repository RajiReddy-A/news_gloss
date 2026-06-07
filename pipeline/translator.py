"""Translate English bulletins into supported Indian languages."""

from __future__ import annotations

from pipeline.config import LANG_NAMES
from pipeline.llm import chat_complete, is_configured


def translate(english_summary: str, lang_code: str) -> str:
    lang_name = LANG_NAMES.get(lang_code, lang_code)
    if not english_summary.strip():
        return ""
    if not is_configured():
        return f"[{lang_name} demo translation] {english_summary}"

    return chat_complete(
        messages=[
            {
                "role": "system",
                "content": (
                    "/no_think\n"
                    f"Translate the text into {lang_name}. Keep it natural for spoken "
                    "listening, not formal writing. Preserve the meaning exactly. "
                    "Do not add facts. Output only the translated text."
                ),
            },
            {"role": "user", "content": english_summary},
        ],
        max_tokens=360,
        temperature=0.2,
    )
