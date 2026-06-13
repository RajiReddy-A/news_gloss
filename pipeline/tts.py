"""Text-to-speech with gTTS."""

from __future__ import annotations

from pathlib import Path


LANG_TO_GTTS = {
    "hi": "hi",
    "mr": "mr",
    "ta": "ta",
    "te": "te",
    "bn": "bn",
    "kn": "kn",
    "ml": "ml",
}

def warmup() -> None:
    pass


def synthesise(text: str, lang_code: str, output_path: Path) -> tuple[Path, str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    path = _synthesise_gtts(text, lang_code, output_path)
    return path, "gTTS"


def _synthesise_gtts(text: str, lang_code: str, output_path: Path) -> Path:
    from gtts import gTTS

    gtts_lang = LANG_TO_GTTS.get(lang_code, "hi")
    tts = gTTS(text=text, lang=gtts_lang, slow=False)
    tts.save(str(output_path))
    return output_path
