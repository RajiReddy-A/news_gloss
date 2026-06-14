"""Text-to-speech with gTTS."""

from __future__ import annotations

from pathlib import Path


import asyncio
import logging

logger = logging.getLogger(__name__)

LANG_TO_VOICE = {
    "hi": "hi-IN-SwaraNeural",
    "mr": "mr-IN-AarohiNeural",
    "ta": "ta-IN-PallaviNeural",
    "te": "te-IN-ShrutiNeural",
    "bn": "bn-IN-TanishaaNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "en": "en-IN-NeerjaNeural"
}

LANG_TO_GTTS = {
    "hi": "hi",
    "mr": "mr",
    "ta": "ta",
    "te": "te",
    "bn": "bn",
    "kn": "kn",
    "ml": "ml",
    "en": "en",
}

def warmup() -> None:
    pass


def synthesise(text: str, lang_code: str, output_path: Path) -> tuple[Path, str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        path = _synthesise_gtts(text, lang_code, output_path)
        return path, "gTTS"
    except Exception as e:
        logger.warning(f"gTTS failed ({e}), falling back to Edge TTS")
        try:
            path = _synthesise_edge(text, lang_code, output_path)
            return path, "Edge TTS (Neural)"
        except Exception as e2:
            logger.exception("Both TTS engines failed")
            raise e2

async def _amain(text: str, voice: str, output_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def _synthesise_edge(text: str, lang_code: str, output_path: Path) -> Path:
    voice = LANG_TO_VOICE.get(lang_code, "hi-IN-SwaraNeural")
    asyncio.run(_amain(text, voice, str(output_path)))
    return output_path

def _synthesise_gtts(text: str, lang_code: str, output_path: Path) -> Path:
    from gtts import gTTS

    gtts_lang = LANG_TO_GTTS.get(lang_code, "hi")
    tts = gTTS(text=text, lang=gtts_lang, slow=False)
    tts.save(str(output_path))
    return output_path
