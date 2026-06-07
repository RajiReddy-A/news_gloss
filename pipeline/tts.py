"""Text-to-speech with IndicF5 first and gTTS fallback."""

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

_indic_model = None
_indic_error: str | None = None


def _load_indicf5():
    global _indic_model, _indic_error
    if _indic_model is not None or _indic_error is not None:
        return _indic_model

    try:
        from transformers import AutoModel

        _indic_model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True)
    except Exception as exc:
        _indic_error = str(exc)
        _indic_model = None
    return _indic_model


def warmup() -> None:
    _load_indicf5()


def synthesise(text: str, lang_code: str, output_path: Path) -> tuple[Path, str]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = _load_indicf5()
    ref_audio = Path("assets") / f"ref_{lang_code}.wav"
    if model is not None and ref_audio.exists():
        try:
            path = _synthesise_indicf5(model, text, lang_code, output_path, ref_audio)
            return path, "IndicF5 (Local Model)"
        except Exception:
            pass

    path = _synthesise_gtts(text, lang_code, output_path)
    return path, "gTTS (Google TTS Fallback)"


def _synthesise_indicf5(model, text: str, lang_code: str, output_path: Path, ref_audio: Path) -> Path:
    import numpy as np
    import soundfile as sf
    from pydub import AudioSegment

    wav_path = output_path.with_suffix(".wav")
    audio, sample_rate = model(
        text,
        ref_audio_path=str(ref_audio),
        ref_text="",
        language=lang_code,
    )
    sf.write(str(wav_path), np.array(audio), sample_rate)
    AudioSegment.from_wav(str(wav_path)).export(str(output_path), format="mp3")
    return output_path


def _synthesise_gtts(text: str, lang_code: str, output_path: Path) -> Path:
    from gtts import gTTS

    gtts_lang = LANG_TO_GTTS.get(lang_code, "hi")
    tts = gTTS(text=text, lang=gtts_lang, slow=False)
    tts.save(str(output_path))
    return output_path
