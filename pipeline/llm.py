"""Hugging Face Inference Providers client wrapper."""

from __future__ import annotations

import os
import re
import time

from settings import load_environment


load_environment()

MODEL = os.environ.get("HF_MODEL", "Qwen/Qwen3-8B")
MODEL_SUFFIX = os.environ.get("HF_MODEL_SUFFIX", "")
PROVIDER = os.environ.get("HF_PROVIDER", "auto")
MAX_RETRIES = int(os.environ.get("HF_MAX_RETRIES", "3"))

_client: InferenceClient | None = None


def _get_client() -> InferenceClient:
    global _client
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set")
    if _client is None:
        from huggingface_hub import InferenceClient

        _client = InferenceClient(api_key=token, provider=PROVIDER)
    return _client


def chat_complete(
    messages: list[dict[str, str]],
    max_tokens: int = 300,
    temperature: float = 0.2,
) -> str:
    for attempt in range(MAX_RETRIES):
        try:
            output = _get_client().chat_completion(
                messages=messages,
                model=f"{MODEL}{MODEL_SUFFIX}",
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return _clean_qwen_output(output.choices[0].message.content or "")
        except Exception as exc:
            if not _should_retry(exc) or attempt >= MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt)

    raise RuntimeError("Hugging Face chat completion failed")


def _should_retry(exc: Exception) -> bool:
    if isinstance(exc, RuntimeError):
        return False

    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    return exc.__class__.__name__ == "InferenceTimeoutError"


def _clean_qwen_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def is_configured() -> bool:
    return bool(os.environ.get("HF_TOKEN"))
