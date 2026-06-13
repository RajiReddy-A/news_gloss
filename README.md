---
title: News in Your Language
emoji: 📻
colorFrom: orange
colorTo: green
sdk: gradio
sdk_version: "6.0.0"
app_file: app.py
pinned: false
tags:
  - news
  - multilingual
  - indian-languages
  - tts
  - accessibility
  - gradio
---

# News in Your Language

A Gradio app that clusters English news into story groups, simplifies each story, translates it into Indian languages, and reads it aloud.

## Runtime Secrets

- `CURRENTS_API_KEY`: Currents API key for live English news.
- `HF_TOKEN`: Hugging Face token for Inference Providers.

## Local Setup

Create a local environment file from the provided template:

```bash
cp .env.example .env
```

Edit `.env` and replace the two placeholder secret values. The app loads this
file automatically. Existing shell or deployment environment variables take
priority over values in `.env`.

Install and run with uv:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run python app.py
```

The `.env` file is ignored by Git and must never be committed.

## Hugging Face Spaces Configuration

Do not upload `.env` to a Space. In the Space repository, open
**Settings → Variables and secrets** and add:

- Secrets: `CURRENTS_API_KEY`, `HF_TOKEN`
- Variables: `HF_MODEL`, `HF_PROVIDER`, `FETCH_INTERVAL_MIN`,
  `TOP_N_PREGEN`, `MAX_ARTICLES`, `SUPPORTED_LANGS`, and `CACHE_DIR`

The optional variables can be omitted to use the defaults below.

## Optional Variables

- `HF_MODEL`: defaults to `Qwen/Qwen3-8B`
- `HF_PROVIDER`: defaults to `auto`
- `HF_MODEL_SUFFIX`: optional provider policy suffix
- `FETCH_INTERVAL_MIN`: defaults to `30`
- `TOP_N_PREGEN`: defaults to `20`
- `CACHE_DIR`: defaults to `.cache`
- `SUPPORTED_LANGS`: defaults to `hi,mr,ta,te,bn,kn,ml`
- `MAX_ARTICLES`: defaults to `20`; requests are always capped at the Currents free-tier limit of `50`

## Notes

We use gTTS to synthesize the translated news bulletins into audio for all supported Indian languages.
