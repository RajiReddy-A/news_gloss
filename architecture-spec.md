# Multilingual News Reader — Architecture Spec

---

## Project overview

A Gradio app hosted on Hugging Face Spaces that:
1. Fetches English news every 30 minutes
2. Clusters related articles into story groups (like Google News)
3. When a user clicks a cluster and picks a language, simplifies + translates the story and reads it aloud
4. Caches everything so the second user to request the same story+language gets instant audio playback

**Target users:** People who cannot read — elderly, low-literacy, or visually impaired. The app converts dense news into a warm, spoken radio-bulletin in their language.

**Hackathon constraint:** All models ≤ 32B parameters total. No cloud LLM APIs except Groq (which runs open-source models). TTS runs locally.

---

## Tech stack — final decisions

| Component               | Tool                                 | Reason                                                      |
| -------------------------| --------------------------------------| -------------------------------------------------------------|
| News data               | Currents API                         | Free, 1000 req/day, real-time English news                  |
| Clustering              | BERTopic + all-MiniLM-L6-v2          | Industry standard, zero config, fast on CPU                 |
| Article text extraction | trafilatura                          | Scrapes full article body from URL                          |
| Summarise + Translate   | Huggingface Inference API — Qwen3-8B | Free tier, 14,400 req/day, strong multilingual              |
| TTS                     | gTTS                                 | Uses Google Translate TTS API, instant, reliable            |
| UI framework            | Gradio 6.x — gr.Blocks()             | Required by hackathon rules                                 |
| Hosting                 | Hugging Face Spaces — CPU Basic      | Free tier: 2 vCPU, 16 GB RAM                                |
| Background jobs         | APScheduler — BackgroundScheduler    | Runs in same process as Gradio                              |
| Caching                 | In-memory dict + /tmp JSON files     | Two-level: fast in-memory, disk fallback                    |

---

## Supported languages

| Code | Language |
|---|---|
| `hi` | Hindi |
| `mr` | Marathi |
| `ta` | Tamil |
| `te` | Telugu |
| `bn` | Bengali |
| `kn` | Kannada |
| `ml` | Malayalam |

---

## Project file structure

```
your-space/
├── app.py                  # Gradio UI + all event handlers
├── scheduler.py            # APScheduler setup — fetch + cluster + pre-generate
├── pipeline/
│   ├── __init__.py
│   ├── fetcher.py          # Currents API calls
│   ├── clusterer.py        # BERTopic wrapper — embed + cluster + label
│   ├── summariser.py       # Groq Qwen3-8B — simplify to spoken English
│   ├── translator.py       # Groq Qwen3-8B — translate simplified text to target lang
│   └── tts.py              # gTTS inference logic
├── cache/
│   ├── __init__.py
│   ├── store.py            # read/write helpers for all cache files
│   └── audio/              # generated .mp3 files — named by cache key
├── requirements.txt
└── README.md               # HF Space card metadata
```

---

## Environment variables

Set these in HF Spaces → Settings → Variables and Secrets.

| Variable | Type | Value |
|---|---|---|
| `CURRENTS_API_KEY` | Secret | Your Currents API key |
| `GROQ_API_KEY` | Secret | Your Groq API key |
| `FETCH_INTERVAL_MIN` | Variable | `30` |
| `TOP_N_PREGEN` | Variable | `10` |
| `GROQ_MODEL` | Variable | `qwen3-8b` |
| `CACHE_DIR` | Variable | `/tmp/cache` |
| `SUPPORTED_LANGS` | Variable | `hi,mr,ta,te,bn,kn,ml` |
| `MAX_ARTICLES` | Variable | `100` |

Access in Python:
```python
import os
CURRENTS_API_KEY = os.environ["CURRENTS_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
```

---

## requirements.txt

```
gradio>=6.0.0,<7.0.0
apscheduler>=3.10
requests>=2.31
bertopic>=0.16
sentence-transformers>=3.0
trafilatura>=1.8
groq>=0.9
transformers>=4.40
soundfile>=0.12
numpy>=1.24
gtts>=2.5
pydub>=0.25
```

---

## HF Spaces constraints — critical, design around these

### 1. Storage is ephemeral on free tier
- Disk resets on every sleep/wake cycle
- `/tmp` survives within a session but not across restarts
- **Mitigation:** Keep an in-memory Python dict as primary cache. Write to `/tmp` as secondary. On startup, always re-fetch and re-cluster. Do not assume mp3 files survive a restart.

### 2. Space sleeps after 48 hours of inactivity
- Free CPU tier goes to sleep when no requests come in
- **Mitigation:** Since we are using gTTS, cold starts are practically instantaneous and require minimal RAM.

### 3. APScheduler shares the Gradio process
- No separate worker process
- Scheduler jobs must be non-blocking — use `ThreadPoolExecutor`
- BERTopic on 100 articles: ~2s on CPU — acceptable
- Full pre-generation of 10 stories × 7 languages takes ~5 min — run async, don't block

### 4. RAM budget
- TTS (gTTS): Minimal RAM
- BERTopic + all-MiniLM-L6-v2: ~300 MB
- Groq calls: outbound HTTP, zero RAM
- Total: ~1.5 GB — well within the 16 GB free ceiling

### 5. API keys
- Never commit keys to the repo
- Use HF Secrets exclusively
- Access via `os.environ`

---

## Startup sequence

```
App starts
    │
    ├── Load IndicF5 into RAM (~5s)
    │
    ├── Create CACHE_DIR if not exists
    │
    ├── Start APScheduler (BackgroundScheduler)
    │       └── Fire immediately: fetch → cluster → pre-generate top 10
    │
    └── Launch Gradio UI (gr.Blocks().launch())
```

---

## Background scheduler — scheduler.py

```python
from apscheduler.schedulers.background import BackgroundScheduler
from pipeline.fetcher import fetch_articles
from pipeline.clusterer import cluster_articles
from pipeline.summariser import summarise
from pipeline.translator import translate
from cache.store import save_articles, save_summary, save_translation, is_cached
import os

FETCH_INTERVAL = int(os.environ.get("FETCH_INTERVAL_MIN", 30))
TOP_N = int(os.environ.get("TOP_N_PREGEN", 10))
LANGS = os.environ.get("SUPPORTED_LANGS", "hi,mr,ta").split(",")

def refresh_news():
    articles = fetch_articles()           # list of dicts: {url, title, description, image}
    clusters = cluster_articles(articles) # list of dicts: {label, articles: [...]}
    save_articles(clusters)

    # Pre-generate summaries + translations for top N clusters
    for cluster in clusters[:TOP_N]:
        rep = cluster["articles"][0]      # representative article
        url = rep["url"]

        # Summarise once in English
        if not is_cached(url, "en"):
            text = fetch_full_text(url)   # trafilatura
            summary_en = summarise(text)
            save_summary(url, summary_en)

        # Translate into all supported languages
        for lang in LANGS:
            if not is_cached(url, lang):
                translated = translate(summary_en, lang)
                save_translation(url, lang, translated)

scheduler = BackgroundScheduler()
scheduler.add_job(refresh_news, "interval", minutes=FETCH_INTERVAL)
scheduler.start()
refresh_news()  # run immediately on startup
```

---

## Cache store — cache/store.py

```python
import json, hashlib, os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/tmp/cache"))
AUDIO_DIR = CACHE_DIR / "audio"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

_mem = {}  # in-memory primary cache

def _key(url: str, lang: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12] + "_" + lang

def is_cached(url: str, lang: str) -> bool:
    k = _key(url, lang)
    return k in _mem or (CACHE_DIR / f"{k}.json").exists()

def save_summary(url: str, text: str):
    k = _key(url, "en")
    _mem[k] = text
    (CACHE_DIR / f"{k}.json").write_text(json.dumps({"text": text}))

def get_summary(url: str) -> str | None:
    k = _key(url, "en")
    if k in _mem:
        return _mem[k]
    p = CACHE_DIR / f"{k}.json"
    if p.exists():
        return json.loads(p.read_text())["text"]
    return None

def save_translation(url: str, lang: str, text: str):
    k = _key(url, lang)
    _mem[k] = text
    (CACHE_DIR / f"{k}.json").write_text(json.dumps({"text": text}))

def get_translation(url: str, lang: str) -> str | None:
    k = _key(url, lang)
    if k in _mem:
        return _mem[k]
    p = CACHE_DIR / f"{k}.json"
    if p.exists():
        return json.loads(p.read_text())["text"]
    return None

def audio_path(url: str, lang: str) -> Path:
    return AUDIO_DIR / f"{_key(url, lang)}.mp3"

def save_articles(clusters: list):
    _mem["__clusters__"] = clusters
    (CACHE_DIR / "articles.json").write_text(json.dumps(clusters))

def get_articles() -> list:
    if "__clusters__" in _mem:
        return _mem["__clusters__"]
    p = CACHE_DIR / "articles.json"
    if p.exists():
        return json.loads(p.read_text())
    return []
```

---

## pipeline/fetcher.py

```python
import requests, os

API_KEY = os.environ["CURRENTS_API_KEY"]
BASE_URL = "https://api.currentsapi.services/v1/latest-news"

def fetch_articles(language="en", max_results=100) -> list[dict]:
    resp = requests.get(BASE_URL, params={
        "apiKey": API_KEY,
        "language": language,
        "page_size": max_results,
    }, timeout=10)
    resp.raise_for_status()
    news = resp.json().get("news", [])
    return [
        {
            "url": a["url"],
            "title": a["title"],
            "description": a.get("description", ""),
            "image": a.get("image", ""),
            "published": a.get("published", ""),
            "category": a.get("category", []),
        }
        for a in news
        if a.get("url") and a.get("title")
    ]
```

---

## pipeline/clusterer.py

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # 80MB, loaded once

def cluster_articles(articles: list[dict]) -> list[dict]:
    if len(articles) < 5:
        return [{"label": "News", "articles": articles}]

    texts = [a["title"] + ". " + a["description"] for a in articles]
    embeddings = _embed_model.encode(texts, show_progress_bar=False)

    topic_model = BERTopic(
        embedding_model=_embed_model,
        min_topic_size=2,
        nr_topics="auto",
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings)

    # Group articles by topic
    clusters = {}
    for idx, topic_id in enumerate(topics):
        if topic_id == -1:  # outlier — skip or put in "Other"
            continue
        clusters.setdefault(topic_id, []).append(articles[idx])

    # Get topic label from BERTopic
    topic_info = topic_model.get_topic_info()
    result = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1 or tid not in clusters:
            continue
        label = row["Name"].replace("_", " ").title()
        result.append({
            "label": label,
            "articles": clusters[tid],
            "count": len(clusters[tid]),
        })

    return sorted(result, key=lambda c: c["count"], reverse=True)
```

---

## pipeline/summariser.py

```python
from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = os.environ.get("GROQ_MODEL", "qwen3-8b")

SYSTEM_PROMPT = """You are rewriting a news article to be read aloud as a radio bulletin.
Rules:
- Maximum 80 words
- First sentence: state the single most important fact
- Use simple spoken language — no jargon, no passive voice
- Third sentence onwards: explain briefly
- Final sentence: "What this means for you:" followed by one practical implication
- Output plain text only, no markdown"""

def summarise(article_text: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Article:\n{article_text[:3000]}"},
        ],
        max_tokens=200,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()
```

---

## pipeline/translator.py

```python
from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = os.environ.get("GROQ_MODEL", "qwen3-8b")

LANG_NAMES = {
    "hi": "Hindi", "mr": "Marathi", "ta": "Tamil",
    "te": "Telugu", "bn": "Bengali", "kn": "Kannada", "ml": "Malayalam",
}

def translate(english_summary: str, lang_code: str) -> str:
    lang_name = LANG_NAMES.get(lang_code, lang_code)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Translate the following text into {lang_name}. "
                    "Keep the translation natural for spoken listening, not formal writing. "
                    "Do not add anything. Output only the translated text."
                ),
            },
            {"role": "user", "content": english_summary},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()
```

---

## pipeline/tts.py

```python
import soundfile as sf
import numpy as np
from pathlib import Path
from gtts import gTTS

# Attempt to load IndicF5 — fall back to gTTS if unavailable
try:
    from transformers import AutoModel
    _indic_model = AutoModel.from_pretrained(
        "ai4bharat/IndicF5",
        trust_remote_code=True,
    )
    _use_indicf5 = True
    print("IndicF5 loaded successfully")
except Exception as e:
    print(f"IndicF5 load failed ({e}), falling back to gTTS")
    _use_indicf5 = False

LANG_TO_GTTS = {
    "hi": "hi", "mr": "mr", "ta": "ta",
    "te": "te", "bn": "bn", "kn": "kn", "ml": "ml",
}

def synthesise(text: str, lang_code: str, output_path: Path) -> Path:
    output_path = Path(output_path)
    if _use_indicf5:
        return _synthesise_indicf5(text, lang_code, output_path)
    else:
        return _synthesise_gtts(text, lang_code, output_path)

def _synthesise_indicf5(text: str, lang_code: str, output_path: Path) -> Path:
    # IndicF5 requires a reference audio prompt — use a bundled default per language
    # See: https://github.com/AI4Bharat/IndicF5 for reference audio format
    audio, sr = _indic_model(
        text,
        ref_audio_path=f"assets/ref_{lang_code}.wav",  # include short reference clips
        ref_text="",
    )
    sf.write(str(output_path.with_suffix(".wav")), np.array(audio), sr)
    # Convert wav to mp3 using pydub
    from pydub import AudioSegment
    AudioSegment.from_wav(str(output_path.with_suffix(".wav"))).export(
        str(output_path), format="mp3"
    )
    return output_path

def _synthesise_gtts(text: str, lang_code: str, output_path: Path) -> Path:
    gtts_lang = LANG_TO_GTTS.get(lang_code, "hi")
    tts = gTTS(text=text, lang=gtts_lang, slow=False)
    tts.save(str(output_path))
    return output_path
```

---

## app.py — Gradio UI

```python
import gradio as gr
import os
from cache.store import get_articles, get_summary, get_translation, audio_path
from pipeline.summariser import summarise
from pipeline.translator import translate
from pipeline.tts import synthesise
from pipeline.fetcher import fetch_articles as _fetch_full_text  # reuse trafilatura
import trafilatura
import scheduler  # importing starts the APScheduler

LANGS = {
    "Hindi": "hi", "Marathi": "mr", "Tamil": "ta",
    "Telugu": "te", "Bengali": "bn", "Kannada": "kn", "Malayalam": "ml",
}

def get_clusters_for_display():
    clusters = get_articles()
    if not clusters:
        return [["No news loaded yet. Please wait 30 seconds for the first fetch."]]
    return [[f"{c['label']} ({c['count']} articles)"] for c in clusters]

def handle_cluster_click(selected_label: str):
    clusters = get_articles()
    cluster = next((c for c in clusters if c["label"] in selected_label), None)
    if not cluster:
        return "Cluster not found", []
    headlines = [a["title"] for a in cluster["articles"][:5]]
    rep_url = cluster["articles"][0]["url"]
    return rep_url, "\n".join(f"• {h}" for h in headlines)

def handle_play(rep_url: str, lang_display: str):
    if not rep_url:
        return None, "Please select a news cluster first."

    lang_code = LANGS.get(lang_display, "hi")
    ap = audio_path(rep_url, lang_code)

    # Cache hit — return immediately
    if ap.exists():
        return str(ap), "Playing from cache."

    # Cache miss — generate
    yield None, "Fetching article..."
    raw = trafilatura.fetch_url(rep_url)
    text = trafilatura.extract(raw) or ""

    summary_en = get_summary(rep_url)
    if not summary_en:
        yield None, "Simplifying..."
        summary_en = summarise(text[:3000])

    yield None, f"Translating to {lang_display}..."
    translated = get_translation(rep_url, lang_code)
    if not translated:
        translated = translate(summary_en, lang_code)

    yield None, "Generating audio..."
    synthesise(translated, lang_code, ap)

    yield str(ap), f"Done. Playing in {lang_display}."

with gr.Blocks(title="News in Your Language") as demo:
    gr.Markdown("## News in Your Language\nClick a story cluster, pick your language, and listen.")

    with gr.Row():
        with gr.Column(scale=1):
            cluster_list = gr.Dataframe(
                headers=["Story cluster"],
                datatype=["str"],
                value=get_clusters_for_display,
                every=1800,  # refresh every 30 min
                interactive=False,
                label="Today's stories",
            )
            refresh_btn = gr.Button("Refresh now", size="sm")

        with gr.Column(scale=2):
            selected_url = gr.State("")
            headlines_box = gr.Textbox(
                label="Headlines in this cluster",
                lines=6,
                interactive=False,
            )
            lang_picker = gr.Radio(
                choices=list(LANGS.keys()),
                value="Hindi",
                label="Listen in",
            )
            play_btn = gr.Button("Play audio bulletin", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False, max_lines=1)
            audio_out = gr.Audio(label="Audio bulletin", type="filepath", autoplay=True)

    # Events
    cluster_list.select(
        fn=handle_cluster_click,
        inputs=[cluster_list],
        outputs=[selected_url, headlines_box],
    )
    play_btn.click(
        fn=handle_play,
        inputs=[selected_url, lang_picker],
        outputs=[audio_out, status_box],
    )
    refresh_btn.click(
        fn=lambda: get_clusters_for_display(),
        outputs=[cluster_list],
    )

demo.launch()
```

---

## Data flow summary

### Happy path (cache hit)
```
User clicks cluster → selects language → clicks Play
    → cache lookup (url_hash + lang_code)
    → mp3 found → serve instantly (~0.1s)
```

### Cache miss path
```
User clicks cluster → selects language → clicks Play
    → cache lookup → miss
    → trafilatura fetches full article text
    → check English summary cache
        → miss: Groq Qwen3-8B summarise (~2s)
        → hit: skip
    → check translation cache
        → miss: Groq Qwen3-8B translate (~2s)
        → hit: skip
    → IndicF5 TTS generate (~5–8s)
    → save mp3 to audio/ dir + update in-memory cache
    → stream mp3 to gr.Audio
    Total: ~8–12s first time, instant for all subsequent users
```

### Background pre-generation
```
APScheduler fires every 30 min
    → Currents API: fetch 100 latest English articles
    → BERTopic: embed + cluster → N story groups
    → For top 10 clusters:
        → trafilatura: fetch full text
        → Groq: summarise in English → cache
        → Groq: translate to all 7 languages → cache each
        (TTS not pre-generated to save time — generated on first user request)
    → save clusters to articles.json + in-memory dict
```

---

## Cache key convention

```python
key = sha256(article_url.encode()).hexdigest()[:12] + "_" + lang_code
# Examples:
# a3f9c12b4e1d_en  → English summary
# a3f9c12b4e1d_hi  → Hindi translation text
# a3f9c12b4e1d_mr  → Marathi translation text (mp3 at audio/a3f9c12b4e1d_mr.mp3)
```

---

## Groq rate limit strategy

- Free tier: Qwen3-8B — ~14,400 req/day
- Each article needs: 1 summarise call + N translate calls (N = number of languages)
- Pre-generating 10 stories × 7 languages = 80 Groq calls per 30-min cycle
- At 48 cycles/day max = 3,840 calls/day — well within the 14,400 daily limit
- If rate limited: exponential backoff, skip pre-gen, generate on demand instead

```python
import time
from groq import RateLimitError

def groq_call_with_retry(fn, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
```

---

## Build order (recommended)

1. `pipeline/fetcher.py` — get raw articles, print them, verify API key works
2. `pipeline/clusterer.py` — cluster the articles, print cluster labels
3. `app.py` (news grid only) — show clusters in Gradio, no audio yet
4. `pipeline/summariser.py` — test Groq summarise on one article
5. `pipeline/translator.py` — test Groq translate to Hindi
6. `pipeline/tts.py` — test gTTS first (simpler), then swap to IndicF5
7. `cache/store.py` — wire up caching, verify hit/miss logic
8. `app.py` (full) — wire play button → full pipeline
9. `scheduler.py` — add APScheduler, test background pre-generation
10. Deploy to HF Spaces — set secrets, push repo, verify cold start

---

## README.md for HF Space card

```yaml
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
```

**Description for the Space card:**
> Converts English news into spoken audio in 7 Indian languages. Clusters related stories together, simplifies the language for listening, and reads it aloud — built for people who cannot read. Powered by Qwen3-8B (via Groq) and IndicF5 TTS (AI4Bharat).

---

## Known risks and mitigations

| Risk | Mitigation |
|---|---|
| IndicF5 fails to install on HF Spaces | gTTS fallback is automatic — app still works |
| Space sleeps, cache is lost | Scheduler fires on startup, rebuilds cache within 30s of wakeup |
| Groq rate limit hit during pre-gen | Skip pre-gen, generate on demand with retry backoff |
| trafilatura can't extract article text | Fall back to `description` field from Currents API |
| BERTopic produces poor clusters | Fall back to category-based grouping from Currents API metadata |
| IndicF5 reference audio missing | Bundle 2s reference .wav files for each language in `assets/` |
