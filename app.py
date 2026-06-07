"""Gradio app for multilingual spoken news bulletins (Individual Articles flow)."""

from __future__ import annotations

import logging
from pathlib import Path
import os
import gradio as gr

from cache.store import (
    audio_path,
    get_articles,
    get_summary,
    get_translation,
    save_summary,
    save_translation,
    AUDIO_DIR,
)
from pipeline.config import LANG_NAMES, get_supported_langs
from pipeline.fetcher import fetch_full_text
from pipeline.summariser import summarise
from pipeline.translator import translate
from pipeline.tts import synthesise, warmup
from pipeline.llm import is_configured, MODEL

import scheduler


logging.basicConfig(level=logging.INFO)

LANG_CHOICES = [(LANG_NAMES[code], code) for code in get_supported_langs()]
DEFAULT_LANG = LANG_CHOICES[0][1] if LANG_CHOICES else "hi"


def _get_pipeline_info_html(
    translation_engine: str = "Pending Generation",
    voice_engine: str = "Pending Generation",
    is_cached: bool = False
) -> str:
    cached_badge = (
        '<span style="background: rgba(167, 139, 250, 0.2); color: #c084fc; '
        'padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; '
        'border: 1px solid rgba(167, 139, 250, 0.4);">Cached</span>'
        if is_cached else ""
    )
    return f"""
    <div style="background: rgba(22, 17, 43, 0.55); border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px; padding: 16px; margin-top: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(139, 92, 246, 0.2); padding-bottom: 8px;">
            <span style="font-weight: 600; font-size: 1.05rem; color: #f1ecff; display: flex; align-items: center; gap: 6px;">
                ℹ️ Pipeline Engine Info
            </span>
            {cached_badge}
        </div>
        <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.9rem;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #a78bfa;">Translation Engine:</span>
                <span style="color: #ffffff; font-weight: 500;">{translation_engine}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: #a78bfa;">Voice Synthesis:</span>
                <span style="color: #ffffff; font-weight: 500;">{voice_engine}</span>
            </div>
        </div>
    </div>
    """


def _article_rows() -> list[list[str]]:
    articles = get_articles()
    if not articles:
        return [["Loading news...", "Please wait for the first refresh"]]

    rows = []
    for article in articles:
        rows.append(
            [
                str(article.get("title", "Untitled")),
                str(article.get("published", "")),
            ]
        )
    return rows


def handle_article_select(evt: gr.SelectData, lang_code: str) -> tuple[str, str, str, str, str, str, str]:
    index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    articles = get_articles()
    if not isinstance(index, int) or index < 0 or index >= len(articles):
        return "", "Select an article from the list.", "", "", "", "No article selected.", _get_pipeline_info_html()

    article = articles[index]
    url = str(article.get("url", ""))
    title = str(article.get("title", "Untitled"))
    description = str(article.get("description", ""))

    details = f"📰 {title}\n\n{description}"
    fallback = f"{title}. {description}".strip()

    # Query cache for English summary and translation
    summary = get_summary(url) or ""
    translated = get_translation(url, lang_code) or ""

    lang_name = LANG_NAMES.get(lang_code, lang_code)
    if translated:
        status = f"Selected article. Cached {lang_name} bulletin loaded. Click Play to listen."
    elif summary:
        status = f"Selected article. English summary cached. Translation to {lang_name} will be generated on Play."
    else:
        status = f"Selected article. Spoken bulletin will be generated on Play."

    # Determine translation engine
    if translated:
        if translated.strip().startswith("[") and "demo translation" in translated:
            trans_engine = "Offline Demo Mode"
        else:
            trans_engine = f"Hugging Face ({MODEL})"
    else:
        trans_engine = "Pending Generation"

    # Determine voice engine
    target_audio = audio_path(url, lang_code)
    if target_audio.exists() and translated:
        # Check system capabilities to guess the method used
        from pipeline.tts import _load_indicf5
        ref_audio = Path("assets") / f"ref_{lang_code}.wav"
        if _load_indicf5() is not None and ref_audio.exists():
            voice_engine = "Cached (IndicF5)"
        else:
            voice_engine = "Cached (gTTS Web)"
        is_cached_audio = True
    else:
        voice_engine = "Pending Generation"
        is_cached_audio = False

    info_html = _get_pipeline_info_html(trans_engine, voice_engine, is_cached=is_cached_audio)

    return url, details, fallback, summary, translated, status, info_html


def handle_lang_change(rep_url: str, lang_code: str) -> tuple[str, str, str]:
    if not rep_url:
        info_html = _get_pipeline_info_html()
        return "", "Select an article from the list.", info_html

    lang_name = LANG_NAMES.get(lang_code, lang_code)
    translated = get_translation(rep_url, lang_code)
    
    if translated:
        if translated.strip().startswith("[") and "demo translation" in translated:
            trans_engine = "Offline Demo Mode"
        else:
            trans_engine = f"Hugging Face ({MODEL})"
            
        target_audio = audio_path(rep_url, lang_code)
        if target_audio.exists():
            from pipeline.tts import _load_indicf5
            ref_audio = Path("assets") / f"ref_{lang_code}.wav"
            if _load_indicf5() is not None and ref_audio.exists():
                voice_engine = "Cached (IndicF5)"
            else:
                voice_engine = "Cached (gTTS Web)"
            is_cached_audio = True
        else:
            voice_engine = "Pending Generation"
            is_cached_audio = False
            
        info_html = _get_pipeline_info_html(trans_engine, voice_engine, is_cached=is_cached_audio)
        return translated, f"Cached {lang_name} bulletin loaded. Click Play to listen.", info_html

    info_html = _get_pipeline_info_html("Pending Generation", "Pending Generation", is_cached=False)
    return "", f"Bulletin will be translated to {lang_name} when you click Play.", info_html


def handle_play(rep_url: str, fallback_text: str, lang_code: str):
    if not rep_url:
        yield None, "Please select an article first.", "", "", _get_pipeline_info_html()
        return

    target_audio = audio_path(rep_url, lang_code)
    lang_name = LANG_NAMES.get(lang_code, lang_code)
    
    # Check if already cached
    summary = get_summary(rep_url)
    translated = get_translation(rep_url, lang_code)

    if target_audio.exists() and summary and translated:
        if translated.strip().startswith("[") and "demo translation" in translated:
            trans_engine = "Offline Demo Mode"
        else:
            trans_engine = f"Hugging Face ({MODEL})"
            
        from pipeline.tts import _load_indicf5
        ref_audio = Path("assets") / f"ref_{lang_code}.wav"
        if _load_indicf5() is not None and ref_audio.exists():
            voice_engine = "Cached (IndicF5)"
        else:
            voice_engine = "Cached (gTTS Web)"
            
        info_html = _get_pipeline_info_html(trans_engine, voice_engine, is_cached=True)
        yield str(target_audio), f"Playing cached {lang_name} bulletin.", summary, translated, info_html
        return

    if is_configured():
        trans_engine = f"Hugging Face ({MODEL})"
    else:
        trans_engine = "Offline Demo Mode"

    try:
        info_html = _get_pipeline_info_html(trans_engine, "Pending Generation", is_cached=False)
        yield None, "Fetching article content...", "", "", info_html
        article_text = fetch_full_text(rep_url, fallback=fallback_text)

        if not summary:
            yield None, "Simplifying the story...", "", "", info_html
            summary = summarise(article_text)
            save_summary(rep_url, summary)

        if not translated:
            yield None, f"Translating to {lang_name}...", summary, "", info_html
            translated = translate(summary, lang_code)
            save_translation(rep_url, lang_code, translated)

        yield None, "Generating spoken audio...", summary, translated, info_html
        audio_file, tts_method = synthesise(translated, lang_code, target_audio)
        
        info_html = _get_pipeline_info_html(trans_engine, tts_method, is_cached=False)
        yield str(audio_file), f"Ready in {lang_name}.", summary, translated, info_html
    except Exception as exc:
        logging.exception("Failed to play audio bulletin")
        info_html = _get_pipeline_info_html(trans_engine, "Failed", is_cached=False)
        yield None, f"Error: {exc}", summary or "", translated or "", info_html


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

body, .gradio-container {
    background: radial-gradient(circle at 50% 0%, #17112a 0%, #090615 100%) !important;
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e2e0ff !important;
}

/* Premium panel containers styling */
.gradio-container .gr-panel, 
.gradio-container .gr-box,
.gradio-container .block,
.gradio-container .gr-card {
    background: rgba(22, 17, 43, 0.55) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4) !important;
}

/* Titles and labels styling */
h1, h2, h3, .gr-form label {
    color: #f1ecff !important;
    font-weight: 600 !important;
}

#title h1 {
    font-size: 2.8rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg, #a78bfa 0%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

/* Highlighted top bar for language selector */
#top-bar {
    margin: 1rem auto 2rem auto !important;
    max-width: 900px !important;
    padding: 0.5rem !important;
}

#lang-selector {
    background: rgba(26, 20, 50, 0.75) !important;
    border: 1px solid rgba(139, 92, 246, 0.45) !important;
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.2) !important;
    border-radius: 12px !important;
}

/* Action button premium styling */
button.primary, .large-button button {
    background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.45) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

button.primary:hover, .large-button button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.6) !important;
}

button:not(.primary) {
    background: rgba(139, 92, 246, 0.12) !important;
    border: 1px solid rgba(139, 92, 246, 0.35) !important;
    color: #e2d3ff !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

button:not(.primary):hover {
    background: rgba(139, 92, 246, 0.22) !important;
    border-color: rgba(139, 92, 246, 0.6) !important;
    color: #ffffff !important;
}

/* Inputs and textareas */
textarea, input[type="text"] {
    background: rgba(10, 7, 22, 0.65) !important;
    color: #f3f0ff !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 3px rgba(167, 139, 250, 0.25) !important;
}

/* Row selection effect on table */
.gr-table tr {
    cursor: pointer !important;
    transition: background 0.2s ease !important;
}

.gr-table tr:hover {
    background: rgba(139, 92, 246, 0.18) !important;
}

.gr-table th {
    background: transparent !important;
    color: #c084fc !important;
    font-weight: 600 !important;
    border-bottom: 2px solid rgba(139, 92, 246, 0.3) !important;
}
"""

with gr.Blocks(title="News in Your Language") as demo:
    gr.Markdown(
        "# News in Your Language\n"
        "Select your preferred language, choose a news article from the list, and listen to a spoken bulletin.",
        elem_id="title",
    )

    selected_url = gr.State("")
    fallback_text = gr.State("")

    # Top level target language selector
    with gr.Row(elem_id="top-bar"):
        with gr.Column():
            lang_picker = gr.Radio(
                choices=LANG_CHOICES,
                value=DEFAULT_LANG,
                label="Preferred Target Language",
                elem_id="lang-selector"
            )

    with gr.Row():
        with gr.Column(scale=5):
            article_list = gr.Dataframe(
                headers=["Headline", "Published"],
                datatype=["str", "str"],
                value=_article_rows,
                every=1800,
                interactive=False,
                wrap=True,
                label="Latest news articles",
            )

        with gr.Column(scale=4):
            details_box = gr.Textbox(
                label="Selected Article details",
                lines=6,
                interactive=False,
                placeholder="Click an article on the left to see details here..."
            )
            play_btn = gr.Button(
                "Play audio bulletin",
                variant="primary",
                size="lg",
                elem_classes=["large-button"],
            )
            status_box = gr.Textbox(label="Status", interactive=False, max_lines=2)
            audio_out = gr.Audio(label="Audio bulletin", type="filepath", autoplay=True)
            
            # Global Pipeline information box (ℹ️)
            pipeline_info = gr.HTML(
                value=_get_pipeline_info_html(),
                label="Pipeline Info"
            )

    with gr.Row():
        with gr.Column():
            english_summary_box = gr.Textbox(
                label="Simplified English bulletin (Read-Along)",
                lines=5,
                interactive=False,
                placeholder="The simplified English bulletin will be displayed here..."
            )
        with gr.Column():
            translated_text_box = gr.Textbox(
                label="Translated spoken bulletin (Read-Along)",
                lines=5,
                interactive=False,
                placeholder="The translated spoken bulletin will be displayed here..."
            )

    article_list.select(
        fn=handle_article_select,
        inputs=[lang_picker],
        outputs=[selected_url, details_box, fallback_text, english_summary_box, translated_text_box, status_box, pipeline_info],
    )
    lang_picker.change(
        fn=handle_lang_change,
        inputs=[selected_url, lang_picker],
        outputs=[translated_text_box, status_box, pipeline_info],
    )
    play_btn.click(
        fn=handle_play,
        inputs=[selected_url, fallback_text, lang_picker],
        outputs=[audio_out, status_box, english_summary_box, translated_text_box, pipeline_info],
    )


if __name__ == "__main__":
    warmup()
    demo.launch(css=CSS, allowed_paths=[str(AUDIO_DIR)])
