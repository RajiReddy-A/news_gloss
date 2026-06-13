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
DEFAULT_LANG = "hi"


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
    <div style="background: #0f172a; border: 1px solid rgba(139, 92, 246, 0.25); border-radius: 12px; padding: 16px; margin-top: 12px; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);">
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
        return [["<div style='padding:16px;'>Loading news... Please wait for the first refresh</div>"]]

    rows = []
    for article in articles:
        title = str(article.get("title", "Untitled"))
        published = str(article.get("published", ""))
        image = str(article.get("image", ""))
        category = article.get("category", [])
        cat_str = category[0].upper() if category else "NEWS"
        
        img_style = f"background-image: url('{image}');" if image else "background: rgba(255,255,255,0.05);"
        
        html = f"""
        <div class="news-list-card">
            <div class="news-list-image" style="{img_style}"></div>
            <div class="news-list-content">
                <div class="news-meta"><span style="color:#38bdf8;font-weight:600;font-size:0.75rem;">{cat_str}</span> <span style="opacity:0.6;font-size:0.75rem;">• {published[:10] if published else ''}</span></div>
                <h4 style="margin: 4px 0 0 0; font-size: 0.95rem; line-height: 1.3; font-weight: 500;">{title}</h4>
            </div>
        </div>
        """
        rows.append([html])
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
    image = str(article.get("image", ""))

    img_tag = f'<img src="{image}" style="width:100%; height:240px; object-fit:cover; border-radius:12px; margin-bottom:16px;" />' if image else ""
    details = f"""
    <div>
        {img_tag}
        <h2 style="margin:0 0 8px 0; font-size:1.6rem; line-height:1.2;">{title}</h2>
        <p style="opacity:0.8; font-size:0.95rem;">{description}</p>
    </div>
    """
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
        voice_engine = "Cached (gTTS)"
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
            voice_engine = "Cached (gTTS)"
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
            
        voice_engine = "Cached (gTTS)"
            
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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

body, .gradio-container {
    background: radial-gradient(circle at 50% 0%, #0f172a 0%, #020617 100%) !important;
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #e2e0ff !important;
}

/* Premium panel containers styling (Glassmorphism) */
.gradio-container .gr-panel, 
.gradio-container .gr-box,
.gradio-container .block,
.gradio-container .gr-card,
#player-card {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
}

#player-card {
    padding: 24px !important;
}

/* Titles and labels styling */
h1, h2, h3, .gr-form label {
    color: #f8fafc !important;
    font-weight: 600 !important;
}

#title h1 {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: left;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    letter-spacing: -0.02em;
}

/* Header bar */
#header-bar {
    margin: 1rem auto 2rem auto !important;
    align-items: center !important;
}

#lang-selector {
    background: rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 12px !important;
}

/* Action button premium styling (Cyan Accent) */
button.primary, .play-button {
    background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(6, 182, 212, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}

button.primary:hover, .play-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(6, 182, 212, 0.6) !important;
}

/* Inputs and textareas */
textarea, input[type="text"] {
    background: #0f172a !important;
    color: #f1f5f9 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    font-family: 'Outfit', sans-serif !important;
}

textarea:focus, input[type="text"]:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25) !important;
}

/* List Items / Dataframe styling */
.gr-table {
    border: none !important;
}

.gr-table th {
    display: none !important; /* Hide spreadsheet headers */
}

.gr-table tr {
    cursor: pointer !important;
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    margin-bottom: 8px !important;
    display: block !important;
    transition: all 0.2s ease !important;
}

.gr-table td {
    border: none !important;
    color: #cbd5e1 !important;
}

.gr-table tr:hover {
    background: rgba(56, 189, 248, 0.1) !important;
    border-color: rgba(56, 189, 248, 0.3) !important;
    transform: translateX(4px) !important;
}

/* Tabs */
.gr-tabs > button.selected {
    border-bottom: 2px solid #38bdf8 !important;
    color: #38bdf8 !important;
    background: transparent !important;
}
.gr-tabs > button {
    border: none !important;
    background: transparent !important;
    color: #64748b !important;
    font-weight: 600 !important;
}

/* Custom Visual News Cards */
.news-list-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px;
}
.news-list-image {
    width: 80px;
    height: 80px;
    border-radius: 8px;
    background-size: cover;
    background-position: center;
    flex-shrink: 0;
}
.news-list-content {
    display: flex;
    flex-direction: column;
    justify-content: center;
}
"""

with gr.Blocks(title="News in Your Language") as demo:
    selected_url = gr.State("")
    fallback_text = gr.State("")

    # Top level header and language selector
    with gr.Row(elem_id="header-bar"):
        with gr.Column(scale=8):
            gr.Markdown("# News Gloss - News in Your Language", elem_id="title")
        with gr.Column(scale=2, min_width=150):
            lang_picker = gr.Radio(
                choices=LANG_CHOICES,
                value=DEFAULT_LANG,
                label="Language",
                elem_id="lang-selector",
                interactive=True,
            )

    # Main Bento Grid Layout
    with gr.Row():
        # Left Column: News Feed
        with gr.Column(scale=4, elem_id="left-column"):
            gr.Markdown("### LATEST HEADLINES")
            article_list = gr.Dataframe(
                headers=["Article"],
                datatype=["html"],
                value=_article_rows,
                every=1800,
                interactive=False,
                wrap=True,
            )

        # Right Column: Media Player Card
        with gr.Column(scale=6, elem_id="right-column"):
            with gr.Column(elem_id="player-card"):
                gr.Markdown("### FEATURED ARTICLE")
                details_box = gr.HTML(
                    value="<div style='padding:20px; opacity:0.5;'>Click an article on the left to view details here...</div>",
                    label="Selected Article details"
                )
                
                with gr.Row():
                    play_btn = gr.Button(
                        "▶ Play Audio",
                        variant="primary",
                        size="lg",
                        elem_classes=["play-button"],
                    )
                    status_box = gr.Textbox(label="", interactive=False, max_lines=1)

                audio_out = gr.Audio(label="", type="filepath", autoplay=True)
                
                with gr.Tabs():
                    with gr.TabItem("TRANSCRIPT"):
                        translated_text_box = gr.Textbox(
                            label="",
                            lines=4,
                            interactive=False,
                            placeholder="The translated spoken transcript will appear here..."
                        )
                    with gr.TabItem("SUMMARY"):
                        english_summary_box = gr.Textbox(
                            label="",
                            lines=4,
                            interactive=False,
                            placeholder="The simplified English summary will appear here..."
                        )
                    with gr.TabItem("INFO"):
                        pipeline_info = gr.HTML(
                            value=_get_pipeline_info_html(),
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
