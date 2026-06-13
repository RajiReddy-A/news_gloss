"""Background refresh job for fetching, clustering, and text pre-generation."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from apscheduler.schedulers.background import BackgroundScheduler

from cache.store import (
    get_summary,
    get_translation,
    save_articles,
    save_summary,
    save_translation,
)
from pipeline.config import get_int_env, get_supported_langs
from pipeline.fetcher import fetch_articles, fetch_full_text
from pipeline.summariser import summarise
from pipeline.translator import translate


logger = logging.getLogger(__name__)

FETCH_INTERVAL = get_int_env("FETCH_INTERVAL_MIN", 30)
TOP_N = get_int_env("TOP_N_PREGEN", 10)
LANGS = get_supported_langs()

_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=1)
_scheduler: BackgroundScheduler | None = None


def refresh_news() -> list[dict]:
    if not _lock.acquire(blocking=False):
        logger.info("Refresh already running; skipping overlapping job")
        return []

    try:
        articles = fetch_articles(max_results=get_int_env("MAX_ARTICLES", 5))
        save_articles(articles)
        _pre_generate_text(articles[:TOP_N])
        return articles
    except Exception:
        logger.exception("News refresh failed")
        return []
    finally:
        _lock.release()


def _pre_generate_text(articles: list[dict]) -> None:
    for article in articles:
        url = article.get("url")
        if not url:
            continue

        summary = get_summary(url)
        if not summary:
            fallback = f"{article.get('title', '')}. {article.get('description', '')}".strip()
            text = fetch_full_text(url, fallback=fallback)
            summary = summarise(text)
            save_summary(url, summary)

        for lang in LANGS:
            if not get_translation(url, lang):
                translated = translate(summary, lang)
                save_translation(url, lang, translated)


def refresh_news_async() -> None:
    _executor.submit(refresh_news)


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if os.environ.get("DISABLE_SCHEDULER") == "1":
        return BackgroundScheduler()
    if _scheduler and _scheduler.running:
        return _scheduler

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        refresh_news_async,
        "cron",
        hour=6,
        minute=0,
        id="refresh_news_daily",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.start()
    refresh_news_async()
    return _scheduler


scheduler = start_scheduler()
