"""Currents API fetching and full-article extraction."""

from __future__ import annotations

import os
import logging
import time
from typing import Any

from pipeline.config import get_int_env


BASE_URL = "https://api.currentsapi.services/v1/latest-news"
CURRENTS_MAX_PAGE_SIZE = 50
DEFAULT_MAX_ARTICLES = 5
logger = logging.getLogger(__name__)


def _fallback_articles() -> list[dict[str, Any]]:
    return [
        {
            "url": "demo://public-health",
            "title": "Health teams expand free eye checkups for older adults",
            "description": (
                "Local clinics are opening weekend camps so elderly residents can "
                "get basic eye tests and referrals without reading long forms."
            ),
            "image": "",
            "published": "",
            "category": ["health"],
        },
        {
            "url": "demo://weather-alerts",
            "title": "Weather office warns of heavy rain in several districts",
            "description": (
                "Officials asked families to avoid flooded roads and keep phones "
                "charged as showers continue through the evening."
            ),
            "image": "",
            "published": "",
            "category": ["weather"],
        },
        {
            "url": "demo://pension-help",
            "title": "New help desks open for pension application support",
            "description": (
                "Volunteers will help senior citizens check documents, update phone "
                "numbers, and understand application status in plain language."
            ),
            "image": "",
            "published": "",
            "category": ["community"],
        },
    ]


def fetch_articles(language: str = "en", max_results: int | None = None) -> list[dict[str, Any]]:
    max_results = max_results or get_int_env("MAX_ARTICLES", DEFAULT_MAX_ARTICLES)
    page_size = min(max(1, max_results), CURRENTS_MAX_PAGE_SIZE)
    api_key = os.environ.get("CURRENTS_API_KEY", "").strip()
    if not api_key:
        return _fallback_articles()

    import requests

    for attempt in range(3):
        try:
            resp = requests.get(
                BASE_URL,
                params={
                    "language": language,
                    "page_size": page_size,
                    "page_number": 1,
                },
                headers={"Authorization": api_key},
                timeout=15,
            )
        except requests.RequestException as exc:
            if attempt < 2:
                time.sleep(2**attempt)
                continue
            logger.warning("Currents API unavailable; using demo stories: %s", exc)
            return _fallback_articles()

        if resp.ok:
            news = resp.json().get("news", [])
            articles = [
                _normalise_article(article)
                for article in news
                if _is_valid_article(article)
            ]
            return articles or _fallback_articles()

        try:
            detail = resp.json()
        except ValueError:
            detail = resp.text[:300]

        if resp.status_code in {401, 403}:
            raise RuntimeError(
                f"Currents API authentication failed with HTTP {resp.status_code}. "
                "Check CURRENTS_API_KEY."
            )

        if _is_transient_error(resp.status_code, detail) and attempt < 2:
            time.sleep(2**attempt)
            continue

        logger.warning(
            "Currents API returned HTTP %s; using demo stories: %s",
            resp.status_code,
            detail,
        )
        return _fallback_articles()

    return _fallback_articles()


def _is_transient_error(status_code: int, detail: Any) -> bool:
    detail_text = str(detail).lower()
    return status_code in {429, 500, 502, 503, 504} or "database error" in detail_text


def _is_valid_article(article: dict[str, Any]) -> bool:
    return bool(article.get("url") and article.get("title"))


def _normalise_article(article: dict[str, Any]) -> dict[str, Any]:
    category = article.get("category", [])
    if isinstance(category, str):
        category = [category]

    return {
        "url": article["url"],
        "title": article["title"],
        "description": article.get("description") or "",
        "image": article.get("image") or "",
        "published": article.get("published") or "",
        "category": category if isinstance(category, list) else [],
    }


def fetch_full_text(url: str, fallback: str = "") -> str:
    if url.startswith("demo://"):
        return fallback

    try:
        import trafilatura

        downloaded = trafilatura.fetch_url(url)
        extracted = trafilatura.extract(downloaded) if downloaded else None
    except Exception:
        extracted = None

    text = (extracted or "").strip()
    return text or fallback
