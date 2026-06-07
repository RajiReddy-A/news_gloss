"""Story clustering with BERTopic and fallback grouping."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


_embed_model = None


def _load_embedder():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def cluster_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not articles:
        return []
    if len(articles) < 5:
        return _fallback_clusters(articles)

    try:
        from bertopic import BERTopic

        embed_model = _load_embedder()
        texts = [_article_text(article) for article in articles]
        embeddings = embed_model.encode(texts, show_progress_bar=False)
        topic_model = BERTopic(
            embedding_model=embed_model,
            min_topic_size=2,
            nr_topics="auto",
            verbose=False,
        )
        topics, _ = topic_model.fit_transform(texts, embeddings)
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for idx, topic_id in enumerate(topics):
            if topic_id != -1:
                grouped[int(topic_id)].append(articles[idx])

        clusters = []
        topic_info = topic_model.get_topic_info()
        for _, row in topic_info.iterrows():
            topic_id = int(row["Topic"])
            if topic_id == -1 or topic_id not in grouped:
                continue
            label = str(row["Name"]).replace("_", " ").title()
            clusters.append(_cluster(label, grouped[topic_id]))

        return sorted(clusters, key=lambda c: c["count"], reverse=True) or _fallback_clusters(articles)
    except Exception:
        return _fallback_clusters(articles)


def _article_text(article: dict[str, Any]) -> str:
    return f"{article.get('title', '')}. {article.get('description', '')}".strip()


def _cluster(label: str, articles: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": articles[0].get("url", label),
        "label": label or "News",
        "articles": articles,
        "count": len(articles),
    }


def _fallback_clusters(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for article in articles:
        categories = article.get("category") or ["News"]
        category = categories[0] if isinstance(categories, list) and categories else "News"
        grouped[str(category).title()].append(article)

    if len(grouped) == 1 and len(articles) > 1:
        return [_cluster("Latest News", articles)]

    return sorted(
        [_cluster(label, grouped_articles) for label, grouped_articles in grouped.items()],
        key=lambda c: c["count"],
        reverse=True,
    )
