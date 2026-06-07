from pipeline import fetcher


def test_normalise_article_handles_string_category():
    article = fetcher._normalise_article(
        {
            "url": "https://example.com",
            "title": "Title",
            "description": None,
            "category": "world",
        }
    )
    assert article["description"] == ""
    assert article["category"] == ["world"]


def test_fetch_full_text_falls_back_for_demo_url():
    assert fetcher.fetch_full_text("demo://story", fallback="fallback text") == "fallback text"


def test_currents_free_tier_page_size_is_capped(monkeypatch):
    captured = {}

    class Response:
        ok = True

        @staticmethod
        def json():
            return {"news": []}

    class Requests:
        @staticmethod
        def get(url, params, headers, timeout):
            captured["page_size"] = params["page_size"]
            return Response()

    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")
    monkeypatch.setitem(__import__("sys").modules, "requests", Requests)

    fetcher.fetch_articles(max_results=100)

    assert captured["page_size"] == 50


def test_transient_database_error_uses_demo_stories(monkeypatch):
    class Response:
        ok = False
        status_code = 400
        text = ""

        @staticmethod
        def json():
            return {"status": "400", "msg": "Database error occurred"}

    class Requests:
        class RequestException(Exception):
            pass

        @staticmethod
        def get(url, params, headers, timeout):
            return Response()

    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")
    monkeypatch.setitem(__import__("sys").modules, "requests", Requests)
    monkeypatch.setattr(fetcher.time, "sleep", lambda seconds: None)

    articles = fetcher.fetch_articles(max_results=5)

    assert len(articles) == 3
    assert all(article["url"].startswith("demo://") for article in articles)
