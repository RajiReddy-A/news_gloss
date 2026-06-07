from pathlib import Path

from cache import store


def test_cache_key_is_stable():
    assert store.cache_key("https://example.com/a", "hi") == store.cache_key(
        "https://example.com/a", "hi"
    )
    assert store.cache_key("https://example.com/a", "hi").endswith("_hi")


def test_text_memory_and_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(store, "AUDIO_DIR", tmp_path / "audio")
    store.AUDIO_DIR.mkdir()
    store.clear_memory_cache()

    store.save_summary("https://example.com/a", "hello")
    assert store.get_summary("https://example.com/a") == "hello"

    store.clear_memory_cache()
    assert store.get_summary("https://example.com/a") == "hello"


def test_audio_path_uses_audio_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "AUDIO_DIR", tmp_path / "audio")
    path = store.audio_path("https://example.com/a", "mr")
    assert isinstance(path, Path)
    assert path.parent == tmp_path / "audio"
    assert path.name.endswith("_mr.mp3")
