from pipeline import translator


def test_translate_demo_without_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    out = translator.translate("A short update.", "hi")
    assert "Hindi demo translation" in out
    assert "A short update." in out
