from pipeline.llm import _clean_qwen_output, chat_complete


def test_clean_qwen_output_removes_thinking_block():
    assert _clean_qwen_output("<think>hidden</think>\nVisible") == "Visible"


def test_chat_complete_success(monkeypatch):
    class MockChoice:
        class MockMessage:
            content = "Hello world"
        message = MockMessage()

    class MockOutput:
        choices = [MockChoice()]

    class MockClient:
        def chat_completion(self, messages, model, max_tokens, temperature):
            return MockOutput()

    monkeypatch.setenv("HF_TOKEN", "mock-token")
    monkeypatch.setattr("pipeline.llm._get_client", lambda: MockClient())

    res = chat_complete([{"role": "user", "content": "hi"}])
    assert res == "Hello world"

