"""Basic smoke tests — don't require an NPU model loaded."""

from ryzenai_serve import __version__


def test_version():
    assert __version__ == "0.1.0"


def test_imports():
    """The packages should import without needing onnxruntime_genai available."""
    from ryzenai_serve import engine, server  # noqa: F401
    from ryzenai_serve.server import ChatRequest, ChatMessage

    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
    assert req.messages[0].content == "hi"
    assert req.max_tokens == 512
