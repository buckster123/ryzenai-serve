"""Basic smoke tests — don't require an NPU model loaded."""

from ryzenai_serve import __version__


def test_version():
    # Just sanity — version string is set and parseable
    parts = __version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_imports():
    """The packages should import without needing onnxruntime_genai available."""
    from ryzenai_serve import engine, server  # noqa: F401
    from ryzenai_serve.server import ChatRequest, ChatMessage

    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
    assert req.messages[0].content == "hi"
    assert req.max_tokens == 512
