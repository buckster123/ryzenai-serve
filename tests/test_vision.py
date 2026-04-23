"""Vision / multimodal unit tests — NO NPU model required.

Exercises the image parsing + resize + message transformation helpers in
isolation. End-to-end VLM inference is covered by the smoke script in
`scripts/vision_smoke.py` which requires a running :8002 Gemma endpoint.
"""

from __future__ import annotations

import base64
import io
import os

import pytest

from ryzenai_serve.server import (
    ChatMessage,
    ChatRequest,
    ImageUrl,
    ImageUrlContent,
    TextContent,
    _fetch_image_to_temp,
    _resize_image_if_needed,
    _transform_messages_for_template,
)


pytest.importorskip("PIL", reason="Pillow needed for vision tests")
from PIL import Image  # noqa: E402


# ---------- fixtures ----------

def _png_bytes(w: int, h: int, color=(255, 0, 0)) -> bytes:
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------- pydantic model parsing ----------

def test_chat_request_accepts_multimodal_content():
    """OpenAI-style multimodal content parts should parse cleanly."""
    req = ChatRequest(
        messages=[
            ChatMessage(
                role="user",
                content=[
                    TextContent(type="text", text="What is in this image?"),
                    ImageUrlContent(
                        type="image_url",
                        image_url=ImageUrl(url="data:image/png;base64,xxx"),
                    ),
                ],
            )
        ]
    )
    assert isinstance(req.messages[0].content, list)
    assert req.messages[0].content[0].text == "What is in this image?"
    assert req.messages[0].content[1].image_url.url.startswith("data:")


def test_chat_request_accepts_string_content():
    """Plain string content still works (backward compat)."""
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
    assert req.messages[0].content == "hi"


# ---------- _transform_messages_for_template ----------

def test_transform_replaces_image_url_with_image():
    """Jinja templates expect {type:'image'}, not OpenAI's image_url."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,xxx"}},
            ],
        }
    ]
    out = _transform_messages_for_template(messages)
    parts = out[0]["content"]
    assert parts[0] == {"type": "text", "text": "hello"}
    assert parts[1] == {"type": "image"}


def test_transform_passes_through_plain_string():
    messages = [{"role": "user", "content": "hello"}]
    out = _transform_messages_for_template(messages)
    assert out[0]["content"] == "hello"


# ---------- _resize_image_if_needed ----------

def test_resize_small_image_is_noop(tmp_path):
    """Images already under the cap should be returned unchanged."""
    p = tmp_path / "small.png"
    p.write_bytes(_png_bytes(256, 256))
    original_size = p.stat().st_size
    out = _resize_image_if_needed(str(p), max_dim=1024)
    assert out == str(p)
    assert p.stat().st_size == original_size
    with Image.open(p) as img:
        assert img.size == (256, 256)


def test_resize_large_image_shrinks(tmp_path):
    """Images over the cap should be shrunk preserving aspect ratio."""
    p = tmp_path / "big.png"
    p.write_bytes(_png_bytes(4096, 2048))
    out = _resize_image_if_needed(str(p), max_dim=1024)
    assert out == str(p)  # same path, overwritten in place
    with Image.open(p) as img:
        w, h = img.size
        assert max(w, h) == 1024
        # aspect ratio preserved (2:1)
        assert abs((w / h) - 2.0) < 0.01


def test_resize_bad_file_returns_path(tmp_path):
    """Non-image files should fall through silently."""
    p = tmp_path / "not-an-image.txt"
    p.write_text("this is not an image")
    out = _resize_image_if_needed(str(p), max_dim=1024)
    assert out == str(p)


# ---------- _fetch_image_to_temp ----------

def test_fetch_local_path_roundtrip(tmp_path):
    """Local file paths should be resolved to absolute existing paths."""
    p = tmp_path / "local.png"
    p.write_bytes(_png_bytes(128, 128))
    out = _fetch_image_to_temp(str(p))
    assert os.path.exists(out)
    with Image.open(out) as img:
        assert img.size == (128, 128)


def test_fetch_data_uri_png_decodes(tmp_path):
    """data:image/png;base64 should decode to a real temp file."""
    b = _png_bytes(64, 64, color=(0, 255, 0))
    uri = "data:image/png;base64," + base64.b64encode(b).decode()
    out = _fetch_image_to_temp(uri)
    try:
        assert os.path.exists(out)
        assert out.endswith(".png")
        with Image.open(out) as img:
            assert img.size == (64, 64)
            assert img.mode == "RGB"
    finally:
        if os.path.exists(out):
            os.unlink(out)


def test_fetch_missing_local_file_raises():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _fetch_image_to_temp("/does/not/exist/nope.png")
    assert exc.value.status_code == 400


def test_fetch_malformed_data_uri_raises():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _fetch_image_to_temp("data:this-is-garbage")
    assert exc.value.status_code == 400
