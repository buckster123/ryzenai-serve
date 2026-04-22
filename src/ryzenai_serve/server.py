"""OpenAI-compatible FastAPI app — chat + embeddings + vision."""

from __future__ import annotations

import base64
import binascii
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, List, Literal, Optional, Union
from urllib import request as urllib_request

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ryzenai_serve import __version__
from ryzenai_serve.engine import GenerationConfig, NPUEngine
from ryzenai_serve.embedder import EmbeddingEngine


# ---------- Pydantic models ----------

class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ChatMessageContent = Union[str, List[Union[TextContent, ImageUrlContent]]]


class ChatMessage(BaseModel):
    role: str
    content: ChatMessageContent


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 50
    repetition_penalty: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"   # "float" only; "base64" not supported


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingsUsage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class EmbeddingsResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingsUsage


# ---------- Helpers ----------

def _extract_text(content: ChatMessageContent) -> str:
    """Flatten message content to plain text for token accounting."""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                parts.append("[image]")
        elif hasattr(part, "type"):
            if part.type == "text":
                parts.append(part.text)
            elif part.type == "image_url":
                parts.append("[image]")
    return "".join(parts)


def _extract_image_urls(content: ChatMessageContent) -> list[str]:
    """Pull image URLs out of multimodal message content."""
    urls: list[str] = []
    if isinstance(content, str):
        return urls
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "image_url":
                iu = part.get("image_url", {})
                url = iu.get("url", "") if isinstance(iu, dict) else getattr(iu, "url", "")
                if url:
                    urls.append(url)
        elif hasattr(part, "type") and part.type == "image_url":
            url = part.image_url.url if hasattr(part.image_url, "url") else part.image_url.get("url", "")
            if url:
                urls.append(url)
    return urls


def _fetch_image_to_temp(url: str) -> str:
    """Download or decode an image and save to a temp file.

    Supports:
      - data:image/png;base64,...  (and jpeg, webp, gif)
      - http://... / https://...
    Returns the temp file path. Caller should unlink when done.
    """
    if url.startswith("data:"):
        # data URI
        try:
            header, b64 = url.split(",", 1)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Malformed data URI")
        # Extract extension from mime type
        mime = header.split(";")[0].split(":")[1]
        ext = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }.get(mime, ".bin")
        try:
            data = base64.b64decode(b64)
        except binascii.Error:
            raise HTTPException(status_code=400, detail="Invalid base64 in data URI")
        fd, path = tempfile.mkstemp(suffix=ext)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        return path

    if url.startswith("http://") or url.startswith("https://"):
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        try:
            urllib_request.urlretrieve(url, path)
        except Exception as e:
            os.unlink(path)
            raise HTTPException(status_code=502, detail=f"Failed to download image: {e}")
        # Try to fix extension by sniffing magic bytes
        with open(path, "rb") as f:
            magic = f.read(8)
        ext = ".bin"
        if magic.startswith(b"\x89PNG"):
            ext = ".png"
        elif magic.startswith(b"\xff\xd8"):
            ext = ".jpg"
        elif magic.startswith(b"RIFF") and magic[8:12] == b"WEBP":
            ext = ".webp"
        elif magic.startswith(b"GIF87a") or magic.startswith(b"GIF89a"):
            ext = ".gif"
        if ext != ".bin":
            new_path = path.replace(".bin", ext)
            os.rename(path, new_path)
            path = new_path
        return path

    # Local file path (relative or absolute)
    p = Path(url).expanduser().resolve()
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"Image not found: {url}")
    return str(p)


def _transform_messages_for_template(messages: List[dict]) -> List[dict]:
    """Replace OpenAI image_url content parts with template-friendly image parts."""
    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            out.append(m)
            continue
        # content is a list of parts
        new_parts: list[dict] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "image_url":
                    new_parts.append({"type": "image"})
                else:
                    new_parts.append(part)
            elif hasattr(part, "type"):
                if part.type == "image_url":
                    new_parts.append({"type": "image"})
                else:
                    new_parts.append(part.model_dump() if hasattr(part, "model_dump") else dict(part))
            else:
                new_parts.append(part)
        out.append({**m, "content": new_parts})
    return out


# ---------- App factory ----------

def create_app(
    engine: Optional[NPUEngine] = None,
    embedder: Optional[EmbeddingEngine] = None,
) -> FastAPI:
    if engine is None and embedder is None:
        raise ValueError("create_app needs at least one of engine/embedder")

    app = FastAPI(title="ryzenai-serve", version=__version__)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {
            "server": "ryzenai-serve",
            "version": __version__,
            "chat_model": engine.model_id if engine else None,
            "embedding_model": embedder.model_id if embedder else None,
            "context_length": engine.context_length if engine else None,
            "embedding_dim": embedder.dim if embedder else None,
        }

    _CHAT_HTML = Path(__file__).with_name("static") / "chat.html"

    @app.get("/chat")
    def chat_ui():
        if _CHAT_HTML.exists():
            return FileResponse(_CHAT_HTML)
        raise HTTPException(status_code=404, detail="chat.html not found")

    @app.get("/v1/models")
    def list_models():
        items = []
        now = int(time.time())
        if engine:
            caps = ["chat.completions"]
            if engine.is_vlm:
                caps.append("vision")
            items.append({"id": engine.model_id, "object": "model",
                          "created": now, "owned_by": "ryzenai-serve",
                          "capabilities": caps})
        if embedder:
            items.append({"id": embedder.model_id, "object": "model",
                          "created": now, "owned_by": "ryzenai-serve",
                          "capabilities": ["embeddings"],
                          "dim": embedder.dim})
        return {"object": "list", "data": items}

    @app.get("/stats")
    def stats():
        out = {}
        if engine:
            s = engine.stats
            out["chat"] = {
                "model": s.model_id,
                "model_path": s.model_path,
                "context_length": s.context_length,
                "init_seconds": s.init_seconds,
                "requests": s.requests,
                "prompt_tokens": s.prompt_tokens,
                "completion_tokens": s.completion_tokens,
            }
        if embedder:
            es = embedder.stats
            out["embeddings"] = {
                "model": es.model_id,
                "model_path": es.model_path,
                "dim": es.dim,
                "init_seconds": es.init_seconds,
                "requests": es.requests,
                "input_tokens": es.input_tokens,
            }
        return out

    # ---------- Chat ----------

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        if engine is None:
            raise HTTPException(status_code=501, detail="No chat model loaded on this server")

        # Extract images from all messages
        image_urls: list[str] = []
        for m in req.messages:
            image_urls.extend(_extract_image_urls(m.content))

        # If images are present, model must be VLM-capable
        if image_urls and not engine.is_vlm:
            raise HTTPException(
                status_code=400,
                detail=f"Model {engine.model_id} is not vision-capable. "
                       f"Load a multimodal model (e.g. gemma-3-4b-it-mm) to use images."
            )

        # Fetch images to temp files
        temp_paths: list[str] = []
        if image_urls:
            for url in image_urls:
                temp_paths.append(_fetch_image_to_temp(url))

        try:
            # Render prompt: transform image_url parts -> image parts for template
            raw_messages = [m.model_dump() for m in req.messages]
            template_messages = _transform_messages_for_template(raw_messages)
            prompt = engine.render_chat(template_messages)

            gc = GenerationConfig(
                max_tokens=req.max_tokens or 512,
                temperature=req.temperature if req.temperature is not None else 0.7,
                top_p=req.top_p if req.top_p is not None else 0.95,
                top_k=req.top_k if req.top_k is not None else 50,
                repetition_penalty=req.repetition_penalty or 1.0,
                do_sample=(req.temperature or 0.7) > 0,
            )

            # Pass temp paths to engine; engine loads og.Images from them
            images_arg = temp_paths if temp_paths else None

            if req.stream:
                return StreamingResponse(
                    _stream_chat(engine, prompt, gc, req.stop or [], req.model or engine.model_id, images_arg),
                    media_type="text/event-stream",
                )

            text_parts: list[str] = []
            stop_hit = False
            for piece in engine.stream(prompt, gc, images=images_arg):
                text_parts.append(piece)
                full = "".join(text_parts)
                if req.stop:
                    for s in req.stop:
                        if s and s in full:
                            full = full.split(s)[0]
                            text_parts = [full]
                            stop_hit = True
                            break
                if stop_hit:
                    break

            text = "".join(text_parts)
            prompt_text_for_toks = _extract_text(req.messages[0].content) if req.messages else ""
            prompt_tokens = len(engine.tokenizer.encode(prompt))
            completion_tokens = len(engine.tokenizer.encode(text))

            return ChatResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=engine.model_id,
                choices=[Choice(message=ChoiceMessage(content=text),
                                finish_reason="stop" if stop_hit else "length")],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )
        finally:
            # Clean up temp files
            for p in temp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    # ---------- Embeddings ----------

    @app.post("/v1/embeddings")
    def embeddings(req: EmbeddingsRequest):
        if embedder is None:
            raise HTTPException(status_code=501, detail="No embedding model loaded on this server")
        if req.encoding_format not in (None, "float"):
            raise HTTPException(status_code=400, detail=f"encoding_format={req.encoding_format!r} not supported; use 'float'")

        inputs = [req.input] if isinstance(req.input, str) else list(req.input)
        if not inputs:
            raise HTTPException(status_code=400, detail="'input' is empty")

        vecs = embedder.embed(inputs)
        data = [
            EmbeddingData(index=i, embedding=v.tolist())
            for i, v in enumerate(vecs)
        ]
        # rough token accounting
        tok_count = sum(len(embedder.tokenizer.encode(t)) for t in inputs)
        return EmbeddingsResponse(
            data=data,
            model=embedder.model_id,
            usage=EmbeddingsUsage(prompt_tokens=tok_count, total_tokens=tok_count),
        )

    return app


def _stream_chat(engine: NPUEngine, prompt: str, gc: GenerationConfig,
                 stop: List[str], model_id: str, images: Optional[list[str]] = None):
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def frame(delta: dict, finish_reason: Optional[str] = None):
        obj = {
            "id": cmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        }
        return f"data: {json.dumps(obj)}\n\n"

    yield frame({"role": "assistant"})

    buf = ""
    stop_hit = False
    try:
        for piece in engine.stream(prompt, gc, images=images):
            buf += piece
            if stop:
                for s in stop:
                    if s and s in buf:
                        tail = buf.split(s)[0]
                        new = tail[len(buf) - len(piece):]
                        if new:
                            yield frame({"content": new})
                        stop_hit = True
                        break
                if stop_hit:
                    break
            yield frame({"content": piece})
    finally:
        yield frame({}, finish_reason="stop" if stop_hit else "length")
        yield "data: [DONE]\n\n"
