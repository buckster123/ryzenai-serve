"""OpenAI-compatible FastAPI app — chat + embeddings."""

from __future__ import annotations

import json
import time
import uuid
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ryzenai_serve import __version__
from ryzenai_serve.engine import GenerationConfig, NPUEngine
from ryzenai_serve.embedder import EmbeddingEngine


# ---------- Pydantic models ----------

class ChatMessage(BaseModel):
    role: str
    content: str


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


# ---------- App factory ----------

def create_app(
    engine: Optional[NPUEngine] = None,
    embedder: Optional[EmbeddingEngine] = None,
) -> FastAPI:
    if engine is None and embedder is None:
        raise ValueError("create_app needs at least one of engine/embedder")

    app = FastAPI(title="ryzenai-serve", version=__version__)

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

    @app.get("/v1/models")
    def list_models():
        items = []
        now = int(time.time())
        if engine:
            items.append({"id": engine.model_id, "object": "model",
                          "created": now, "owned_by": "ryzenai-serve",
                          "capabilities": ["chat.completions"]})
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
        prompt = engine.render_chat([m.model_dump() for m in req.messages])

        gc = GenerationConfig(
            max_tokens=req.max_tokens or 512,
            temperature=req.temperature if req.temperature is not None else 0.7,
            top_p=req.top_p if req.top_p is not None else 0.95,
            top_k=req.top_k if req.top_k is not None else 50,
            repetition_penalty=req.repetition_penalty or 1.0,
            do_sample=(req.temperature or 0.7) > 0,
        )

        if req.stream:
            return StreamingResponse(
                _stream_chat(engine, prompt, gc, req.stop or [], req.model or engine.model_id),
                media_type="text/event-stream",
            )

        text_parts: list[str] = []
        stop_hit = False
        for piece in engine.stream(prompt, gc):
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
                 stop: List[str], model_id: str):
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
        for piece in engine.stream(prompt, gc):
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
