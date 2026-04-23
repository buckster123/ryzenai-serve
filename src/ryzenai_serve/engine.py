"""NPU engine wrapper — thin layer over onnxruntime-genai."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

# onnxruntime_genai is imported lazily so `ryzenai-serve --help` and tests
# that don't need the NPU can run without sourcing the Ryzen AI env.
og = None


def _lazy_import_og():
    global og
    if og is None:
        import onnxruntime_genai as _og
        og = _og
    return og


@dataclass
class GenerationConfig:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.0
    do_sample: bool = True
    # stop: list[str] supported at serve layer (detokenized)


@dataclass
class EngineStats:
    model_path: str
    model_id: str
    context_length: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    requests: int = 0
    init_seconds: float = 0.0


class NPUEngine:
    """Loads one NPU model at startup, serializes generation (NPU is single-session).

    Thread-safety: a single NPU model can only run one generation at a time
    without its own batching support, so we guard with a lock. Concurrent
    requests are serialized, not parallelized.
    """

    def __init__(self, model_dir: str, model_id: Optional[str] = None):
        og = _lazy_import_og()
        p = Path(model_dir).expanduser().resolve()
        if not (p / "genai_config.json").exists():
            raise FileNotFoundError(
                f"No genai_config.json in {p}. Expected an OGA-packaged model directory "
                f"(e.g. amd/<model>_rai_1.7.1_npu_*K)."
            )
        self.model_dir = str(p)
        self.model_id = model_id or p.name

        # Read context length for metadata
        with open(p / "genai_config.json") as f:
            cfg = json.load(f)
        self.context_length = int(
            cfg.get("model", {}).get("context_length")
            or cfg.get("search", {}).get("max_length", 4096)
        )

        # The NPU compile may have a smaller hard limit than the model's
        # theoretical context_length (e.g. Gemma-3-4b is compiled for 4K
        # despite config saying 16K). Read the actual KV-cache limit.
        decoder_opts = cfg.get("model", {}).get("decoder", {}).get("session_options", {})
        kv_limit = decoder_opts.get("max_lenght_for_kv_cache")  # AMD's typo, not ours
        self.max_sequence_length = int(kv_limit or cfg.get("search", {}).get("max_length", self.context_length))

        # Detect VLM (multimodal) models: genai_config has a "vision" section.
        # These models feed the decoder via `inputs_embeds` produced by a
        # separate embedding ONNX, so we must drive generation through
        # MultiModalProcessor (with images=None for text-only requests)
        # rather than tokenizer.encode + generator.append_tokens.
        self.is_vlm = "vision" in cfg.get("model", {})

        t0 = time.time()
        self.model = og.Model(self.model_dir)
        self.tokenizer = og.Tokenizer(self.model)
        self.tokenizer_stream = self.tokenizer.create_stream()
        if self.is_vlm:
            self.mm_processor = self.model.create_multimodal_processor()
            self.mm_stream = self.mm_processor.create_stream()
        else:
            self.mm_processor = None
            self.mm_stream = None
        init_s = time.time() - t0

        self._lock = threading.Lock()
        self.stats = EngineStats(
            model_path=self.model_dir,
            model_id=self.model_id,
            context_length=self.context_length,
            init_seconds=round(init_s, 3),
        )

    # ---------- prompt rendering ----------

    def render_chat(self, messages: List[dict]) -> str:
        """Render OpenAI-style messages into a prompt string using the model's
        chat template if present, else a simple fallback."""
        # Try HF tokenizers template via the model's tokenizer_config.json
        tmpl = self._chat_template_cached()
        if tmpl is not None:
            # Render via jinja2 — AMD models ship HF-compatible templates
            try:
                import jinja2
                env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
                # Some templates call raise_exception (e.g. Gemma) — no-op it
                # since our callers pre-validate message roles.
                env.globals["raise_exception"] = lambda msg: ""
                template = env.from_string(tmpl)
                return template.render(
                    messages=messages,
                    add_generation_prompt=True,
                    bos_token=self._special_tokens_cached().get("bos_token", ""),
                    eos_token=self._special_tokens_cached().get("eos_token", ""),
                )
            except Exception:
                pass  # fall through to simple fallback

        # Simple fallback: <|role|>content<|end|>
        parts = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                # Flatten multimodal content for fallback
                texts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            texts.append(item.get("text", ""))
                        elif item.get("type") == "image":
                            texts.append("[image]")
                    else:
                        texts.append(str(item))
                content = " ".join(texts)
            parts.append(f"<|{m['role']}|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    _chat_template = None
    _chat_template_loaded = False
    _special_tokens: Optional[dict] = None

    def _special_tokens_cached(self) -> dict:
        if self._special_tokens is not None:
            return self._special_tokens
        st: dict = {}
        for fn in ("tokenizer_config.json", "special_tokens_map.json"):
            f = Path(self.model_dir) / fn
            if not f.exists():
                continue
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
                v = d.get(key)
                if isinstance(v, dict):
                    v = v.get("content")
                if isinstance(v, str) and key not in st:
                    st[key] = v
        self._special_tokens = st
        return st

    def _chat_template_cached(self) -> Optional[str]:
        if self._chat_template_loaded:
            return self._chat_template
        self._chat_template_loaded = True
        for fn in ("tokenizer_config.json", "chat_template.json"):
            f = Path(self.model_dir) / fn
            if f.exists():
                try:
                    d = json.loads(f.read_text())
                    tmpl = d.get("chat_template")
                    if isinstance(tmpl, str):
                        self._chat_template = tmpl
                        return tmpl
                except Exception:
                    continue
        # chat_template.jinja as a raw file
        f = Path(self.model_dir) / "chat_template.jinja"
        if f.exists():
            self._chat_template = f.read_text()
        return self._chat_template

    # ---------- generation ----------

    def generate(self, prompt: str, gc: GenerationConfig, images: Optional[list[str]] = None) -> str:
        """Generate full text (non-streaming)."""
        chunks = list(self.stream(prompt, gc, images=images))
        return "".join(chunks)

    def stream(self, prompt: str, gc: GenerationConfig, images: Optional[list[str]] = None) -> Iterator[str]:
        """Stream decoded token pieces."""
        og = _lazy_import_og()
        with self._lock:
            input_tokens = self.tokenizer.encode(prompt)
            prompt_len = len(input_tokens)

            # VLM models: the embedding ONNX is compiled for a fixed max sequence
            # length (e.g. 4096 for Gemma). Don't shrink max_length based on
            # prompt size or image requests will overflow and text requests will
            # underflow the allocated tensors.
            if self.is_vlm:
                max_len = self.max_sequence_length
            else:
                max_len = min(
                    self.context_length,
                    prompt_len + max(1, gc.max_tokens),
                )

            params = og.GeneratorParams(self.model)
            search_opts = {
                "max_length": max_len,
                "do_sample": gc.do_sample,
                "temperature": gc.temperature,
                "top_p": gc.top_p,
                "top_k": gc.top_k,
                "repetition_penalty": gc.repetition_penalty,
            }
            params.set_search_options(**search_opts)

            generator = og.Generator(self.model, params)

            # VLM models take inputs_embeds (from embedding ONNX), not input_ids.
            # Route through MultiModalProcessor with images=None for text-only.
            if self.is_vlm:
                if images:
                    og_images = og.Images.open(*images)
                    inputs = self.mm_processor(prompt, images=og_images)
                else:
                    inputs = self.mm_processor(prompt, images=None)
                generator.set_inputs(inputs)
                decode = self.mm_stream.decode
            else:
                generator.append_tokens(input_tokens)
                decode = self.tokenizer_stream.decode

            completion_tokens = 0
            # For VLM models we allocate a fixed-size KV cache (max_sequence_length)
            # regardless of the request's max_tokens, because the embedding ONNX is
            # compiled for that exact length. But we still honor the caller's
            # max_tokens by breaking the decode loop once that many tokens have
            # been emitted. Without this, Gemma-3-4b runs to natural EOS or the
            # 4K context limit even when the client asked for 80 tokens.
            max_new = max(1, gc.max_tokens)
            try:
                while not generator.is_done():
                    generator.generate_next_token()
                    new_token = generator.get_next_tokens()[0]
                    piece = decode(new_token)
                    if piece:
                        yield piece
                    completion_tokens += 1
                    if completion_tokens >= max_new:
                        break
            finally:
                self.stats.prompt_tokens += prompt_len
                self.stats.completion_tokens += completion_tokens
                self.stats.requests += 1
                del generator
