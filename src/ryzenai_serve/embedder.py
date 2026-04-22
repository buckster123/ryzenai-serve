"""Sentence-embedding engine for ryzenai-serve.

Pluggable backend so we can swap CPU-INT8 ONNX for NPU-VAIML later
without touching the HTTP layer. Current backend: ORT CPU INT8.

Mirrors the BGE/MiniLM convention:
  - encode text -> last_hidden_state[B,L,H]
  - pool: [CLS] token (index 0)   (BGE default; SBERT mean-pool also supported)
  - L2 normalize
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import List, Literal, Optional

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


PoolStrategy = Literal["cls", "mean"]


@dataclass
class EmbedStats:
    model_id: str
    model_path: str
    dim: int
    init_seconds: float
    requests: int = 0
    input_tokens: int = 0


class EmbeddingEngine:
    """CPU-first ORT embedding engine. NPU-ready (swap provider in init).

    Args:
        model_dir: directory with ONNX model + tokenizer files.
        onnx_subpath: which ONNX file under model_dir to load.
        model_id: id reported to clients (default: dir basename).
        pool: pooling strategy -- "cls" (BGE/BERT) or "mean" (MiniLM/SBERT).
        max_length: tokenizer truncation cap.
        providers: ORT providers to try in order. Default: ["CPUExecutionProvider"].
    """

    def __init__(
        self,
        model_dir: str,
        onnx_subpath: str = "model.onnx",
        model_id: Optional[str] = None,
        pool: PoolStrategy = "cls",
        max_length: int = 512,
        providers: Optional[List] = None,
    ):
        self.model_dir = os.path.abspath(model_dir)
        onnx_path = os.path.join(self.model_dir, onnx_subpath)
        if not os.path.isfile(onnx_path):
            # try onnx/model.onnx layout (HF sentence-transformers default)
            alt = os.path.join(self.model_dir, "onnx", "model.onnx")
            if os.path.isfile(alt):
                onnx_path = alt
            else:
                raise FileNotFoundError(
                    f"No ONNX model at {onnx_path} or {alt}"
                )
        self.onnx_path = onnx_path
        self.model_id = model_id or os.path.basename(self.model_dir.rstrip("/"))
        self.pool = pool
        self.max_length = max_length
        self.providers = providers or ["CPUExecutionProvider"]

        self._lock = Lock()
        t0 = time.perf_counter()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        # intra_op_num_threads: let ORT auto-pick but cap at physical cores
        self.session = ort.InferenceSession(
            self.onnx_path, sess_options=so, providers=self.providers
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        # Inspect output dim from model
        out_shape = self.session.get_outputs()[0].shape  # [B, L, H]
        dim = int(out_shape[-1]) if isinstance(out_shape[-1], int) else 0
        if dim == 0:
            # Run a dummy to discover
            enc = self.tokenizer("x", return_tensors="np", padding=True, truncation=True, max_length=8)
            feeds = self._feeds(enc)
            out = self.session.run(None, feeds)[0]
            dim = int(out.shape[-1])
        self.dim = dim
        self._input_names = {i.name for i in self.session.get_inputs()}
        self._output_name = self.session.get_outputs()[0].name
        init = time.perf_counter() - t0

        self.stats = EmbedStats(
            model_id=self.model_id,
            model_path=self.onnx_path,
            dim=self.dim,
            init_seconds=round(init, 2),
        )

    def _feeds(self, enc) -> dict:
        feeds = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in self._input_names:
            tti = enc.get("token_type_ids")
            if tti is None:
                tti = np.zeros_like(enc["input_ids"])
            feeds["token_type_ids"] = tti.astype(np.int64)
        # drop any feed the model doesn't accept
        return {k: v for k, v in feeds.items() if k in self._input_names}

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized [N, dim] float32."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        feeds = self._feeds(enc)
        with self._lock:
            out = self.session.run([self._output_name], feeds)[0]  # [B,L,H]
            self.stats.requests += 1
            self.stats.input_tokens += int(enc["input_ids"].size)

        if self.pool == "cls":
            pooled = out[:, 0, :]  # [B, H]
        else:
            # mean pool with attention mask
            mask = feeds["attention_mask"].astype(np.float32)[..., None]  # [B,L,1]
            s = (out * mask).sum(axis=1)
            n = np.clip(mask.sum(axis=1), 1e-9, None)
            pooled = s / n

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        pooled = pooled / np.clip(norms, 1e-12, None)
        return pooled.astype(np.float32)
