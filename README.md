# ryzenai-serve

OpenAI-compatible HTTP server for AMD Ryzen AI on Linux — NPU LLMs (`/v1/chat/completions`), vision-language models (`/v1/chat/completions` with images), and CPU/NPU sentence embeddings (`/v1/embeddings`).

Wraps `onnxruntime-genai` for the chat side so pre-quantized NPU models from the `amd/ryzen-ai-171-npu-{4K,16K}` HuggingFace collections are servable behind OpenAI-shaped endpoints. Wraps stock `onnxruntime` for the embeddings side so any HF `sentence-transformers` ONNX (BGE, MiniLM, etc., INT8 or FP32) works as a drop-in vector store backend. A single binary can run either or both engines concurrently — making it a drop-in backend for LMStudio/Ollama-compatible clients, LangChain, CerebroCortex's Dream Engine, and anything else that speaks the OpenAI API.

## Status

`0.3.0` — adds **vision support** (OpenAI multimodal `image_url`), a **built-in web chat UI**, and CORS. Chat + embeddings from v0.2.0 still supported. Tested against Llama-3.2-3B NPU-16K, Gemma-3-4b-it NPU-4K (multimodal) on Krackan Point (Ryzen AI 5 340) and BAAI/bge-small-en-v1.5 INT8 on CPU.

## Requirements

- Linux with AMD Ryzen AI SDK 1.7.1+ installed for NPU use (see `mlops/amd-xdna-linux` skill). Not required for embeddings-only.
- XDNA NPU (Krackan / Strix / Strix Halo) for NPU use
- Python 3.10+
- For chat: an OGA-packaged NPU model (directory containing `genai_config.json`)
- For embeddings: an ONNX sentence-embedding model dir with tokenizer files (HF `sentence-transformers` layout works as-is)

## Install

```bash
# Inside the Ryzen AI venv (so onnxruntime_genai is available for NPU LLM)
source ~/run_llm/env.sh
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH"

pip install -e .
```

## Run

Three modes — launch chat only, embeddings only, or both:

```bash
# chat only (NPU text)
ryzenai-serve \
  --model-dir ~/run_llm/Llama-3.2-3B-Instruct_rai_1.7.1_npu_16K \
  --host 127.0.0.1 --port 8000

# chat only (NPU vision — Gemma multimodal)
# IMPORTANT: start from the model dir because genai_config has relative cache paths
cd ~/run_llm/gemma-3-4b-it_rai_1.7.1_npu_4K
ryzenai-serve \
  --model-dir . \
  --model-id gemma-3-4b-it-npu-4K \
  --host 127.0.0.1 --port 8002

# embeddings only (CPU)
ryzenai-serve \
  --embedder-dir ~/run_llm/bge-small-en-v1.5-int8 \
  --embedder-pool cls \
  --host 127.0.0.1 --port 8001

# both, one process
ryzenai-serve \
  --model-dir ~/run_llm/Llama-3.2-3B-Instruct_rai_1.7.1_npu_16K \
  --embedder-dir ~/run_llm/bge-small-en-v1.5-int8 \
  --host 127.0.0.1 --port 8000
```

Recommended production layout: separate processes on ports 8000/8001/8002 so you can restart them independently and avoid serializing NPU vs CPU work behind one uvicorn worker.

## Web chat UI

Open `http://localhost:8000/chat` (or whichever port the server runs on) for a zero-dependency browser chat interface. Features:

- **Streaming** — token-by-token display
- **Endpoint switching** — switch between :8000 Llama, :8002 Gemma, or custom URLs
- **Image upload** — click "+img" to attach images (multi-select supported). Sent as base64 `image_url` in OpenAI multimodal format. Only works against vision-capable models (Gemma); non-VLM models return a 400 with a helpful message.
- **System prompt** — click "system" to set
- **Parameter tweaking** — temperature, top_p, max_tokens
- **Copy / retry** buttons on assistant messages

## Test

```bash
# Chat — non-streaming
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":100}' \
  | jq .

# Chat — streaming (SSE)
curl -N localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me a joke."}],"stream":true,"max_tokens":80}'

# Vision — base64 image (Gemma on :8002)
curl -s localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{
      "role":"user",
      "content":[
        {"type":"text","text":"What color is this?"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}}
      ]
    }],
    "max_tokens":20
  }' | jq '.choices[0].message.content'

# Embeddings
curl -s localhost:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":["The NPU is fast.","Memory consolidation happens in REM."]}' \
  | jq '.data[0].embedding | length'   # -> 384

# Model listing
curl -s localhost:8001/v1/models | jq .

# Stats
curl -s localhost:8000/stats | jq .
curl -s localhost:8001/stats | jq .
```

## Use with CerebroCortex

```json
{
  "llm": {
    "primary_provider": "openai_compat",
    "primary_model": "Llama-3.2-3B-Instruct-npu-16K",
    "openai_compat_base_url": "http://127.0.0.1:8000",
    "openai_compat_embedding_base_url": "http://127.0.0.1:8001",
    "openai_compat_embedding_model": "bge-small-en-v1.5-int8"
  }
}
```

`./cerebro dream run` routes LLM calls through the NPU and embedding calls through the CPU INT8 BGE — full local stack, ~15W sustained. Dream cycle on 14-memory corpus: ~83s, 9 LLM calls + ~20 embedding batches. See the `mlops/cerebro-npu-backend` skill for the full runbook including corpus-migration scripts.

## Building an INT8 embedder

Stock `onnxruntime.quantization.quantize_dynamic` (no custom ops — avoids Quark's libcustom_ops ABI drift on SDK 1.7.1):

```python
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    model_input="bge-small-en-v1.5/onnx/model.onnx",
    model_output="bge-small-en-v1.5-int8/model.onnx",
    weight_type=QuantType.QInt8,
    per_channel=True,
    op_types_to_quantize=["MatMul", "Gemm"],
)
```

Copy the tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, `special_tokens_map.json`, `config.json`, `modules.json`, `config_sentence_transformers.json`, `sentence_bert_config.json`) next to `model.onnx`. Result: ~2x faster, ~50% smaller, cosine-vs-FP32 ≥ 0.996.

## Design notes

- **Pluggable embedding backend** — `EmbeddingEngine` accepts any ORT provider list. Today it defaults to `CPUExecutionProvider`; when BGE-on-VitisAI-EP unblocks (tracking onnxruntime pinning + non-Quark quant paths), the same class will swap providers without any HTTP changes.
- **Single model per role, load-at-startup**. NPU init is ~4-7s and peak RAM is the model size; swapping isn't cheap enough for multi-model hosting on 24GB systems. Use multiple ports if you need multiple models.
- **Generation is serialized** under a threading lock — the NPU runs one sequence at a time. Embeddings also serialize (ORT session isn't thread-safe in general).
- **Chat templating** uses the model's `chat_template` from `tokenizer_config.json` via jinja2, so AMD-shipped models work without per-model configuration. Multimodal messages with images are automatically transformed to template-friendly `{"type": "image"}` parts.
- **Vision support** follows the OpenAI API: `content` can be a string or a list of `{"type":"text"}` / `{"type":"image_url", "image_url":{"url":"..."}}` parts. Supported URL types: `data:image/...;base64,...`, `http://...`, `https://...`, local file paths.
- **VLM max_length** is read from the compiled model's KV cache limit (`max_lenght_for_kv_cache` in decoder session options), not the config's `context_length`. This avoids tensor shape mismatches when the NPU compile is smaller than the advertised context (e.g. Gemma-3-4b claims 16K but is compiled for 4K).
- **Pooling** — `--embedder-pool cls` (BGE/BERT default) or `--embedder-pool mean` (MiniLM/SBERT). L2-normalized output in all cases.

## Known limits

- No batching on chat (NPU single-session).
- No function-calling / tool use (pass-through only).
- No `/v1/completions` (legacy endpoint — use chat completions).
- Embeddings `encoding_format: "base64"` not yet supported; use `"float"`.
- `stop` strings are detokenized post-hoc; very long stop strings may overshoot by a few tokens.
- BGE-on-NPU via Vitis AI EP is blocked as of SDK 1.7.1 (Quark libcustom_ops ABI drift, see `mlops/cerebro-npu-backend` skill). CPU-INT8 is the shipping default.
- Gemma text-only can loop occasionally at low temperatures; use temperature ≥ 0.7 for best results.

## License

MIT
