# ryzenai-serve

OpenAI-compatible HTTP server for AMD Ryzen AI NPU LLMs on Linux.

Wraps `onnxruntime-genai` so pre-quantized NPU models from the `amd/ryzen-ai-171-npu-{4K,16K}` HuggingFace collections can be served behind `/v1/chat/completions` — making them a drop-in backend for LMStudio/Ollama/OpenAI-compatible clients, LangChain, CerebroCortex's Dream Engine, and anything else that speaks the OpenAI API.

## Status

`0.1.0` — single-model serve, chat completions (sync + streaming), `/v1/models`, `/stats`. Tested against Llama-3.2-3B, Phi-4-mini, Qwen2.5-{3B,7B} NPU-16K on Krackan Point (Ryzen AI 5 340).

## Requirements

- Linux with AMD Ryzen AI SDK 1.7.1+ installed (see `mlops/amd-xdna-linux` skill)
- XDNA NPU (Krackan / Strix / Strix Halo)
- Python 3.10+
- An OGA-packaged NPU model (any directory containing `genai_config.json`)

## Install

```bash
# Inside the Ryzen AI venv (so onnxruntime_genai is available)
source ~/run_llm/env.sh
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH"

pip install -e .
```

## Run

```bash
ryzenai-serve \
  --model-dir ~/run_llm/Llama-3.2-3B-Instruct_rai_1.7.1_npu_16K \
  --host 127.0.0.1 --port 8000
```

## Test

```bash
# Non-streaming
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Why is the sky blue?"}],"max_tokens":100}' \
  | jq .

# Streaming
curl -N localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me a joke."}],"stream":true,"max_tokens":80}'

# Model listing
curl -s localhost:8000/v1/models | jq .

# Stats
curl -s localhost:8000/stats | jq .
```

## Use with CerebroCortex

In CerebroCortex's config (`src/cerebro/config.py` or env overrides):

```python
LLM_PRIMARY_PROVIDER = "openai_compat"
LLM_PRIMARY_MODEL = "Llama-3.2-3B-Instruct_rai_1.7.1_npu_16K"   # any string; server ignores
OPENAI_COMPAT_BASE_URL = "http://localhost:8000"
```

Then `./cerebro dream --run` routes all Dream Engine LLM calls through the NPU — offline overnight consolidation at ~15W sustained.

## Design notes

- **Single model, load-at-startup**. NPU init is ~4-7s and peak RAM is the model size; swapping isn't cheap enough for multi-model hosting on 24GB systems. Use multiple ports if you need multiple models.
- **Generation is serialized** under a threading lock — the NPU runs one sequence at a time.
- **Chat templating** uses the model's `chat_template` from `tokenizer_config.json` via jinja2, so AMD-shipped models work without per-model configuration.
- **Streaming is SSE** (standard OpenAI format).
- **`/v1/embeddings` is not implemented yet** — planned for when BGE-on-NPU is wired up (see `amd-xdna-linux` skill, Phase 3).

## Known limits

- No batching (NPU single-session).
- No function-calling / tool use (pass-through only).
- No `/v1/completions` (legacy endpoint — use chat completions).
- `stop` strings are detokenized post-hoc; very long stop strings may overshoot by a few tokens.

## License

MIT
