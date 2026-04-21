#!/usr/bin/env bash
# End-to-end integration test: ryzenai-serve → CerebroCortex LLMClient → NPU Llama-3.2-3B-Instruct-16K
# Prereqs: NPU idle (no bench running), pmode=turbo, bench already completed
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/run_llm/Llama-3.2-3B-Instruct_rai_1.7.1_npu_16K}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

echo "=== 1. Sanity checks ==="
[ -f "$MODEL_DIR/genai_config.json" ] || { echo "Missing $MODEL_DIR/genai_config.json"; exit 1; }
pgrep -f model_benchmark >/dev/null && { echo "NPU busy (model_benchmark running) — wait for bench"; exit 2; }
curl -s "http://$HOST:$PORT/" >/dev/null 2>&1 && { echo "Something already listening on $HOST:$PORT"; exit 3; }
echo "  OK: model dir present, NPU idle, port $PORT free"

echo ""
echo "=== 2. Launch ryzenai-serve (background) ==="
source /opt/xilinx/xrt/setup.sh >/dev/null
source "$HOME/ryzen_ai/venv/bin/activate"
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:$HOME/ryzen_ai/venv/deployment/lib:${LD_LIBRARY_PATH:-}"
export RYZEN_AI_INSTALLATION_PATH="$VIRTUAL_ENV"
export RYZENAI_EP_PATH="$VIRTUAL_ENV/deployment/lib/libonnxruntime_providers_ryzenai.so"
export XRT_INI_PATH="$HOME/run_llm/xrt.ini"

LOG="$HOME/run_llm/results/ryzenai_serve.log"
: > "$LOG"
ryzenai-serve --model-dir "$MODEL_DIR" --host "$HOST" --port "$PORT" --log-level info \
  >> "$LOG" 2>&1 &
SERVER_PID=$!
echo "  server PID: $SERVER_PID — logs: $LOG"

cleanup() { echo "  stopping server..."; kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo ""
echo "=== 3. Wait for server to load model (up to 30s) ==="
for i in $(seq 1 30); do
  if curl -sf "http://$HOST:$PORT/" >/dev/null 2>&1; then
    echo "  ready after ${i}s"; break
  fi
  sleep 1
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "  SERVER DIED — dumping last 40 lines of log:"
    tail -40 "$LOG"
    exit 4
  fi
done
curl -sf "http://$HOST:$PORT/" >/dev/null 2>&1 || { echo "  server never came ready"; tail -40 "$LOG"; exit 5; }

echo ""
echo "=== 4. Direct HTTP probe ==="
curl -s "http://$HOST:$PORT/" | python3 -m json.tool
echo ""
echo "--- chat completion (non-streaming) ---"
curl -s "http://$HOST:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"In one sentence, what is a neural network?"}],"max_tokens":80,"temperature":0.5}' \
  | python3 -m json.tool

echo ""
echo "=== 5. Via CerebroCortex LLMClient ==="
cd "$HOME/projects/CerebroCortex"
OPENAI_COMPAT_BASE_URL="http://$HOST:$PORT" \
LLM_PRIMARY_PROVIDER="openai_compat" \
LLM_PRIMARY_MODEL="Llama-3.2-3B-Instruct-npu-16K" \
./venv/bin/python -c "
import os
# Apply env overrides to CC config
from cerebro import config
config.LLM_PRIMARY_PROVIDER = os.environ['LLM_PRIMARY_PROVIDER']
config.LLM_PRIMARY_MODEL = os.environ['LLM_PRIMARY_MODEL']
config.OPENAI_COMPAT_BASE_URL = os.environ['OPENAI_COMPAT_BASE_URL']
config.LLM_FALLBACK_PROVIDER = 'ollama'  # dummy, avoid anthropic key demand
config.LLM_FALLBACK_MODEL = 'none'
config.OPENAI_COMPAT_NO_THINK = False     # Qwen2.5 isn't a thinking model
config.OPENAI_COMPAT_STRIP_THINK = False

from cerebro.utils.llm import LLMClient
client = LLMClient()
resp = client.generate(
    prompt='List three traits of a good consolidation engine for an AI memory system. Be brief.',
    system='You are concise and technical.',
    max_tokens=150,
    temperature=0.3,
)
print(f'Provider: {resp.provider}  Model: {resp.model}  Tokens: {resp.tokens_used}  Fallback: {resp.was_fallback}')
print('---')
print(resp.text)
print('---')
print('Client stats:', client.stats())
"

echo ""
echo "=== 6. Server-side stats ==="
curl -s "http://$HOST:$PORT/stats" | python3 -m json.tool

echo ""
echo "=== DONE ==="
