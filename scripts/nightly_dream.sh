#!/usr/bin/env bash
# nightly_dream.sh — run CerebroCortex Dream Engine overnight on NPU
# Designed to be invoked from cron/systemd at 3am. Starts ryzenai-serve
# ephemerally, runs dream, tears down. Logs to ~/.cerebro-cortex/dream_logs/.
#
# Cron example (crontab -e):
#   0 3 * * * /home/andre/projects/ryzenai-serve/scripts/nightly_dream.sh

set -uo pipefail

MODEL_DIR="${NPU_MODEL_DIR:-$HOME/run_llm/Llama-3.2-3B-Instruct_rai_1.7.1_npu_16K}"
PORT="${NPU_PORT:-8000}"
CC_DIR="${CC_DIR:-$HOME/projects/CerebroCortex}"
LOG_DIR="$HOME/.cerebro-cortex/dream_logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
SERVE_LOG="$LOG_DIR/serve_${TS}.log"
DREAM_LOG="$LOG_DIR/dream_${TS}.log"

# Preflight: bail if NPU busy
if pgrep -f model_benchmark >/dev/null || pgrep -f "ryzenai-serve --" >/dev/null; then
  echo "[$(date)] NPU busy, skipping nightly dream" >> "$DREAM_LOG"
  exit 0
fi

# Source env
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
source "$HOME/ryzen_ai/venv/bin/activate"
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:$HOME/ryzen_ai/venv/deployment/lib:${LD_LIBRARY_PATH:-}"
export RYZEN_AI_INSTALLATION_PATH="$VIRTUAL_ENV"
export RYZENAI_EP_PATH="$VIRTUAL_ENV/deployment/lib/libonnxruntime_providers_ryzenai.so"
export XRT_INI_PATH="$HOME/run_llm/xrt.ini"

# Launch server in background
ryzenai-serve --model-dir "$MODEL_DIR" --host 127.0.0.1 --port "$PORT" --log-level warning \
  > "$SERVE_LOG" 2>&1 &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Wait for ready
for i in $(seq 1 30); do
  curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1 && break
  sleep 1
  kill -0 "$SERVER_PID" 2>/dev/null || { echo "[$(date)] server died during startup" >> "$DREAM_LOG"; tail -20 "$SERVE_LOG" >> "$DREAM_LOG"; exit 1; }
done

# Run dream (CLI auto-loads settings.json which should point at localhost:8000)
cd "$CC_DIR"
echo "[$(date)] starting dream cycle" >> "$DREAM_LOG"
./venv/bin/python -m cerebro.interfaces.cli dream run >> "$DREAM_LOG" 2>&1
echo "[$(date)] dream cycle done" >> "$DREAM_LOG"
