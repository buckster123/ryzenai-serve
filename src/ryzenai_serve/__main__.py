"""CLI entry point: `ryzenai-serve --model-dir PATH [--host 0.0.0.0 --port 8000]`"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ryzenai-serve",
        description="OpenAI-compatible HTTP server for AMD Ryzen AI NPU LLMs.",
    )
    parser.add_argument("--model-dir", required=False,
                        help="Path to OGA model directory (contains genai_config.json). "
                             "E.g. ~/run_llm/Llama-3.2-3B-Instruct_rai_1.7.1_npu_16K")
    parser.add_argument("--model-id", default=None,
                        help="Model ID reported to clients. Defaults to dir basename.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info",
                        choices=["critical", "error", "warning", "info", "debug"])
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)

    if args.version:
        from ryzenai_serve import __version__
        print(__version__)
        return 0

    if not args.model_dir:
        parser.error("--model-dir is required")

    # Environment sanity — AMD venv + XRT must be sourced
    import os
    if not os.environ.get("RYZEN_AI_INSTALLATION_PATH"):
        print("WARNING: RYZEN_AI_INSTALLATION_PATH not set. "
              "Did you `source ~/run_llm/env.sh` first?", file=sys.stderr)
    if "/opt/xilinx/xrt/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
        print("WARNING: /opt/xilinx/xrt/lib not in LD_LIBRARY_PATH. "
              "libxrt_coreutil.so may fail to load.", file=sys.stderr)

    print(f"[ryzenai-serve] loading model: {args.model_dir}", file=sys.stderr)
    from ryzenai_serve.engine import NPUEngine
    engine = NPUEngine(args.model_dir, model_id=args.model_id)
    print(f"[ryzenai-serve] ready: model={engine.model_id} "
          f"ctx={engine.context_length} init={engine.stats.init_seconds}s",
          file=sys.stderr)

    import uvicorn
    from ryzenai_serve.server import create_app
    app = create_app(engine)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
