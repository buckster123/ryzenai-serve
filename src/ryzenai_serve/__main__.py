"""CLI entry point for ryzenai-serve.

Supports three startup modes:
  • chat only:        --model-dir PATH
  • embeddings only:  --embedder-dir PATH
  • both:             --model-dir PATH --embedder-dir PATH
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ryzenai-serve",
        description="OpenAI-compatible HTTP server for AMD Ryzen AI (NPU LLMs + CPU/NPU embeddings).",
    )
    parser.add_argument("--model-dir", default=None,
                        help="Path to OGA LLM model directory (contains genai_config.json). "
                             "Enables /v1/chat/completions.")
    parser.add_argument("--model-id", default=None,
                        help="LLM model id reported to clients. Defaults to dir basename.")
    parser.add_argument("--embedder-dir", default=None,
                        help="Path to ONNX sentence-embedding model directory (with tokenizer files). "
                             "Enables /v1/embeddings.")
    parser.add_argument("--embedder-id", default=None,
                        help="Embedding model id reported to clients. Defaults to dir basename.")
    parser.add_argument("--embedder-onnx", default="model.onnx",
                        help="Relative ONNX filename inside --embedder-dir (default: model.onnx; "
                             "'onnx/model.onnx' auto-fallback).")
    parser.add_argument("--embedder-pool", default="cls", choices=["cls", "mean"],
                        help="Pooling: 'cls' for BGE/BERT, 'mean' for MiniLM/SBERT (default: cls).")
    parser.add_argument("--embedder-max-length", type=int, default=512,
                        help="Max tokenizer length for embedder (default: 512).")
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

    if not args.model_dir and not args.embedder_dir:
        parser.error("at least one of --model-dir or --embedder-dir is required")

    import os
    if args.model_dir and not os.environ.get("RYZEN_AI_INSTALLATION_PATH"):
        print("WARNING: RYZEN_AI_INSTALLATION_PATH not set. "
              "Did you `source ~/run_llm/env.sh` first?", file=sys.stderr)
    if args.model_dir and "/opt/xilinx/xrt/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
        print("WARNING: /opt/xilinx/xrt/lib not in LD_LIBRARY_PATH. "
              "libxrt_coreutil.so may fail to load.", file=sys.stderr)

    engine = None
    embedder = None

    if args.model_dir:
        print(f"[ryzenai-serve] loading LLM: {args.model_dir}", file=sys.stderr)
        from ryzenai_serve.engine import NPUEngine
        engine = NPUEngine(args.model_dir, model_id=args.model_id)
        print(f"[ryzenai-serve] LLM ready: model={engine.model_id} "
              f"ctx={engine.context_length} init={engine.stats.init_seconds}s",
              file=sys.stderr)

    if args.embedder_dir:
        print(f"[ryzenai-serve] loading embedder: {args.embedder_dir}", file=sys.stderr)
        from ryzenai_serve.embedder import EmbeddingEngine
        embedder = EmbeddingEngine(
            args.embedder_dir,
            onnx_subpath=args.embedder_onnx,
            model_id=args.embedder_id,
            pool=args.embedder_pool,
            max_length=args.embedder_max_length,
        )
        print(f"[ryzenai-serve] embedder ready: model={embedder.model_id} "
              f"dim={embedder.dim} init={embedder.stats.init_seconds}s",
              file=sys.stderr)

    import uvicorn
    from ryzenai_serve.server import create_app
    app = create_app(engine=engine, embedder=embedder)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
