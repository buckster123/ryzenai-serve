"""ryzenai-serve — OpenAI-compatible HTTP server for AMD Ryzen AI NPU LLMs.

Wraps onnxruntime-genai models (amd/*_rai_1.7.1_npu_{4K,16K} collections)
with a /v1/chat/completions endpoint compatible with OpenAI, LMStudio,
and llama.cpp server clients.
"""

__version__ = "0.3.2"
