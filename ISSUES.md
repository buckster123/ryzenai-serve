# Known issues

## Python onnxruntime-genai 0.11.2: degenerate output on non-Llama NPU-16K models

**Status**: Upstream bug, workaround in place (default to Llama-3.2-3B).
**Severity**: Blocks serving Qwen2.5-{3B,7B} and Phi-4-mini-{instruct,reasoning}.
**Platform**: Krackan Point (Ryzen AI 5 340), Ubuntu 25.10, SDK 1.7.1, ORT-GenAI 0.11.2.

### Symptom

Any non-Llama NPU-16K model loaded through the Python `onnxruntime_genai.Model` + `Generator` API produces 1-2 correct tokens, then collapses into a repeating single-token loop regardless of sampling parameters:

```
Qwen2.5-3B:    "TheGGGGGGGGGGGGGGGGGGG..."
Qwen2.5-7B:    "TheGGGGGGGGGGGGGGGGGGG..."
Phi-4-mini:    "TheGGGGGGGGGGGGGGGGGGG..."
Phi-4-reason:  "<thGGGGGGGGGGGGGGGGGGG..."
Llama-3.2-3B:  "The capital of France is Paris."     ← only this one works
```

### Reproducer

```bash
source ~/run_llm/env.sh
export LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:$LD_LIBRARY_PATH"
python3 scripts/probe_all_models_oga.py
```

Script is in `scripts/probe_all_models_oga.py` — loads each of the 5 published NPU-16K models in turn, encodes a canonical "What is the capital of France?" prompt via the correct chat template for each family, and runs 40 decode steps.

### Confirmed not the cause

- **Not a chat template issue**: every model is rendered with the correct family-specific template (ChatML for Qwen, Llama-3 for Llama, Phi tags for Phi). Llama works, others don't, through the exact same code path.
- **Not a sampling issue**: greedy (`do_sample=False`, `top_k=1`), sampling with penalties, and defaults-only all fail identically.
- **Not a model-compile issue**: AMD's `model_benchmark` (C++ binary) runs all 5 models cleanly at the published tok/s numbers. The models are fine; the Python API path breaks them.
- **Not unique to our wrapper**: AMD's own `run_model.py` reproduces the same degeneration on Qwen2.5 and emits a telling warning: `Unsupported model type 'qwen2'. Defaulting to ONNXTokenizerWrapper.`

### Suspected cause

Some combination of:

1. `tokenizer_factory.get_tokenizer()` dispatching Qwen2/Phi model types to a generic `ONNXTokenizerWrapper` that mishandles BOS/EOS or special token IDs, leaving the KV cache with a corrupt first state.
2. Tied embeddings (`tie_word_embeddings=True` on Qwen2.5) interacting badly with the NPU EP's logits layer output, producing collapsed distributions.
3. A logits processor in the Python path that's nulling out top-k except token 38 ("G" in these tokenizers' vocabs).

C++ `model_benchmark` bypasses the Python tokenizer factory and uses a different generator loop, which is why it works.

### Workaround

Use `amd/Llama-3.2-3B-Instruct_rai_1.7.1_npu_16K` as the default model. It's slower on paper (18.2 vs 21.8 t/s decode, 10.5 vs 7.3 GB peak RAM vs Qwen2.5-3B) but it's the only one that currently generates coherent text through the Python API.

### If you're debugging this upstream

The entry point to look at is `onnxruntime_genai/models/builder.py`'s `Qwen2ForCausalLM` branch (line 4730 in 0.11.2) and the tokenizer factory in `/home/andre/ryzen_ai/venv/LLM/examples/tokenizer_factory.py`. Also worth comparing generated ONNX graphs between Llama and Qwen2.5 for KV cache layer differences.

### How we'll know it's fixed

Re-run `scripts/probe_all_models_oga.py` after any Ryzen AI SDK or `onnxruntime-genai` bump. When all 5 models output coherent sentences, remove this file.
