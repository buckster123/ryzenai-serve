"""Quick Python-OGA smoke test across all 5 NPU-16K models.
For each model: load, encode a canonical prompt, generate 40 tokens, report output.
Helps identify which models are usable via the Python API (what ryzenai-serve needs)
vs which only work via the C++ model_benchmark binary."""
import onnxruntime_genai as og
from pathlib import Path
import sys, time, traceback

MODELS = [
    # (dir_name, chat_template_render_function)
    ("Llama-3.2-3B-Instruct_rai_1.7.1_npu_16K",
     lambda q: f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"),
    ("Qwen2.5_3B_Instruct_rai_1.7.1_npu_16K",
     lambda q: f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"),
    ("Qwen2.5-7B-Instruct_rai_1.7.1_npu_16K",
     lambda q: f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"),
    ("Phi-4-mini-instruct_rai_1.7.1_npu_16K",
     lambda q: f"<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n{q}<|end|>\n<|assistant|>\n"),
    ("Phi-4-mini-reasoning_rai_1.7.1_npu_16K",
     lambda q: f"<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\n{q}<|end|>\n<|assistant|>\n"),
]

QUESTION = "What is the capital of France? Answer in one short sentence."
GEN = 40

print(f"{'model':40s}  {'status':8s}  output")
print("-"*140)

for name, tmpl in MODELS:
    path = Path.home() / "run_llm" / name
    if not path.exists():
        print(f"{name:40s}  SKIP      (dir missing)")
        continue
    try:
        t0 = time.time()
        m = og.Model(str(path))
        tok = og.Tokenizer(m)
        prompt = tmpl(QUESTION)
        enc = tok.encode(prompt)

        params = og.GeneratorParams(m)
        params.set_search_options(
            max_length=len(enc)+GEN,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
        )
        gen = og.Generator(m, params)
        gen.append_tokens(enc)
        stream = tok.create_stream()
        out = ""
        while not gen.is_done():
            gen.generate_next_token()
            t = gen.get_next_tokens()[0]
            out += stream.decode(t)
        elapsed = time.time() - t0
        # Classify: degenerate output has huge run of same char, normal has spaces + variety
        max_run = 1
        cur = 1
        for i in range(1, len(out)):
            if out[i] == out[i-1]:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 1
        status = "BROKEN" if max_run > 15 else "OK"
        print(f"{name:40s}  {status:8s}  {out!r}")
        del gen, m, tok
    except Exception as e:
        print(f"{name:40s}  ERROR     {type(e).__name__}: {e}")
        traceback.print_exc()
