---
title: KakeyaLattice KV-Cache Codec Demo
emoji: 🧊
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - kv-cache
  - kv-cache-compression
  - quantization
  - lattice-quantization
  - e8-lattice
  - d4-lattice
  - vllm
  - llm-inference
  - long-context
  - transformer
models:
  - Qwen/Qwen3-4B
  - google/gemma-4-e4b
  - zai-org/GLM-4-9B-Chat
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
datasets:
  - wikitext
---

# KakeyaLattice — KV-Cache Compression Demo

Interactive demo for **KakeyaLattice**, a GPU-native D4 / E8 nested-lattice
KV-cache compression codec for transformer LLMs.

- 📦 **Code**: <https://github.com/FluffyAIcode/LLM-KV--Cache-compress>
- 📄 **Paper**: [arXiv (pending)](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/reports/paper/kakeyalattice.pdf)
- 📊 **Papers with Code**: (pending)
- 🔌 **vLLM plugin**: `pip install -e vllm_backend` after cloning repo

This Space lets you:

1. **Try the codec on synthetic KV tensors** — visualise MSE, bit-rate, and
   reconstruction error for D4 (v1.4) vs E8 (v1.5) vs Z^N scalar baseline.
2. **Reproduce the headline PPL/MSE tables** by loading the frozen JSON
   from `reports/v1_4_release/` and `reports/v1_5_release/`.
3. **Inspect the nine-step pipeline** (unit-norm, Hadamard, q_max, lattice,
   clamp) step by step on a single KV vector.

This Space does **not** run a full LLM (too heavy for the free tier). To try
KakeyaLattice on a live model, install the vLLM plugin locally:

```bash
git clone https://github.com/FluffyAIcode/LLM-KV--Cache-compress
cd LLM-KV--Cache-compress
pip install -e kakeyalattice -e vllm_backend
export KAKEYA_SNAPSHOT_QWEN3=1
vllm serve Qwen/Qwen3-4B
```

## Citation

```bibtex
@misc{li2026kakeyalattice,
  author       = {Allen Li},
  title        = {{KakeyaLattice}: Nested-Lattice {KV}-Cache Compression
                  with Kakeya-Style Discrete Codebooks},
  year         = {2026},
  howpublished = {\url{https://github.com/FluffyAIcode/LLM-KV--Cache-compress}},
  note         = {D4 (v1.4) + E8 (v1.5) joint release; arXiv preprint in progress}
}
```

## License

Code: Apache-2.0. Paper: CC BY 4.0 on arXiv.
