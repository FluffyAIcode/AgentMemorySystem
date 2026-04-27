# Snippet: KV-cache compression section for model cards

If you publish a KakeyaLattice-compressed checkpoint (e.g. a Qwen3-4B
fine-tune that ships with a pre-computed lattice parity table), add this
section to the HuggingFace model card. It takes ~60 seconds and creates
another inbound backlink to the repo.

```markdown
## KV-cache compression

This model is compatible with [**KakeyaLattice**](https://github.com/FluffyAIcode/LLM-KV--Cache-compress),
a GPU-native D4 / E8 nested-lattice KV-cache codec that plugs into vLLM
as a `vllm.general_plugins` entry point. Measured on H200 bf16:

| Config | CR | |Δppl| | NIAH @ 16k |
|---|---|---|---|
| KakeyaLattice v1.5 Q=10 | 2.77× | 1.45% | 100% |
| KakeyaLattice v1.5 Q=22 | 1.73× | <1% | 100% |
| TurboQuant b=4 (baseline) | 2.18× | 6.58% | — |

Enable with:

​```bash
pip install -e git+https://github.com/FluffyAIcode/LLM-KV--Cache-compress.git#egg=kakeyalattice \
           -e git+https://github.com/FluffyAIcode/LLM-KV--Cache-compress.git#egg=kakeya_v1_4_snapshot\&subdirectory=vllm_backend
export KAKEYA_SNAPSHOT_QWEN3=1
vllm serve <this-model>
​```
```

## Which model cards to edit (if you own or co-maintain them)

The highest-value cards to add this snippet to are any where **you**
personally already publish weights:

- Any `FluffyAIcode/*` models
- Any model you've published for AgentMemorySystem
- Any KakeyaLattice-quantised variant you publish (e.g.
  `FluffyAIcode/Qwen3-4B-KakeyaLattice-Q10` — worth publishing even as a
  tiny config-only repo, because the HF hub's search indexes the model
  card and creates a backlink)

Do **not** edit model cards you don't own — it's considered spammy and
will get the repo flagged.
