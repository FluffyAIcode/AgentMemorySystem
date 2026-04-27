## Summary

Sharing **KakeyaLattice** — a KV-cache compression codec that plugs into
vLLM via `vllm.general_plugins` and compresses K/V post-QK/V-norm, pre-RoPE.
Validated on **real vLLM + real HF weights + FlashAttention bf16** on an
NVIDIA H200 across four open-source model families.

Motivation is the same class of problem as
[#39241 (NexusQuant / E8 VQ)](https://github.com/vllm-project/vllm/issues/39241):
KV-cache memory is the dominant constraint at 128k+ contexts. We attack it
from a slightly different angle — a **Zamir-Feder nested-lattice quantiser**
(D4 in v1.4, E8 in v1.5) with Sylvester-Hadamard rotation and per-vector
adaptive q_max, applied as a pure per-vector function so no cross-token
state is needed (streaming out of the box).

Repo: <https://github.com/FluffyAIcode/LLM-KV--Cache-compress>
Paper (v1.4 + v1.5 joint release): `reports/paper/kakeyalattice.pdf` in-repo
(arXiv submission in progress).
License: Apache-2.0.

## Measured results

All numbers are **live vLLM + FlashAttention bf16** on H200,
WikiText-103 prefill, protocol details in `reports/v1_4_release/` and
`reports/v1_5_release/`.

### iso-PPL compression advantage (|Δppl| ≤ 2%, n=8 passages, 512 target tokens)

| Model | KakeyaLattice CR | TurboQuant CR | Advantage |
|---|---|---|---|
| Qwen3-4B | **2.77×** | 2.18× | **+26.9%** |
| GLM-4-9B-Chat | **2.44×** | 1.77× | **+37.8%** |
| Gemma-4-E4B | 3.04× | 3.04× | tied (saturated) |
| DeepSeek-R1-Distill-1.5B | **2.43×** | 2.36× | **+3.3%** |

### iso-bit |Δppl| advantage at aggressive point (Q=10 vs TQ b=4, ~3.6-3.9× CR, n=4)

| Model | KakeyaLattice |Δppl| | TQ |Δppl| | Better by |
|---|---|---|---|
| Qwen3-4B | **1.45%** | 6.58% | **4.5×** |
| GLM-4-9B-Chat | **6.52%** | 10.74% | **1.6×** |
| Gemma-4-E4B | **0.33%** | 1.04% | **3.2×** |
| DeepSeek-R1-Distill-1.5B | **2.22%** | 3.47% | **1.6×** |

### Rigorous n=32 in-forward evaluation (95% CI, no-boundary, v1.5 E8)

E8 reduces |Δppl| by **28–53%** over D4 across three deployable models at
Q∈{4,10}, with **+1.3 to +2.0 dB per-layer K-MSE gain** — 4–6× the +0.29 dB
theoretical shaping-only minimum, because E8's two-coset structure handles
coarse-quantisation outliers better than D4's single parity flip.

### Streaming latency

Per-decode-step codec overhead (1 new token × all layers × all KV heads,
batched): **~0.25 ms** across all 4 models × 3 operating points. At typical
15–30 ms bf16 decode step on H200, codec overhead is **< 2%** of total
decode latency.

### NIAH retrieval (long-context quality check)

- Qwen3-4B at 16k ctx: **100%** recall at Q=10
- Gemma-4-E4B at 16k ctx: **100%** recall at Q=10
- GLM-4-9B-Chat at 16k ctx: **89%** (1 of 27 cells degrades, logged)
- DeepSeek-R1-Distill-1.5B at 16k ctx: **100%** recall at Q=10

## Integration with vLLM

The plugin is a clean `vllm.general_plugins` entry point, no vLLM fork:

```bash
pip install -e kakeyalattice        # pure-Python codec
pip install -e vllm_backend         # registers the plugin entry point
export KAKEYA_SNAPSHOT_QWEN3=1      # env-gated, off by default
vllm serve Qwen/Qwen3-4B
```

It monkey-patches `Attention.forward` on the Qwen3 / Qwen2 / Gemma4 / GLM
families to capture K and V **post-QK-norm / post-V-norm, pre-RoPE**, run
the codec, and write the decoded tensors back before the RoPE+attn step
proceeds. This means:

- ✅ PagedAttention unchanged
- ✅ No changes to block manager or scheduler
- ✅ Works with chunked prefill and prefix caching
- ✅ FlashAttention backend compatible
- ⚠️ Currently **gated behind env vars per model family**, so default vLLM
  behaviour is untouched — users opt in.

## What we'd like feedback on

1. **Plugin interface stability**: the entry-point ABI we're using
   (`vllm.general_plugins`) is what's documented in the plugin docs as of
   v0.10+, but we've seen it churn between minor releases. Is there a
   preferred interface for attention-level codec plugins?
2. **Native paged-block compact storage**: right now we decompress
   per-forward so the KV cache in the paged block is still FP16. Getting
   actual VRAM savings requires storing compressed bytes natively in the
   paged block, the way NexusQuant proposed in #39241. Is there appetite
   for a shared KV-codec abstraction both NexusQuant and KakeyaLattice
   could target?
3. **Attention hook registration**: we currently monkey-patch per-model; is
   there a cleaner point to hook into post-norm/pre-RoPE K/V across model
   families?
4. **Speculative-decoding compatibility**: any known issues with K/V codecs
   under EAGLE / DFlash speculative decoding backends? Our plugin is a pure
   per-vector function so it should compose, but we haven't tested this
   end-to-end yet.

Happy to open a draft PR if the community thinks this is the right shape.

## Related work

- #39241 — NexusQuant (E8 VQ with token eviction, similar motivation but
  different codec structure and eviction strategy)
- #16160 — R-KV cache compression (closed as stale, but similar plugin-level
  integration questions)
- [NestQuant (Savkin et al., ICML 2025)](https://arxiv.org/abs/2502.09720) —
  nested Gosset lattice for W4A4KV4, closest academic precedent
- [KV-Compress (Rehg, 2024)](https://arxiv.org/abs/2410.00161) — paged KV
  eviction with variable per-head rates
