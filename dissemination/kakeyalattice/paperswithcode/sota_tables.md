# Pre-filled PwC leaderboard rows

Copy these Markdown tables into the PwC benchmark pages after creating
them. Each cell corresponds 1-to-1 with a form field in PwC's
"Add Result" dialog.

## Benchmark: KV Cache Compression on WikiText-103 (iso-PPL, |Δppl| ≤ 2%)

| Method | Model | CR | Hardware | Protocol |
|---|---|---|---|---|
| **KakeyaLattice v1.4 (D4)** | Qwen/Qwen3-4B | **2.77×** | NVIDIA H200 | snapshot, n=8, 512 tokens |
| **KakeyaLattice v1.4 (D4)** | zai-org/GLM-4-9B-Chat | **2.44×** | NVIDIA H200 | snapshot, n=8, 512 tokens |
| **KakeyaLattice v1.4 (D4)** | google/gemma-4-E4B | **3.04×** | NVIDIA H200 | snapshot, n=8, 512 tokens |
| **KakeyaLattice v1.4 (D4)** | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | **2.43×** | NVIDIA H200 | snapshot, n=8, 512 tokens |
| TurboQuant b=4 | Qwen/Qwen3-4B | 2.18× | NVIDIA H200 | snapshot, n=8, 512 tokens |
| TurboQuant b=4 | zai-org/GLM-4-9B-Chat | 1.77× | NVIDIA H200 | snapshot, n=8, 512 tokens |
| TurboQuant b=4 | google/gemma-4-E4B | 3.04× | NVIDIA H200 | snapshot, n=8, 512 tokens |
| TurboQuant b=4 | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 2.36× | NVIDIA H200 | snapshot, n=8, 512 tokens |

## Benchmark: KV Cache Compression on WikiText-103 (iso-bit, Q=10 / b=4)

| Method | Model | |Δppl| | CR | Hardware |
|---|---|---|---|---|
| **KakeyaLattice v1.4 (D4)** | Qwen/Qwen3-4B | **1.45%** | 3.85× | NVIDIA H200 |
| **KakeyaLattice v1.4 (D4)** | zai-org/GLM-4-9B-Chat | **6.52%** | 3.85× | NVIDIA H200 |
| **KakeyaLattice v1.4 (D4)** | google/gemma-4-E4B | **0.33%** | 3.85× | NVIDIA H200 |
| **KakeyaLattice v1.4 (D4)** | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | **2.22%** | 3.85× | NVIDIA H200 |
| TurboQuant b=4 | Qwen/Qwen3-4B | 6.58% | 3.90× | NVIDIA H200 |
| TurboQuant b=4 | zai-org/GLM-4-9B-Chat | 10.74% | 3.90× | NVIDIA H200 |
| TurboQuant b=4 | google/gemma-4-E4B | 1.04% | 3.90× | NVIDIA H200 |
| TurboQuant b=4 | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | 3.47% | 3.90× | NVIDIA H200 |

## Benchmark: KV Cache Compression (in-forward rigorous, n=32, 95% CI)

| Method | Model | K-MSE gain vs v1.4 | |Δppl| reduction vs v1.4 | Hardware |
|---|---|---|---|---|
| **KakeyaLattice v1.5 (E8)** | Qwen/Qwen3-4B @ Q=10 | **+1.8 dB** | **−31.5%** | NVIDIA H200 |
| **KakeyaLattice v1.5 (E8)** | Qwen/Qwen3-4B @ Q=4 | **+2.0 dB** | **−53.4%** | NVIDIA H200 |
| **KakeyaLattice v1.5 (E8)** | google/gemma-4-E4B @ Q=10 | +1.3 dB | −28% | NVIDIA H200 |
| **KakeyaLattice v1.5 (E8)** | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B @ Q=10 | +1.5 dB | −30% | NVIDIA H200 |

## Benchmark: Needle In A Haystack @ 16k context

| Method | Model | Retrieval recall |
|---|---|---|
| **KakeyaLattice v1.5 (E8) Q=10** | Qwen/Qwen3-4B | **100%** |
| **KakeyaLattice v1.5 (E8) Q=10** | google/gemma-4-E4B | **100%** |
| **KakeyaLattice v1.5 (E8) Q=10** | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B | **100%** |
| **KakeyaLattice v1.5 (E8) Q=10** | zai-org/GLM-4-9B-Chat | 89% (1 of 27 cells) |
| Full FP16 KV | all | 100% (baseline) |
