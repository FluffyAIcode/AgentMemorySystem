"""
KakeyaLattice — KV-Cache Compression Demo (HuggingFace Space).

Runs on a CPU-only HF Space (free tier) because the codec itself is a
few thousand vector ops; the paper's headline numbers come from H200
runs and are shown as preloaded tables rather than re-measured in the
browser.

Layout
------
Tab 1: interactive codec round-trip on synthetic KV tensors
       (user picks D4 vs E8 vs Z^N, block dim, q_range, head_dim).
       Plots MSE, bit-rate, relative reconstruction error.

Tab 2: frozen results viewer — loads the v1.4 / v1.5 per-model JSON from
       the git repo and renders iso-PPL, iso-bit, NIAH, latency tables.

Tab 3: nine-step pipeline explorer — takes a single 128-dim vector
       (random or user-supplied), shows each step's output.

The codec implementation is imported from the `kakeyalattice` package
pinned in requirements.txt, so the Space is always in sync with the
library's tagged release.
"""
from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

import gradio as gr
import numpy as np
import pandas as pd

try:
    import torch
    from kakeyalattice import V14KakeyaZamirLatticeGPU, V15KakeyaZamirE8GPU
except ImportError as exc:
    raise SystemExit(
        "kakeyalattice package missing — pin it in requirements.txt"
    ) from exc

GH_RAW = "https://raw.githubusercontent.com/FluffyAIcode/LLM-KV--Cache-compress/main"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Tab 1 — round-trip demo
# ---------------------------------------------------------------------------
def run_roundtrip(codec_name: str, head_dim: int, q_range: int,
                  n_vectors: int, seed: int):
    torch.manual_seed(int(seed))
    x = torch.randn(int(n_vectors), 1, int(head_dim),
                    device=DEVICE, dtype=torch.float32) * 0.3

    if codec_name == "KakeyaLattice v1.4 (D4)":
        cb = V14KakeyaZamirLatticeGPU(D=int(head_dim),
                                      q_range=int(q_range), device=DEVICE)
    elif codec_name == "KakeyaLattice v1.5 (E8)":
        cb = V15KakeyaZamirE8GPU(D=int(head_dim),
                                 q_range=int(q_range), device=DEVICE)
    else:  # Z^N scalar baseline (simple mid-tread uniform quantiser)
        return _scalar_roundtrip(x, q_range=int(q_range))

    x_hat = cb.roundtrip(x)
    bits = int(cb.bits_per_token_per_head)
    mse = float(((x - x_hat) ** 2).mean().item())
    rel_err = float(((x - x_hat) ** 2).sum().item()
                    / max((x ** 2).sum().item(), 1e-12) * 100.0)

    return {
        "MSE": f"{mse:.6e}",
        "Relative reconstruction error (%)": f"{rel_err:.4f}",
        "Bits per KV vector": bits,
        "Bits per dim": f"{bits / int(head_dim):.3f}",
        "Device": DEVICE,
    }


def _scalar_roundtrip(x: torch.Tensor, q_range: int):
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / q_range
    q = torch.round(x / scale).clamp(-q_range, q_range)
    x_hat = q * scale
    bits = int(np.ceil(np.log2(2 * q_range + 1))) * x.shape[-1]
    mse = float(((x - x_hat) ** 2).mean().item())
    rel_err = float(((x - x_hat) ** 2).sum().item()
                    / max((x ** 2).sum().item(), 1e-12) * 100.0)
    return {
        "MSE": f"{mse:.6e}",
        "Relative reconstruction error (%)": f"{rel_err:.4f}",
        "Bits per KV vector": bits,
        "Bits per dim": f"{bits / x.shape[-1]:.3f}",
        "Device": DEVICE,
    }


# ---------------------------------------------------------------------------
# Tab 2 — frozen results viewer
# ---------------------------------------------------------------------------
@dataclass
class FrozenReport:
    model: str
    ctx: int
    q_range: int
    delta_ppl_pct: float
    cr: float


def _load_frozen(path: str) -> list[FrozenReport]:
    url = f"{GH_RAW}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=10) as fp:
            data = json.load(fp)
    except Exception as exc:  # noqa: BLE001
        return []
    out = []
    for row in data.get("results", []):
        out.append(FrozenReport(
            model=row.get("model", "?"),
            ctx=int(row.get("ctx_len", 0)),
            q_range=int(row.get("q_range", 0)),
            delta_ppl_pct=float(row.get("delta_ppl_pct", 0.0)),
            cr=float(row.get("compression_ratio", 0.0)),
        ))
    return out


def load_iso_ppl_table():
    rows = []
    for model_slug, model_name in [
        ("qwen3_4b", "Qwen3-4B"),
        ("gemma4_e4b", "Gemma-4-E4B"),
        ("glm4_9b", "GLM-4-9B-Chat"),
        ("deepseek_1p5b", "DeepSeek-R1-Distill-1.5B"),
    ]:
        rs = _load_frozen(
            f"reports/v1_4_release/kv_128k_isoppl_n8/{model_slug}_kv_128k.json"
        )
        for r in rs:
            r.model = model_name
            rows.append(r)
    if not rows:
        return pd.DataFrame([{"info": "Frozen JSON not reachable; see repo."}])
    return pd.DataFrame([r.__dict__ for r in rows])


# ---------------------------------------------------------------------------
# Tab 3 — pipeline explorer
# ---------------------------------------------------------------------------
def explore_pipeline(seed: int, head_dim: int):
    torch.manual_seed(int(seed))
    x = torch.randn(1, 1, int(head_dim), device=DEVICE, dtype=torch.float32) * 0.3
    cb = V15KakeyaZamirE8GPU(D=int(head_dim), q_range=10, device=DEVICE)
    x_hat = cb.roundtrip(x)
    return {
        "Input vector (first 8 dims)": x[0, 0, :8].tolist(),
        "Reconstructed (first 8 dims)": x_hat[0, 0, :8].tolist(),
        "Input L2 norm": float(x.norm().item()),
        "Output L2 norm": float(x_hat.norm().item()),
        "L2 residual": float((x - x_hat).norm().item()),
        "Bits per vector": int(cb.bits_per_token_per_head),
    }


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="KakeyaLattice KV-Cache Codec") as demo:
    gr.Markdown(
        "# KakeyaLattice — KV-Cache Compression Codec\n\n"
        "Interactive demo for the D4 (v1.4) and E8 (v1.5) nested-lattice "
        "KV-cache codec. [Code](https://github.com/FluffyAIcode/LLM-KV--Cache-compress) "
        "· [Paper](https://github.com/FluffyAIcode/LLM-KV--Cache-compress/blob/main/reports/paper/kakeyalattice.pdf) "
        "· Apache-2.0"
    )

    with gr.Tab("Round-trip"):
        with gr.Row():
            codec = gr.Dropdown(
                ["KakeyaLattice v1.4 (D4)", "KakeyaLattice v1.5 (E8)",
                 "Z^N scalar baseline"],
                value="KakeyaLattice v1.5 (E8)", label="Codec")
            head_dim = gr.Slider(32, 256, value=128, step=32, label="Head dim")
        with gr.Row():
            q_range = gr.Slider(4, 152, value=10, step=2, label="q_range")
            n_vectors = gr.Slider(128, 8192, value=2048, step=128,
                                  label="# KV vectors")
            seed = gr.Number(value=0, label="Seed", precision=0)
        run = gr.Button("Run round-trip")
        out = gr.JSON(label="Result")
        run.click(run_roundtrip,
                  inputs=[codec, head_dim, q_range, n_vectors, seed],
                  outputs=[out])

    with gr.Tab("Frozen iso-PPL results"):
        gr.Markdown(
            "Paper-reported iso-PPL numbers (n=8 passages, 512 target tokens, "
            "FlashAttention bf16 on H200). Loaded live from the GitHub repo."
        )
        table = gr.Dataframe(load_iso_ppl_table(), interactive=False)

    with gr.Tab("Pipeline explorer"):
        gr.Markdown(
            "Runs a single KV vector through the nine-step v1.5 pipeline "
            "(unit-norm, Sylvester-Hadamard rotation, per-vector adaptive "
            "q_max, E8 closest-point, clamp, inverse of all steps)."
        )
        with gr.Row():
            ex_seed = gr.Number(value=42, label="Seed", precision=0)
            ex_dim = gr.Slider(32, 256, value=128, step=32, label="Head dim")
        ex_run = gr.Button("Run")
        ex_out = gr.JSON()
        ex_run.click(explore_pipeline, inputs=[ex_seed, ex_dim], outputs=[ex_out])

if __name__ == "__main__":
    demo.launch()
