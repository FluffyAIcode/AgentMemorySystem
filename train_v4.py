#!/usr/bin/env python3
"""Training driver for AMS v4 (v4.6).

Uses the same 9-sentence rotating corpus design as v3.46's train_v346.py,
with v4's five loss terms. Writes ckpt/v4_trained.pt.

Usage:
  python3 train_v4.py --steps 60 --out ckpt/v4_trained.pt
  python3 train_v4.py --steps 20 --batch 3 --llm-name distilgpt2  # smaller debug run
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ams_v4 import Cfg4, MemLLM4
from ams_v4.training.trainer import Trainer4


MUSIC = [
    "He practiced piano for hours perfecting a difficult Chopin nocturne.",
    "She studied music theory and harmonic progression at the conservatory.",
    "The orchestra performed Beethoven symphony with remarkable precision.",
]
SPACE = [
    "The telescope revealed distant galaxies beyond the Milky Way.",
    "Astronauts trained for the Mars mission in simulated zero gravity.",
    "The nebula emitted radiation across the electromagnetic spectrum.",
]
GENERIC = [
    "The pianist practiced arpeggios and Chopin nocturnes until midnight.",
    "A musician refined finger technique, phrasing, and pedal control.",
    "Classical interpretation often depends on dynamics, tempo rubato, and touch.",
    "A conservatory student studied etudes, scales, and expressive keyboard skills.",
    "Distant astronomers observed galaxies quasars and stellar evolution.",
    "Space orbital mechanics explains satellites and planetary motion.",
]
ALL = MUSIC + SPACE + GENERIC


def _build_cfg(llm_name: str) -> Cfg4:
    from transformers import AutoConfig
    ac = AutoConfig.from_pretrained(llm_name)
    return Cfg4(
        llm_name=llm_name,
        d_LLM=ac.hidden_size,
        vocab_size=ac.vocab_size,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--out", type=str, default="ckpt/v4_trained.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=str, default="ckpt/v4_train_log.jsonl")
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    torch.manual_seed(args.seed)

    cfg = _build_cfg(args.llm_name)
    model = MemLLM4(cfg)
    model.load()

    dev = model.backbone.device
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[build] device={dev}  params total={n_params:,}  trainable={n_train:,}")
    assert dev.type == "cuda" or os.environ.get("AMS_ALLOW_CPU_TRAIN") == "1", (
        "train_v4 expects CUDA. Set AMS_ALLOW_CPU_TRAIN=1 to force CPU."
    )

    trainer = Trainer4(model, cfg, lr=args.lr)

    pre_probe = trainer.probe_weights()
    print(f"[probe pre-train] {pre_probe}")

    t_start = time.time()
    log_f = open(args.log, "w")
    try:
        for step in range(args.steps):
            start = (step * args.batch) % len(ALL)
            batch = [ALL[(start + i) % len(ALL)] for i in range(args.batch)]
            stats = trainer.step(batch)
            stats.step = step
            print(
                f"step {step:3d}  total={stats.total:.4f}  "
                f"psa={stats.components.get('prefix_semantic_anchor', 0):.3f}  "
                f"baa={stats.components.get('bundle_axis_alignment', 0):.3f}  "
                f"cbi={stats.components.get('cross_bundle_independence', 0):.3f}  "
                f"rec={stats.components.get('recon', 0):.4f}  "
                f"wp={stats.components.get('write_policy', 0):.3f}  "
                f"dt={stats.dt_s:.1f}s"
            )
            log_f.write(json.dumps({
                "step": step, "total": stats.total, "dt_s": stats.dt_s,
                **stats.components,
            }) + "\n")
            log_f.flush()
    finally:
        log_f.close()

    elapsed = time.time() - t_start
    post_probe = trainer.probe_weights()
    print(f"[probe post-train] {post_probe}")
    print(f"[train] elapsed {elapsed:.1f}s  avg/step={elapsed/max(1, args.steps):.2f}s")

    trainer.save(args.out, steps=args.steps, elapsed_s=elapsed,
                 pre_probe=pre_probe, post_probe=post_probe)
    print(f"[save] wrote {args.out}")


if __name__ == "__main__":
    main()
