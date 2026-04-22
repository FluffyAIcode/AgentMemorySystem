"""Trainer4 — v4 training loop.

See ARCHITECTURE_v4_TRAIN.md. Freezes backbone; trains v4 adapter modules
(bundles, cross_attn). Kakeya registry is rebuilt from the current store
at each step where it becomes inactive, but its skeleton tensors are NOT
gradient-trained.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ams_v4.core.config import Cfg4
from ams_v4.core.mem_store import MemStore
from ams_v4.kakeya.registry import KakeyaRegistry
from ams_v4.training.batch_encode import encode_batch_for_training
from ams_v4.training.losses import (
    loss_bundle_axis_alignment,
    loss_cross_bundle_independence,
    loss_prefix_semantic_anchor,
    loss_recon,
    loss_write_policy,
)


@dataclass
class StepStats:
    step: int
    total: float
    dt_s: float
    components: Dict[str, float] = field(default_factory=dict)


class Trainer4:
    """v4.6 trainer. Separates build / step / save."""

    def __init__(self, model, cfg: Optional[Cfg4] = None, lr: float = 3e-4,
                 weight_decay: float = 0.01, grad_clip: float = 1.0):
        self.model = model
        self.cfg = cfg or model.cfg

        # Sanity: backbone is frozen
        for p in model.backbone.model.parameters():
            assert not p.requires_grad, "backbone params must be frozen"

        # Collect trainable params from v4 adapter modules
        trainable_modules = [
            model.bundle_time, model.bundle_topic, model.bundle_ctx,
            model.cross_attn, model.bridge,
        ]
        trainable_params = []
        for m in trainable_modules:
            for p in m.parameters():
                if p.requires_grad:
                    trainable_params.append(p)
        if not trainable_params:
            raise RuntimeError("no trainable params — did you forget to freeze the backbone?")
        self.trainable_params = trainable_params
        self.n_trainable = sum(p.numel() for p in trainable_params)
        self.opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip

    # ─── Stepping ─────────────────────────────────────────────────────────

    def step(self, batch_texts: List[str]) -> StepStats:
        """Run one training step on a list of text strings.

        Returns StepStats with total loss and per-component losses.
        """
        import time
        t0 = time.time()
        model = self.model

        # Seed a fresh store + registry every step so kakeya rebuild is deterministic
        model.store = MemStore(self.cfg)
        model.kakeya = KakeyaRegistry(self.cfg)
        model._session_summary = None
        # Use the production write() path to populate the store (detached copies)
        # so that the retrieve-side data (DirectionTrees, kakeya) mirrors real inference.
        # Then encode again for gradient purposes.
        for text in batch_texts:
            model.write(text)
        # Now encode gradient-bearing copies for the loss math
        be = encode_batch_for_training(model, batch_texts)

        # ─── Compute all loss components ─────────────────────────────
        loss_map: Dict[str, torch.Tensor] = {
            "prefix_semantic_anchor": loss_prefix_semantic_anchor(model, be),
            "bundle_axis_alignment":  loss_bundle_axis_alignment(model, be),
            "cross_bundle_independence": loss_cross_bundle_independence(model, be),
            "recon":        loss_recon(model, be),
            "write_policy": loss_write_policy(model, be),
        }

        weights = self.cfg.loss_weights
        total = sum(weights.get(k, 1.0) * v for k, v in loss_map.items())

        self.opt.zero_grad()
        total.backward()
        nn.utils.clip_grad_norm_(self.trainable_params, max_norm=self.grad_clip)
        self.opt.step()

        components = {k: float(v.detach().item()) for k, v in loss_map.items()}
        return StepStats(
            step=-1, total=float(total.detach().item()),
            dt_s=time.time() - t0, components=components,
        )

    # ─── Probe / save ─────────────────────────────────────────────────────

    def probe_weights(self) -> Dict[str, float]:
        """Checkpoint-time snapshot of key weight magnitudes."""
        m = self.model
        out: Dict[str, float] = {}
        try:
            out["cross_attn.lift_time[0].w_abs_mean"] = float(
                m.cross_attn.lift_time[0].weight.detach().abs().mean().item()
            )
        except Exception as e:
            out["cross_attn.lift_time[0]"] = f"ERR {type(e).__name__}"
        try:
            out["topic_enc.down_project[0].w_abs_mean"] = float(
                m.bundle_topic.encoder.down_project[0].weight.detach().abs().mean().item()
            )
        except Exception as e:
            out["topic_enc.down_project[0]"] = f"ERR {type(e).__name__}"
        try:
            out["bundle_time._axis_raw_abs_mean"] = float(
                m.bundle_time._axis_raw.detach().abs().mean().item()
            )
        except Exception as e:
            out["bundle_time._axis_raw"] = f"ERR {type(e).__name__}"
        return out

    def save(self, path: str, steps: int, elapsed_s: float,
             pre_probe: Dict[str, float], post_probe: Dict[str, float],
             provenance: str = "AgentMemory/v347-architecture-realign-b7fa") -> None:
        import os
        m = self.model
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        sd = {}
        # Only dump trainable, non-backbone parameters
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            sd[name] = p.detach().cpu()
        blob = {
            "state_dict": sd,
            "cfg_summary": {
                "d_LLM": self.cfg.d_LLM,
                "L_mem": self.cfg.L_mem,
                "d_time": self.cfg.d_time, "d_F_time": self.cfg.d_F_time,
                "d_topic": self.cfg.d_topic, "d_F_topic": self.cfg.d_F_topic,
                "d_ctx": self.cfg.d_ctx, "d_F_ctx": self.cfg.d_F_ctx,
                "n_kakeya_sets": self.cfg.n_kakeya_sets,
            },
            "provenance": provenance,
            "steps": steps,
            "elapsed_s": elapsed_s,
            "pre_probe": pre_probe,
            "post_probe": post_probe,
            "n_trainable": self.n_trainable,
        }
        torch.save(blob, path)
