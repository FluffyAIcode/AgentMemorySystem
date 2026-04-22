"""MemLLM4 — top-level model.

Composes:
  - backbone LLM (Qwen2.5-1.5B-Instruct by default)
  - three bundles (temporal, topic, context)
  - KakeyaRegistry
  - CrossBundleAttention
  - EmbBridge4
  - MemStore (owns entries and per-bundle DirectionTrees)

The public interface intentionally mirrors v3.46 MemLLM at the top level
(`write`, `generate`, `prepare_decode_context`) so session_viability.py can
later swap v3.46 MemLLM for v4 MemLLM4 with a one-line change at the
benchmark site — but the *internal* composition is fully the v4 design.

What this file does NOT contain (things v3.46 MemLLM had that v4 does not):
  - `content_bias_*` logit shaping
  - `strict_overlap_*` retrieval gate
  - `keyword_tail_slot` / `use_top1_exclusive_content_bias`
  - `tail_slot_residual_dominant`
  - `use_functional_suppression` / `decode_fs_*` / `fwd_function_suppression_*`
  - `use_mixture_decoding` / circuit breaker for mixture gate

Each of those was a decode-time patch for an upstream encoding deficit. The
v4 architecture addresses the upstream cause (explicit bundle axes +
kakeya-bundle linkage) and these patches should become unnecessary. If any
turn out to still be needed after v4.5 ships, they are added as an
identifiable subsequent PR with a stated reason — not ported en masse.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from ams_v4.attention.cross_bundle import CrossBundleAttention
from ams_v4.bundles.context import ContextBundle
from ams_v4.bundles.temporal import TemporalBundle
from ams_v4.bundles.topic import TopicBundle
from ams_v4.core.config import Cfg4
from ams_v4.core.mem_store import MemStore
from ams_v4.core.types import Tensor
from ams_v4.kakeya.registry import KakeyaRegistry
from ams_v4.projection.bridge import EmbBridge4


class MemLLM4(nn.Module):
    """Top-level model.

    Usage (after v4.5 implementation lands):
      cfg = Cfg4()
      m = MemLLM4(cfg)
      m.load()                      # load backbone weights
      m.write("some fact")          # encode into all three bundles + kakeya
      out = m.generate("a query", mt=30)
    """

    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        # v4.5 composition:
        #   self.backbone       = LLMBackbone4(cfg)        # thin wrapper over HF AutoModel
        #   self.bundle_time    = TemporalBundle(cfg)
        #   self.bundle_topic   = TopicBundle(cfg)
        #   self.bundle_ctx     = ContextBundle(cfg)
        #   self.kakeya         = KakeyaRegistry(cfg)
        #   self.cross_attn     = CrossBundleAttention(cfg)
        #   self.bridge         = EmbBridge4(cfg)
        #   self.store          = MemStore(cfg)
        raise NotImplementedError("v4-skel: MemLLM4.__init__ — lands in v4.5")

    # ─── v3.46-compatible public surface (for session_viability drop-in) ──

    def load(self, name: Optional[str] = None) -> None:
        """Load the backbone LLM weights."""
        raise NotImplementedError("v4-skel: MemLLM4.load — lands in v4.5")

    def write(self, text: str, training_mode: bool = False) -> Optional[int]:
        """Encode `text` through all three bundles + kakeya, insert into MemStore.

        Returns the new mid if the write-gate accepts, else None.
        """
        raise NotImplementedError("v4-skel: MemLLM4.write — lands in v4.5")

    def prepare_decode_context(self, ids: Tensor, mask: Tensor,
                               update_stats: bool = False):
        """Run per-bundle retrieval + CrossBundleAttention to produce a prefix.

        Returns a DecodeContext4 with the prefix tensor and diagnostics.
        """
        raise NotImplementedError("v4-skel: MemLLM4.prepare_decode_context — lands in v4.5")

    def generate(self, prompt: str, mt: int = 40, greedy: bool = True) -> str:
        """Generate `mt` new tokens conditioned on a memory-derived prefix.

        In v4 there is no CFG double-forward by default (see EmbBridge4 note).
        Set cfg.cfg_scale > 0 to enable it.
        """
        raise NotImplementedError("v4-skel: MemLLM4.generate — lands in v4.5")
