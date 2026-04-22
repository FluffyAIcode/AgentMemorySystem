"""EmbBridge4 — the prefix → backbone injection bridge.

Thin compared to v3.46's EmbBridge: v4's CrossBundleAttention already
returns the prefix in the correct (L_mem, d_LLM) shape. EmbBridge4 handles:

  1. prepending the prefix to the backbone's input embeddings
  2. assembling the matching attention mask + position_ids
  3. optionally running CFG-style double-forward (kept optional to make the
     benchmark gap between A_ams_prefix and D_full_history auditable — with
     CFG off, the prefix channel is isolated cleanly)

No logit shaping, content_bias, strict_overlap gate, or keyword_tail_slot
logic lives here in v4. Those were v3.46 decode-time workarounds for the
lack of explicit bundle axes; v4 fixes the upstream cause and does not
need them.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


class EmbBridge4(nn.Module):
    """Prefix-prepend bridge. Takes a (B, L_mem, d_LLM) prefix and a token
    input (ids, mask) and returns the combined input for the backbone.
    """

    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        # v4.5 implementation:
        #   self.prefix_post_ln = nn.LayerNorm(cfg.d_LLM)  # redundant with CrossBundleAttention's
        #                                                  # but cheap, catches numeric drift
        raise NotImplementedError("v4-skel: EmbBridge4.__init__ — lands in v4.5")

    def build_inputs(self, prefix: Tensor, ids: Tensor, mask: Tensor,
                     wte: nn.Embedding) -> Tuple[Tensor, Tensor]:
        """Merge prefix with token embeddings.

        prefix: (B, L_mem, d_LLM)
        ids:    (B, T)
        mask:   (B, T)
        wte:    the backbone's word-token embedding module

        Returns:
          input_embeds: (B, L_mem + T, d_LLM)
          input_mask:   (B, L_mem + T)

        Position IDs are handled by the caller because they depend on the
        backbone's rotary/relative encoding scheme.
        """
        raise NotImplementedError("v4-skel: EmbBridge4.build_inputs — lands in v4.5")
