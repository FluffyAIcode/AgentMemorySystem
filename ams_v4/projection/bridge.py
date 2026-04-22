"""EmbBridge4 — thin prefix → backbone bridge.

The prefix channel in v4 is minimal: prepend the (L_mem, d_LLM) prefix tensor
to the token embeddings and extend the attention mask to cover it. No CFG,
no content_bias, no logit shaping — those were v3.46 decode-time patches
for missing upstream structure. v4's upstream is explicit, so we don't
reintroduce them here.
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn

from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


class EmbBridge4(nn.Module):
    """Prefix-prepend bridge."""

    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        self.prefix_post_ln = nn.LayerNorm(cfg.d_LLM)

    def build_inputs(
        self, prefix: Tensor, ids: Tensor, mask: Tensor, wte: nn.Module,
    ) -> Tuple[Tensor, Tensor]:
        """Merge prefix with token embeddings.

        prefix:     (B, L_mem, d_LLM)
        ids:        (B, T)
        mask:       (B, T)  (1 = attend, 0 = pad)
        wte:        backbone word-token embedding module (callable on int ids)

        Returns:
          input_embeds: (B, L_mem + T, d_LLM)
          input_mask:   (B, L_mem + T)
        """
        assert prefix.dim() == 3 and prefix.shape[-1] == self.cfg.d_LLM
        assert prefix.shape[1] == self.cfg.L_mem, (
            f"prefix must have L_mem={self.cfg.L_mem} slots, got {prefix.shape[1]}"
        )
        assert ids.dim() == 2 and mask.dim() == 2
        assert ids.shape[0] == prefix.shape[0] == mask.shape[0]

        tok_emb = wte(ids)                                # (B, T, d_LLM)
        # Cast prefix to backbone dtype for concat
        prefix_n = self.prefix_post_ln(prefix.to(tok_emb.dtype))
        input_embeds = torch.cat([prefix_n, tok_emb], dim=1)

        B = mask.shape[0]
        prefix_mask = torch.ones(
            B, self.cfg.L_mem, dtype=mask.dtype, device=mask.device,
        )
        input_mask = torch.cat([prefix_mask, mask], dim=1)
        return input_embeds, input_mask
