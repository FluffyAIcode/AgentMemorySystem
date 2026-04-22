"""LLMBackbone4 — thin wrapper over HF AutoModelForCausalLM.

Design goals for v4.5:
  - Backbone weights are FROZEN (we do not fine-tune the LM in v4).
  - Expose a `wte` property and `hidden_states(ids, mask)` for encoders.
  - Expose `forward_with_prefix` and `generate_with_prefix` for inference.

No logit shaping, CFG, content_bias, or any v3.46 decode hacks. The prefix
is delivered as prepended `inputs_embeds` and the extended attention_mask.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ams_v4.core.config import Cfg4
from ams_v4.core.types import Tensor


_DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


class LLMBackbone4(nn.Module):
    def __init__(self, cfg: Cfg4):
        super().__init__()
        self.cfg = cfg
        self._loaded = False
        self.tok = None      # HF tokenizer
        self.model = None    # HF model

    # ─── Load ────────────────────────────────────────────────────────────

    def load(self, name: Optional[str] = None, device: Optional[torch.device] = None) -> None:
        """Load the backbone LM. If name is None, uses cfg.llm_name."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = name or self.cfg.llm_name
        dtype = _DTYPE_MAP[self.cfg.llm_dtype]
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except TypeError:
            # Older transformers signature
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.eval()
        # Freeze backbone parameters — v4 does not train the LM
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Default device: use CUDA if available and caller didn't override
        if device is None and torch.cuda.is_available():
            device = torch.device("cuda")
        if device is not None:
            self.model.to(device)
        # Validate hidden_size
        actual_d_LLM = self.model.config.hidden_size
        assert actual_d_LLM == self.cfg.d_LLM, (
            f"Cfg4.d_LLM={self.cfg.d_LLM} but backbone {model_name} has "
            f"hidden_size={actual_d_LLM}; update Cfg4.d_LLM"
        )
        self._loaded = True

    def assert_loaded(self) -> None:
        assert self._loaded and self.model is not None, \
            "LLMBackbone4.load() must be called before use"

    # ─── Accessors ───────────────────────────────────────────────────────

    @property
    def wte(self) -> nn.Module:
        """The word-token embedding module (callable on int ids)."""
        self.assert_loaded()
        return self.model.get_input_embeddings()

    @property
    def device(self) -> torch.device:
        self.assert_loaded()
        return next(self.model.parameters()).device

    def tokenize(self, text: str, return_tensors: str = "pt") -> Tuple[Tensor, Tensor]:
        self.assert_loaded()
        t = self.tok(text, return_tensors=return_tensors)
        ids = t["input_ids"].to(self.device)
        mask = t["attention_mask"].to(self.device)
        return ids, mask

    # ─── Forward helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def hidden_states(self, ids: Tensor, mask: Tensor) -> Tensor:
        """Return last-layer hidden states: (B, T, d_LLM)."""
        self.assert_loaded()
        out = self.model(
            input_ids=ids, attention_mask=mask,
            output_hidden_states=True, use_cache=False,
        )
        # HF returns .hidden_states as a tuple (layers+1, B, T, d_LLM) with the
        # embedding as [0] and each layer following. Last is the final hidden state.
        return out.hidden_states[-1]

    @torch.no_grad()
    def forward_with_prefix(
        self, prefix_embeds: Tensor, prefix_mask: Tensor,
        ids: Tensor, mask: Tensor,
    ) -> Tensor:
        """Run a forward with prepended prefix embeddings.

        Returns logits: (B, L_mem + T, vocab_size).
        """
        self.assert_loaded()
        tok_emb = self.wte(ids)
        input_embeds = torch.cat([prefix_embeds, tok_emb], dim=1)
        attn = torch.cat([prefix_mask, mask], dim=1)
        out = self.model(
            inputs_embeds=input_embeds, attention_mask=attn,
            use_cache=False,
        )
        return out.logits

    @torch.no_grad()
    def generate_with_prefix(
        self, prefix_embeds: Tensor, prefix_mask: Tensor,
        ids: Tensor, mask: Tensor, max_new_tokens: int, greedy: bool = True,
    ) -> Tensor:
        """Greedy generation conditioned on (prefix, ids).

        Returns the full token id sequence including the prompt `ids` portion
        (but not the prefix — prefix is embedding-space, not token-space).
        Output shape: (B, T + N_new).
        """
        self.assert_loaded()
        tok_emb = self.wte(ids)
        cur_embeds = torch.cat([prefix_embeds, tok_emb], dim=1)
        cur_mask = torch.cat([prefix_mask, mask], dim=1)

        gen_ids = ids.clone()
        eos = self.tok.eos_token_id

        # We do a manual loop rather than HF generate() because HF generate()
        # with inputs_embeds has rough edges across versions. One forward per
        # step keeps correctness obvious and is fine for smoke-testing.
        for _ in range(max_new_tokens):
            out = self.model(
                inputs_embeds=cur_embeds, attention_mask=cur_mask, use_cache=False,
            )
            logits = out.logits[:, -1, :]  # (B, vocab)
            if greedy:
                next_tok = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            gen_ids = torch.cat([gen_ids, next_tok], dim=1)
            # Append embed of next token
            next_emb = self.wte(next_tok)
            cur_embeds = torch.cat([cur_embeds, next_emb], dim=1)
            new_mask = torch.ones(
                cur_mask.shape[0], 1, dtype=cur_mask.dtype, device=cur_mask.device,
            )
            cur_mask = torch.cat([cur_mask, new_mask], dim=1)
            # Stop on EOS across whole batch
            if eos is not None and bool((next_tok.squeeze(-1) == eos).all().item()):
                break
        return gen_ids
