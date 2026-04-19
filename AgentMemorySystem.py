from scheme_b_v330 import *
import scheme_b_v330 as v330

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_dev = v330._dev
_Node = v330._Node


def _resolve_dtype(name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


@dataclass
class Cfg(v330.Cfg):
    llm_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_dtype: str = "bf16"
    use_chat_template_for_gen: bool = False
    d_LLM: int = 1536
    vocab_size: int = 151936
    degen_early_punct_penalty: float = 6.0
    degen_early_newline_penalty: float = 6.0
    content_bias_scale: float = 4.0
    cfg_scale: float = 2.0
    tail_head_hidden: int = 1024
    late_newline_penalty: float = 20.0
    newline_hard_gate_min_step: int = 12
    newline_hard_gate_min_content: int = 6
    eos_hard_mask_steps: int = 10
    wte_neighbor_max_vocab: int = 60000

    def __post_init__(self):
        super().__post_init__()
        assert self.llm_dtype in ("bf16", "fp16", "fp32")


class LLMBackbone(nn.Module):
    def __init__(self, name: str, dtype_name: str = "bf16"):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self._dtype = _resolve_dtype(dtype_name)
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError(f"Tokenizer for {name} has no pad/eos token")
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        )
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        cfg = self.model.config
        self.d_model = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.n_layers = cfg.num_hidden_layers
        self.has_chat_template = getattr(self.tokenizer, "chat_template", None) is not None
        with torch.no_grad():
            self._wte_fp32 = self.model.get_input_embeddings().weight.detach().float().clone()

    def input_embedding_weight(self) -> torch.Tensor:
        return self._wte_fp32

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(ids)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        return self

    def forward(self, ids: torch.Tensor, attention_mask: torch.Tensor, prefix: Optional[torch.Tensor] = None):
        te = self.embed_tokens(ids)
        if prefix is not None:
            prefix_cast = prefix.to(te.dtype)
            inputs_embeds = torch.cat([prefix_cast, te], dim=1)
            B, P = prefix_cast.shape[:2]
            pm = torch.ones(B, P, device=ids.device, dtype=attention_mask.dtype)
            ext_mask = torch.cat([pm, attention_mask], dim=1)
            pl = P
        else:
            inputs_embeds = te
            ext_mask = attention_mask
            pl = 0
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=ext_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return {
            "logits": out.logits.float(),
            "hs": [h.float() for h in out.hidden_states],
            "pl": pl,
            "mask": ext_mask,
        }

    def build_chat_text(self, user_text: str) -> str:
        if not self.has_chat_template:
            return user_text
        msgs = [{"role": "user", "content": user_text}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class FiberConnection(v330.FiberConnection):
    def __init__(self, d_M, d_F, metric, grad_coupling=True):
        super().__init__(d_M, d_F, metric, grad_coupling=grad_coupling)
        idx = torch.triu_indices(d_M, d_M)
        self.register_buffer('_tri_r', idx[0], persistent=False)
        self.register_buffer('_tri_c', idx[1], persistent=False)

    def forward(self, x, v):
        g = self.metric(x)
        gf = g[:, self._tri_r, self._tri_c]
        if not self.grad_coupling:
            gf = gf.detach()
        raw = self.net(torch.cat([x, v, gf], -1)).reshape(-1, self.d_F, self.d_F)
        return (raw - raw.transpose(1, 2)) / 2


class AMM(v330.AMM):
    def __init__(self, c):
        super().__init__(c)
        self.conn = FiberConnection(c.d_M, c.d_F, self.metric, grad_coupling=True)
        self.trans = v330.FiberTransporter(self.conn, c)


class MemLLM(v330.MemLLM):
    def __init__(self, c):
        super().__init__(c)
        self.amm = AMM(c)
        self.backbone = None

    def load(self, name: Optional[str] = None, dtype_name: Optional[str] = None):
        name = name or self.c.llm_name
        dev = next(self.parameters()).device
        dtype_name = dtype_name or self.c.llm_dtype
        if dev.type == 'mps' and dtype_name == 'bf16':
            dtype_name = 'fp16'
        self.backbone = LLMBackbone(name, dtype_name=dtype_name)
        self.backbone.to(dev)
        self.tok = self.backbone.tokenizer
        self.c.d_LLM = self.backbone.d_model
        self.c.vocab_size = self.backbone.vocab_size
        if self.amm.ctx.f1.in_features != self.c.d_LLM:
            self.amm = AMM(self.c).to(dev)
            self.bridge = v330.EmbBridge(self.c).to(dev)
            self.semantic_probe = v330.PrefixSemanticProbe(self.c.d_LLM, self.c.L_mem, self.c.d_F).to(dev)
            self.vocab_proj = v330.MemoryVocabProjector(self.c.d_F, self.c.d_LLM).to(dev)
        self.layer_pool = v330.AdaptiveLayerPool(self.backbone.n_layers + 1, self.c.d_LLM).to(dev)
        self.content_classifier = v330.ContentTokenClassifier(self.tok, self.c)
        self._degen_guard = v330.DegenerationGuard(self.tok, self.c, self.content_classifier)
        wte_fp32 = self.backbone.input_embedding_weight()
        with torch.no_grad():
            si = min(5000, wte_fp32.shape[0])
            idx = torch.randperm(wte_fp32.shape[0])[:si]
            self.bridge.aligner._target_std.fill_(float(wte_fp32[idx].std().item()))
            self.bridge.aligner._calibrated = True
        self._wte_normed = F.normalize(wte_fp32.detach().cpu(), dim=-1, eps=1e-8)
        self.amm.wte_normed = self._wte_normed
        self._build_wte_neighbor_cache()
        self._compute_filler_centroid()
        return self

    def _build_wte_neighbor_cache(self):
        if self.backbone is None or self.content_classifier is None:
            return
        V = self.backbone.vocab_size
        if V > self.c.wte_neighbor_max_vocab:
            self._wte_neighbor_cache = {}
            return
        wte_n = self._wte_normed
        cc = self.content_classifier
        valid = [t for t in sorted(cc.content_ids) if t < wte_n.shape[0]]
        self._wte_neighbor_cache = {}
        K = self.c.wte_neighbor_k
        thresh = self.c.wte_neighbor_threshold
        batch_size = 500
        for start in range(0, len(valid), batch_size):
            batch_ids = valid[start : start + batch_size]
            batch_t = torch.tensor(batch_ids, device=wte_n.device)
            sims = wte_n[batch_t] @ wte_n.T
            topk_vals, topk_ids = sims.topk(K + 1, dim=-1)
            for i, tid in enumerate(batch_ids):
                neighbors = []
                for score, nid in zip(topk_vals[i], topk_ids[i]):
                    nid_int = int(nid.item())
                    if nid_int == tid:
                        continue
                    if score.item() >= thresh and nid_int in cc.content_ids:
                        neighbors.append(nid_int)
                self._wte_neighbor_cache[tid] = neighbors

    def _expand_content_ids(self, content_ids):
        if not self._wte_neighbor_cache:
            return content_ids
        expanded = set(content_ids)
        for tid in content_ids:
            expanded.update(self._wte_neighbor_cache.get(tid, []))
        return list(expanded)

    def _compute_filler_centroid(self):
        if self.content_classifier is None or self.backbone is None:
            self._filler_centroid = None
            return
        wte = self.backbone.input_embedding_weight()
        valid = [tid for tid in sorted(self.content_classifier.filler_ids) if tid < wte.shape[0]]
        if len(valid) < 3:
            self._filler_centroid = None
            return
        filler_vecs = wte[torch.tensor(valid)]
        centroid = F.normalize(filler_vecs.mean(0), dim=-1, eps=1e-8)
        self._filler_centroid = centroid.to(next(self.parameters()).device)

    def fwd(self, ids, mask, prefix=None):
        return self.backbone(ids, mask, prefix=prefix)


    def forward_logits_only(self, ids, attention_mask, prefix=None):
        te = self.backbone.embed_tokens(ids)
        if prefix is not None:
            prefix_cast = prefix.to(te.dtype)
            inputs_embeds = torch.cat([prefix_cast, te], dim=1)
            B, P = prefix_cast.shape[:2]
            pm = torch.ones(B, P, device=ids.device, dtype=attention_mask.dtype)
            ext_mask = torch.cat([pm, attention_mask], dim=1)
            pl = P
        else:
            inputs_embeds = te
            ext_mask = attention_mask
            pl = 0
        out = self.backbone.model(
            inputs_embeds=inputs_embeds,
            attention_mask=ext_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        return {"logits": out.logits.float(), "pl": pl, "mask": ext_mask}

    def _compute_vocab_bias(self, fiber_summary):
        if fiber_summary is None:
            return None
        with torch.no_grad():
            mem_emb = self.vocab_proj.proj(fiber_summary).float().cpu()
            mem_n = F.normalize(mem_emb, dim=-1, eps=1e-8)
            wte_n = self._wte_normed
            parts = []
            chunk = 8192
            for start in range(0, wte_n.shape[0], chunk):
                parts.append(mem_n @ wte_n[start : start + chunk].T)
            return torch.cat(parts, dim=-1).to(fiber_summary.device)


class Trainer(v330.Trainer):
    def encoder_throughput_loss(self, ids, mask, fiber):
        B = ids.shape[0]
        dev = ids.device
        fiber_unsq = fiber.unsqueeze(1)
        mem_mask_ones = torch.ones(B, 1, device=dev)
        prefix = self.m.bridge.inject(fiber_unsq, mem_mask_ones, fiber_summary=fiber)
        o2 = self.m.forward_logits_only(ids, mask, prefix)
        lg = o2['logits'][:, o2['pl']:-1]
        tg = ids[:, 1:]
        ml = min(lg.shape[1], tg.shape[1])
        if ml == 0:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        return F.cross_entropy(lg[:, :ml].reshape(-1, lg.shape[-1]), tg[:, :ml].reshape(-1))

    def _recon_forward(self, text):
        tk = self.m.tok(text, return_tensors='pt', padding=True, truncation=True)
        dev = next(self.m.parameters()).device
        ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
        with torch.no_grad():
            bo = self.m.fwd(ids, mask)
        prefix = self.m._get_prefix(bo['hs'], mask, update_stats=False, ids=ids)
        o = self.m.forward_logits_only(ids, mask, prefix)
        lg = o['logits'][:, o['pl']:-1]
        tg = ids[:, 1:]
        ml = min(lg.shape[1], tg.shape[1])
        if ml == 0:
            zero = ids.new_tensor(0.0, dtype=torch.float, requires_grad=True)
            return zero, prefix, self.m.bridge._last_fiber_summary
        l_r = F.cross_entropy(lg[:, :ml].reshape(-1, lg.shape[-1]), tg[:, :ml].reshape(-1))
        fs = self.m.bridge._last_fiber_summary
        if fs is None:
            fs = torch.zeros(1, self.c.d_F, device=dev)
        return l_r, prefix, fs

    def semantic_alignment_loss(self, fiber, target_ids, target_mask):
        dev = fiber.device
        wte = self.m.backbone.input_embedding_weight().to(dev)
        vocab_logits = self.m.vocab_proj(fiber, wte)
        B, V = vocab_logits.shape
        cc = self.m.content_classifier
        if cc is None:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        target = torch.zeros(B, V, device=dev)
        valid_count = 0
        for b in range(B):
            valid = target_ids[b][target_mask[b].bool()].tolist()
            content_ids = cc.get_content_ids_from_tokens(valid)
            if content_ids:
                uids = [uid for uid in set(content_ids) if uid < V]
                if uids:
                    target[b, uids] = 1.0 / len(uids)
                    valid_count += 1
        if valid_count == 0:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        log_probs = F.log_softmax(vocab_logits / self.c.semantic_align_temp, dim=-1)
        return F.kl_div(log_probs, target, reduction="none").sum(-1).mean()

    def vocab_anchor_loss(self, prefix):
        dev = prefix.device
        wte = self.m.backbone.input_embedding_weight().to(dev)
        pn = F.normalize(prefix.reshape(-1, prefix.shape[-1]), dim=-1)
        wn = F.normalize(wte, dim=-1)
        sim = pn @ wn.T
        return -sim.topk(self.c.vocab_anchor_topk, dim=-1).values.mean()

    def tail_semantic_anchor_loss(self, fiber, ids, mask):
        if not (self.c.use_content_semantic_tail and self.c.content_tail_slots > 0):
            return torch.tensor(0.0, device=fiber.device, requires_grad=True)
        tail = self.m.bridge.tail_head(fiber)
        if tail is None:
            return torch.tensor(0.0, device=fiber.device, requires_grad=True)
        dev = fiber.device
        wte = self.m.backbone.input_embedding_weight().to(dev)
        B, _, _ = tail.shape
        V = wte.shape[0]
        cc = self.m.content_classifier
        if cc is None:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        losses = []
        tn = F.normalize(tail, dim=-1)
        wn = F.normalize(wte, dim=-1)
        for b in range(B):
            valid = ids[b][mask[b].bool()].tolist()
            content_tids = [t for t in set(cc.get_content_ids_from_tokens(valid)) if t < V]
            if not content_tids:
                continue
            target = torch.zeros(V, device=dev)
            target[content_tids] = 1.0 / len(content_tids)
            slot_logits = tn[b] @ wn.T / 0.3
            log_probs = F.log_softmax(slot_logits, dim=-1)
            losses.append(
                F.kl_div(
                    log_probs,
                    target.unsqueeze(0).expand_as(log_probs),
                    reduction="none",
                ).sum(-1).mean()
            )
        if not losses:
            return torch.tensor(0.0, device=dev, requires_grad=True)
        return torch.stack(losses).mean()
