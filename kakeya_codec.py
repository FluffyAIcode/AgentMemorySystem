#!/usr/bin/env python3
"""
KakeyaCodec — Kakeya-like Set Compression for AMS v3.12
=========================================================

Transparent compression layer for MemEntry's 768-dim semantic_emb field.
Wraps MemLLM without modifying AgentMemorySystem.py.

v3.12 adaptation:
  - Only semantic_emb remains as 768-dim field (content_wte_centroid removed)
  - Single skeleton for semantic_emb compression
  - Compatible with v3.12's forward_maxsim retrieval scoring

Construction:
  1. Global PCA: R^768 → R^d_eff (retain 99% variance)
  2. Temporal direction separation: coeff → (α scalar, perp vector)
  3. Spherical K-means on perp directions → K segment centers (Kakeya skeleton)
  4. Each memory encoded as (seg_id, α, t, sparse_residual)
  5. Decode: reconstruct approximate 768-dim vector on demand
"""

import torch
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CompressedVec:
    """Compressed representation of a single 768-dim vector."""
    seg_id: int
    alpha: float
    t: float
    residual_vals: torch.Tensor
    residual_idx: torch.Tensor


@dataclass
class KakeyaSkeleton:
    """The Kakeya-like skeleton: basis + temporal direction + segment centers."""
    basis: torch.Tensor           # [d_eff, d_LLM]
    mean: torch.Tensor            # [d_LLM]
    t_dir: torch.Tensor           # [d_eff]
    centers: torch.Tensor         # [K, d_eff]
    d_eff: int
    K: int
    d_res: int


class KakeyaCodec:
    """
    Kakeya-like set compression codec for 768-dim semantic_emb vectors.
    """

    def __init__(self, d_LLM: int = 768, variance_ratio: float = 0.99,
                 K: int = 16, d_res: int = 5, min_entries_to_build: int = 8):
        self.d_LLM = d_LLM
        self.variance_ratio = variance_ratio
        self.K = K
        self.d_res = d_res
        self.min_entries = min_entries_to_build

        self.sem_skeleton: Optional[KakeyaSkeleton] = None
        self.sem_compressed: Dict[int, CompressedVec] = {}

        self._is_active = False
        self._stats = {
            'total_encoded': 0,
            'total_decoded': 0,
            'skeleton_builds': 0,
        }

    @property
    def is_active(self) -> bool:
        return self._is_active

    def _compute_pca(self, vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        mu = vecs.mean(0)
        centered = vecs - mu.unsqueeze(0)
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        cumvar = S.pow(2).cumsum(0) / S.pow(2).sum()
        d_eff_arr = (cumvar >= self.variance_ratio).nonzero(as_tuple=True)[0]
        d_eff = (d_eff_arr[0].item() + 1) if len(d_eff_arr) > 0 else len(S)
        d_eff = max(d_eff, 2)
        basis = Vh[:d_eff]
        return basis, mu, d_eff

    def _spherical_kmeans(self, dirs: torch.Tensor, K: int,
                          max_iter: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        N, d = dirs.shape
        K = min(K, N)
        if K <= 1:
            return dirs[:1].clone(), torch.zeros(N, dtype=torch.long, device=dirs.device)
        centers = [dirs[0].clone()]
        for _ in range(K - 1):
            sims = torch.stack([dirs @ c for c in centers], dim=1)
            max_sim = sims.max(dim=1)[0]
            farthest = max_sim.argmin()
            centers.append(dirs[farthest].clone())
        centers = torch.stack(centers)
        assignments = torch.zeros(N, dtype=torch.long, device=dirs.device)
        for _ in range(max_iter):
            sims = dirs @ centers.T
            new_assign = sims.argmax(dim=1)
            if (new_assign == assignments).all():
                break
            assignments = new_assign
            for k in range(K):
                mask = assignments == k
                if mask.any():
                    centers[k] = F.normalize(dirs[mask].mean(0), dim=0, eps=1e-8)
                else:
                    far = (dirs @ centers.T).max(1)[0].argmin()
                    centers[k] = dirs[far].clone()
                    assignments[far] = k
        return centers, assignments

    def _build_skeleton(self, vecs: torch.Tensor) -> KakeyaSkeleton:
        basis, mu, d_eff = self._compute_pca(vecs)
        coeffs = (vecs - mu.unsqueeze(0)) @ basis.T
        mu_coeff = coeffs.mean(0)
        mu_norm = mu_coeff.norm()
        if mu_norm > 1e-8:
            t_dir = mu_coeff / mu_norm
        else:
            t_dir = torch.zeros(d_eff, device=vecs.device)
            t_dir[0] = 1.0
        alpha = coeffs @ t_dir
        perp = coeffs - alpha.unsqueeze(-1) * t_dir.unsqueeze(0)
        perp_norms = perp.norm(dim=-1)
        valid_mask = perp_norms > 1e-8
        if valid_mask.sum() >= 2:
            perp_dirs = F.normalize(perp[valid_mask], dim=-1)
            K_actual = min(self.K, perp_dirs.shape[0])
            centers, _ = self._spherical_kmeans(perp_dirs, K_actual)
        else:
            centers = F.normalize(torch.randn(1, d_eff, device=vecs.device), dim=-1)
            K_actual = 1
        return KakeyaSkeleton(
            basis=basis, mean=mu, t_dir=t_dir,
            centers=centers, d_eff=d_eff, K=K_actual, d_res=self.d_res)

    def build(self, store: dict):
        sem_vecs = []
        mids_sem = []
        for mid, entry in store.items():
            if entry.semantic_emb is not None:
                sem_vecs.append(entry.semantic_emb)
                mids_sem.append(mid)
        if len(sem_vecs) >= self.min_entries:
            sem_mat = torch.stack(sem_vecs)
            self.sem_skeleton = self._build_skeleton(sem_mat)
            self.sem_compressed.clear()
            for i, mid in enumerate(mids_sem):
                self.sem_compressed[mid] = self._encode_vec(
                    sem_vecs[i], self.sem_skeleton)
        self._is_active = self.sem_skeleton is not None
        self._stats['skeleton_builds'] += 1

    def _encode_vec(self, vec: torch.Tensor, skel: KakeyaSkeleton) -> CompressedVec:
        coeff = (vec - skel.mean) @ skel.basis.T
        alpha = (coeff @ skel.t_dir).item()
        perp = coeff - alpha * skel.t_dir
        perp_norm = perp.norm()
        if perp_norm > 1e-8:
            perp_dir = perp / perp_norm
            sims = skel.centers @ perp_dir
            seg_id = sims.argmax().item()
        else:
            seg_id = 0
        t = (perp @ skel.centers[seg_id]).item()
        residual = perp - t * skel.centers[seg_id]
        d_res = min(skel.d_res, skel.d_eff)
        if d_res < skel.d_eff:
            _, top_idx = residual.abs().topk(d_res)
            r_vals = residual[top_idx]
        else:
            top_idx = torch.arange(skel.d_eff, device=vec.device)
            r_vals = residual
        self._stats['total_encoded'] += 1
        return CompressedVec(
            seg_id=seg_id, alpha=alpha, t=t,
            residual_vals=r_vals.detach().cpu(),
            residual_idx=top_idx.detach().cpu())

    def _decode_vec(self, comp: CompressedVec, skel: KakeyaSkeleton,
                    device: torch.device) -> torch.Tensor:
        residual = torch.zeros(skel.d_eff, device=device)
        idx = comp.residual_idx.to(device)
        vals = comp.residual_vals.to(device)
        residual[idx] = vals
        perp_approx = comp.t * skel.centers[comp.seg_id].to(device) + residual
        coeff_approx = comp.alpha * skel.t_dir.to(device) + perp_approx
        vec_approx = coeff_approx @ skel.basis.to(device) + skel.mean.to(device)
        self._stats['total_decoded'] += 1
        return vec_approx

    def encode_entry(self, mid: int, semantic_emb: Optional[torch.Tensor]):
        if self.sem_skeleton is not None and semantic_emb is not None:
            self.sem_compressed[mid] = self._encode_vec(semantic_emb, self.sem_skeleton)

    def decode_sem(self, mid: int, device: torch.device) -> Optional[torch.Tensor]:
        if mid in self.sem_compressed and self.sem_skeleton is not None:
            return self._decode_vec(self.sem_compressed[mid], self.sem_skeleton, device)
        return None

    def remove_entry(self, mid: int):
        self.sem_compressed.pop(mid, None)

    def get_stats(self) -> dict:
        sem_entries = len(self.sem_compressed)
        original_bytes = 0
        compressed_bytes = 0
        if self.sem_skeleton is not None:
            sk = self.sem_skeleton
            original_bytes += sem_entries * self.d_LLM * 4
            basis_bytes = sk.d_eff * self.d_LLM * 4
            mean_bytes = self.d_LLM * 4
            tdir_bytes = sk.d_eff * 4
            centers_bytes = sk.K * sk.d_eff * 4
            per_entry = 4 + 4 + 4 + sk.d_res * 4 + sk.d_res * 4
            compressed_bytes += basis_bytes + mean_bytes + tdir_bytes + centers_bytes
            compressed_bytes += sem_entries * per_entry
        return {
            'is_active': self._is_active,
            'sem_entries': sem_entries,
            'sem_d_eff': self.sem_skeleton.d_eff if self.sem_skeleton else 0,
            'sem_K': self.sem_skeleton.K if self.sem_skeleton else 0,
            'original_bytes': original_bytes,
            'compressed_bytes': compressed_bytes,
            'compression_ratio': original_bytes / max(compressed_bytes, 1),
            'skeleton_builds': self._stats['skeleton_builds'],
            'total_encoded': self._stats['total_encoded'],
            'total_decoded': self._stats['total_decoded'],
        }

    def save(self, path: str):
        torch.save({
            'sem_skeleton': self.sem_skeleton,
            'sem_compressed': self.sem_compressed,
            'config': {
                'd_LLM': self.d_LLM, 'variance_ratio': self.variance_ratio,
                'K': self.K, 'd_res': self.d_res, 'min_entries': self.min_entries,
            },
            'stats': self._stats,
        }, path)

    def load(self, path: str):
        data = torch.load(path, weights_only=False)
        self.sem_skeleton = data['sem_skeleton']
        self.sem_compressed = data['sem_compressed']
        cfg = data['config']
        self.d_LLM = cfg['d_LLM']
        self.variance_ratio = cfg['variance_ratio']
        self.K = cfg['K']
        self.d_res = cfg['d_res']
        self.min_entries = cfg['min_entries']
        self._stats = data.get('stats', self._stats)
        self._is_active = self.sem_skeleton is not None


class KakeyaMemLLM:
    """
    Wrapper around MemLLM that transparently applies Kakeya compression
    on semantic_emb. Exposes identical public interface to MemLLM.

    v3.12 compatible: only compresses semantic_emb (no content_wte_centroid).
    """

    def __init__(self, mem_llm, codec: Optional[KakeyaCodec] = None,
                 auto_build_threshold: int = 8):
        self._m = mem_llm
        self._codec = codec or KakeyaCodec()
        self._auto_threshold = auto_build_threshold

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(self._m, name)

    @property
    def codec(self) -> KakeyaCodec:
        return self._codec

    def _maybe_build_codec(self):
        store = self._m.amm.tree.store
        if len(store) >= self._auto_threshold:
            self._codec.build(store)
            if self._codec.is_active:
                self._compress_all()

    def _compress_all(self):
        store = self._m.amm.tree.store
        for mid, entry in store.items():
            if entry.semantic_emb is not None and mid not in self._codec.sem_compressed:
                self._codec.encode_entry(mid, entry.semantic_emb)

    def _decompress_entry(self, entry):
        dev = next(self._m.parameters()).device
        mid = entry.mid
        if entry.semantic_emb is None and mid in self._codec.sem_compressed:
            entry.semantic_emb = self._codec.decode_sem(mid, dev)

    def _decompress_all(self):
        for mid, entry in self._m.amm.tree.store.items():
            self._decompress_entry(entry)

    def _release_originals(self):
        if not self._codec.is_active:
            return
        for mid, entry in self._m.amm.tree.store.items():
            if mid in self._codec.sem_compressed:
                entry.semantic_emb = None

    # ─── MemLLM Public Interface ─────────────────────────────────

    def load(self, name="gpt2"):
        self._m.load(name)

    def write(self, text, training_mode=False):
        if self._codec.is_active:
            self._decompress_all()
        result = self._m.write(text, training_mode=training_mode)
        if len(self._m.amm.tree.store) >= self._auto_threshold:
            if not self._codec.is_active:
                self._maybe_build_codec()
            else:
                for mid, entry in self._m.amm.tree.store.items():
                    if entry.semantic_emb is not None and mid not in self._codec.sem_compressed:
                        self._codec.encode_entry(mid, entry.semantic_emb)
                self._release_originals()
        return result

    def generate(self, prompt, mt=50, greedy=False):
        if self._codec.is_active:
            self._decompress_all()
        try:
            return self._m.generate(prompt, mt=mt, greedy=greedy)
        finally:
            if self._codec.is_active:
                self._release_originals()

    def fwd(self, ids, mask, prefix=None):
        return self._m.fwd(ids, mask, prefix)

    def extract_state(self, hs, mask=None, pl=0):
        return self._m.extract_state(hs, mask, pl)

    def _get_prefix(self, *args, **kwargs):
        if self._codec.is_active:
            self._decompress_all()
        try:
            return self._m._get_prefix(*args, **kwargs)
        finally:
            if self._codec.is_active:
                self._release_originals()

    def _compute_vocab_bias(self, fiber_summary):
        return self._m._compute_vocab_bias(fiber_summary)

    def _build_content_bias(self, *args, **kwargs):
        return self._m._build_content_bias(*args, **kwargs)

    def _compute_content_semantic_emb(self, *args, **kwargs):
        return self._m._compute_content_semantic_emb(*args, **kwargs)

    def _compute_content_wte_mean(self, *args, **kwargs):
        return self._m._compute_content_wte_mean(*args, **kwargs)

    def _expand_content_ids(self, *args, **kwargs):
        return self._m._expand_content_ids(*args, **kwargs)

    def _refresh_all_memories(self):
        if self._codec.is_active:
            self._decompress_all()
        result = self._m._refresh_all_memories()
        if self._codec.is_active:
            self._codec.sem_compressed.clear()
            self._codec.sem_skeleton = None
            self._codec._is_active = False
            self._maybe_build_codec()
        return result

    def save_memory(self, path):
        if self._codec.is_active:
            self._decompress_all()
        self._m.save_memory(path)
        if self._codec.is_active:
            self._release_originals()
            codec_path = path + '.kakeya'
            self._codec.save(codec_path)

    def load_memory(self, path):
        self._m.load_memory(path)
        codec_path = path + '.kakeya'
        import os
        if os.path.exists(codec_path):
            self._codec.load(codec_path)
        elif len(self._m.amm.tree.store) >= self._auto_threshold:
            self._maybe_build_codec()

    def train(self, mode=True):
        return self._m.train(mode)

    def eval(self):
        return self._m.eval()

    def zero_grad(self):
        return self._m.zero_grad()

    def parameters(self, recurse=True):
        return self._m.parameters(recurse)

    def named_parameters(self, prefix='', recurse=True):
        return self._m.named_parameters(prefix, recurse)

    def state_dict(self, *args, **kwargs):
        return self._m.state_dict(*args, **kwargs)
