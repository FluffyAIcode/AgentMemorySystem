#!/usr/bin/env python3
"""
AMS v3.31 — Multimodal Branch-Local Test Suite
==============================================

This suite aligns multimodal coverage to the `v3.31` public interface.
It uses the real `MemLLM` pipeline and stores image/video-like memories
through `AMM.store_mem()` with embeddings built from the branch's real
input embedding matrix.

Rules:
  - no mocks
  - no fallback logic
  - no source modifications during the run
  - real model + real tokenizer + real retrieval/generation path
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from typing import Iterable, List

import torch
import torch.nn.functional as F

from AgentMemorySystem import MemEntry, _Node
import v331_blackbox_eval as audit


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []

    def check(self, name, cond, msg=""):
        if cond:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {msg}")
            print(f"  ✗ {name}: {msg}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 70}")
        print(f"  {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("  FAILURES:")
            for err in self.errors:
                print(f"    - {err}")
        print(f"{'=' * 70}")
        return self.failed == 0


NATURE_KW = {
    "sunset", "mountain", "ocean", "forest", "river", "landscape",
    "valley", "waterfall", "meadow", "beach", "canyon", "horizon",
}
CITY_KW = {
    "skyscraper", "traffic", "street", "urban", "city", "downtown",
    "highway", "bridge", "subway", "metro", "billboard", "skyline",
}
ANIMAL_KW = {
    "lion", "elephant", "eagle", "dolphin", "tiger", "whale", "wildlife",
    "savanna", "migration", "flock", "herd", "predator",
}
MUSIC_KW = {
    "piano", "chopin", "nocturne", "orchestra", "beethoven", "symphony",
    "harmony", "melody", "music", "musician", "composer", "violin",
}


def _reset(m):
    m.amm.tree.store.clear()
    m.amm.tree.root = _Node()
    m.amm.tree.nid = 0
    m.amm.time = 0


def _dev(m):
    return next(m.parameters()).device


def _input_wte_cpu(m) -> torch.Tensor:
    return m.backbone.input_embedding_weight().detach().float().cpu()


def _concept_embedding(m, words: Iterable[str]) -> torch.Tensor:
    wte = _input_wte_cpu(m)
    ids = []
    for word in words:
        ids.extend(m.tok.encode(" " + word))
    valid = [tid for tid in ids if tid < wte.shape[0]]
    if not valid:
        return torch.zeros(wte.shape[1])
    return wte[valid].mean(0)


def _make_image_embedding(m, visual_concepts: Iterable[str], seed: int = 1000) -> torch.Tensor:
    base = _concept_embedding(m, visual_concepts)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    offset = torch.randn(base.shape, generator=gen) * 0.15
    emb = F.normalize(base + offset, dim=0) * base.norm().clamp(min=1e-8)
    return emb.to(_dev(m))


def _make_video_embedding(m, frame_concepts_list: List[List[str]], seed: int = 2000) -> torch.Tensor:
    frames = []
    for i, concepts in enumerate(frame_concepts_list):
        frames.append(_concept_embedding(m, concepts) * (1.0 + 0.1 * i))
    if not frames:
        return torch.zeros(m.c.d_LLM, device=_dev(m))
    base = torch.stack(frames).mean(0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    offset = torch.randn(base.shape, generator=gen) * 0.2
    emb = F.normalize(base + offset, dim=0) * base.norm().clamp(min=1e-8)
    return emb.to(_dev(m))


def _content_ids_for_words(m, words: Iterable[str]) -> List[int]:
    cc = m.content_classifier
    ids = []
    for word in words:
        for tid in m.tok.encode(" " + word):
            if tid in cc.content_ids:
                ids.append(tid)
    return list(set(ids))


def _store_image_memory(m, visual_concepts: List[str], label: str, surprise: float = 1.5) -> MemEntry:
    h = _make_image_embedding(m, visual_concepts)
    content_ids = _content_ids_for_words(m, visual_concepts)
    return m.amm.store_mem(
        h,
        surprise,
        training_mode=True,
        source_text=f"[IMAGE] {label}",
        content_token_ids=content_ids,
        content_semantic_emb=h.clone(),
        expanded_content_ids=m._expand_content_ids(content_ids),
    )


def _store_video_memory(m, frame_concepts_list: List[List[str]], label: str, surprise: float = 2.0) -> MemEntry:
    flat = [w for frame in frame_concepts_list for w in frame]
    h = _make_video_embedding(m, frame_concepts_list)
    content_ids = _content_ids_for_words(m, flat)
    return m.amm.store_mem(
        h,
        surprise,
        training_mode=True,
        source_text=f"[VIDEO] {label}",
        content_token_ids=content_ids,
        content_semantic_emb=h.clone(),
        expanded_content_ids=m._expand_content_ids(content_ids),
    )


def _content_bias_top_tokens(m, query: str, k: int = 20):
    dev = _dev(m)
    tk = m.tok(query, return_tensors="pt")
    ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        _, _, diag, cb = m._get_prefix(o["hs"], mask, update_stats=False, return_extra=True, ids=ids)
    topk_ids = cb[0].topk(k).indices.tolist()
    toks = [m.tok.decode([tid]).strip().lower() for tid in topk_ids]
    return toks, cb, diag


def test_image_memory_stored(m, R):
    print("\n── M1. Image memory stored ──")
    _reset(m)
    entry = _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature_landscape")
    R.check("m1_entry_created", isinstance(entry, MemEntry))
    R.check("m1_has_image_tag", "[IMAGE]" in entry.source_text)
    R.check("m1_has_content_ids", len(entry.content_token_ids) > 0)
    R.check("m1_has_sem_emb", entry.semantic_emb is not None)
    R.check("m1_has_expanded_ids", len(entry.expanded_content_ids) >= len(entry.content_token_ids))


def test_video_memory_stored(m, R):
    print("\n── M2. Video memory stored ──")
    _reset(m)
    frames = [["lion", "savanna"], ["lion", "predator"], ["lion", "migration"]]
    entry = _store_video_memory(m, frames, "lion_documentary")
    R.check("m2_entry_created", isinstance(entry, MemEntry))
    R.check("m2_has_video_tag", "[VIDEO]" in entry.source_text)
    R.check("m2_has_content_ids", len(entry.content_token_ids) >= 2)
    R.check("m2_has_sem_emb", entry.semantic_emb is not None)


def test_text_image_video_coexist(m, R):
    print("\n── M3. Text/image/video coexist ──")
    _reset(m)
    m.write("The pianist practiced a Chopin nocturne before the concert.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature_landscape")
    _store_video_memory(m, [["lion", "savanna"], ["lion", "predator"]], "lion_documentary")
    tags = [entry.source_text for entry in m.amm.tree.store.values()]
    R.check("m3_has_text", any(not text.startswith("[") for text in tags))
    R.check("m3_has_image", any(text.startswith("[IMAGE]") for text in tags))
    R.check("m3_has_video", any(text.startswith("[VIDEO]") for text in tags))


def test_text_query_retrieves_image_concepts(m, R):
    print("\n── M4. Text query retrieves image concepts ──")
    _reset(m)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature_landscape")
    _store_image_memory(m, ["skyscraper", "traffic", "city"], "city_scene")
    top_toks, _, _ = _content_bias_top_tokens(m, "Describe a mountain sunset over the ocean.")
    nature_hits = sum(tok in NATURE_KW for tok in top_toks)
    city_hits = sum(tok in CITY_KW for tok in top_toks)
    R.check("m4_nature_hits_dominate", nature_hits > city_hits, f"nature={nature_hits}, city={city_hits}, top={top_toks}")


def test_text_query_retrieves_video_concepts(m, R):
    print("\n── M5. Text query retrieves video concepts ──")
    _reset(m)
    _store_video_memory(m, [["lion", "savanna"], ["lion", "predator"]], "lion_documentary")
    _store_video_memory(m, [["skyscraper", "traffic"], ["city", "bridge"]], "city_timelapse")
    top_toks, _, _ = _content_bias_top_tokens(m, "Tell me about a lion moving across the savanna.")
    animal_hits = sum(tok in ANIMAL_KW for tok in top_toks)
    city_hits = sum(tok in CITY_KW for tok in top_toks)
    R.check("m5_animal_hits_dominate", animal_hits > city_hits, f"animal={animal_hits}, city={city_hits}, top={top_toks}")


def test_cross_modal_domain_isolation(m, R):
    print("\n── M6. Cross-modal domain isolation ──")
    _reset(m)
    m.write("The pianist practiced a Chopin nocturne for the orchestra.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature_landscape")
    _store_video_memory(m, [["lion", "savanna"], ["lion", "predator"]], "lion_documentary")
    top_toks, _, _ = _content_bias_top_tokens(m, "Explain piano technique and orchestra phrasing.")
    music_hits = sum(tok in MUSIC_KW for tok in top_toks)
    animal_hits = sum(tok in ANIMAL_KW for tok in top_toks)
    nature_hits = sum(tok in NATURE_KW for tok in top_toks)
    R.check("m6_music_stays_on_top", music_hits > max(animal_hits, nature_hits), f"music={music_hits}, animal={animal_hits}, nature={nature_hits}")


def test_save_load_multimodal(m, R):
    print("\n── M7. Save/load multimodal ──")
    _reset(m)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature_landscape")
    _store_video_memory(m, [["lion", "savanna"], ["lion", "predator"]], "lion_documentary")
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        m.save_memory(path)
        m2 = audit.build_model(77)
        m2.load_memory(path)
        tags = [entry.source_text for entry in m2.amm.tree.store.values()]
        R.check("m7_count_preserved", len(m2.amm.tree.store) == len(m.amm.tree.store))
        R.check("m7_image_tag_preserved", any(text.startswith("[IMAGE]") for text in tags))
        R.check("m7_video_tag_preserved", any(text.startswith("[VIDEO]") for text in tags))
    finally:
        os.unlink(path)


def test_generation_with_multimodal_store(m, R):
    print("\n── M8. Generation with multimodal store ──")
    _reset(m)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature_landscape")
    blank = audit.build_model(91)
    with torch.no_grad():
        out_mem = m.generate("Describe the landscape.", mt=24, greedy=True)
        out_blank = blank.generate("Describe the landscape.", mt=24, greedy=True)
    R.check("m8_generation_nonempty", len(out_mem) > len("Describe the landscape."))
    R.check("m8_memory_changes_output", out_mem != out_blank, f"mem={out_mem!r} blank={out_blank!r}")


def test_multimodal_retrieval_diag(m, R):
    print("\n── M9. Multimodal retrieval diag ──")
    _reset(m)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature_landscape")
    _store_video_memory(m, [["lion", "savanna"], ["lion", "predator"]], "lion_documentary")
    dev = _dev(m)
    tk = m.tok("Describe the mountain sunset.", return_tensors="pt")
    ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        _, _, diag, _ = m._get_prefix(o["hs"], mask, update_stats=False, return_extra=True, ids=ids)
    R.check("m9_diag_has_weights", len(diag.batch_mem_weights) == 1)
    R.check("m9_diag_has_dominant", diag.dominant_memory_id is not None)


def test_multimodal_tree_consistency(m, R):
    print("\n── M10. Multimodal tree consistency ──")
    _reset(m)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature_landscape")
    _store_image_memory(m, ["skyscraper", "traffic", "city"], "city_scene")
    _store_video_memory(m, [["lion", "savanna"], ["lion", "predator"]], "lion_documentary")
    errs = m.amm.tree.verify_consistency()
    R.check("m10_tree_consistent", len(errs) == 0, str(errs))


def main() -> int:
    torch.manual_seed(42)
    R = TestResults()
    sep = "=" * 70
    print(f"\n{sep}")
    print("  AMS v3.31 — Multimodal Branch-Local Test Suite")
    print(f"{sep}")
    t0 = time.time()

    print("\n[Building MemLLM via the v331 audit runner]")
    m = audit.build_model(42)
    total = sum(p.numel() for p in m.parameters())
    train_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  Params: total={total:,}  trainable={train_p:,}  frozen={total-train_p:,}")

    test_image_memory_stored(m, R)
    test_video_memory_stored(m, R)
    test_text_image_video_coexist(m, R)
    test_text_query_retrieves_image_concepts(m, R)
    test_text_query_retrieves_video_concepts(m, R)
    test_cross_modal_domain_isolation(m, R)
    test_save_load_multimodal(m, R)
    test_generation_with_multimodal_store(m, R)
    test_multimodal_retrieval_diag(m, R)
    test_multimodal_tree_consistency(m, R)

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")
    ok = R.summary()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
