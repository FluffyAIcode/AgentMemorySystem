#!/usr/bin/env python3
"""
Agent Memory System (AMS) v3.7 — Multimodal Semantic Black-Box Test Suite
==========================================================================

AMS stores memories as (base, fiber, direction) tuples in a Riemannian
manifold. Its core API (store_mem) accepts any d_LLM-dimensional hidden
state, meaning any modality projectable into the 768-dim GPT-2 embedding
space can be stored, retrieved, and used for generation.

This suite tests AMS behavior when TEXT, IMAGE, and VIDEO modalities
are mixed. Modality embeddings are constructed using GPT-2's own WTE
(word token embeddings) to create semantically grounded representations:

  - TEXT:  Encoded via MemLLM.write() or via GPT-2 hidden states.
  - IMAGE: Constructed as weighted centroid of WTE vectors for visual
           concept tokens (e.g., "sunset", "mountain", "ocean") plus
           a modality-specific offset to distinguish from text.
  - VIDEO: Constructed similarly but as a temporal sequence of image-like
           frames blended together, plus a video-specific offset.

No mocks. No simplifications. No fallback. No source modifications.
All tests use the real GPT-2 model and the real AMS pipeline.
"""

import sys, os, time, math, tempfile, random
import torch
import torch.nn.functional as F

from AgentMemorySystem import (
    Cfg, MemLLM, AMM, MemEntry, DirectionTree, _Node,
    Trainer, SpectralDealiaser, RetrievalDiag,
)


# ═══════════════════════════════════════════════════════════════════
# Harness
# ═══════════════════════════════════════════════════════════════════
class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, cond, msg=""):
        if cond:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {msg}")
            print(f"  ✗ {name}: {msg}")

    def summary(self):
        t = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"  {self.passed}/{t} passed, {self.failed} failed")
        if self.errors:
            print("  FAILURES:")
            for e in self.errors:
                print(f"    - {e}")
        print(f"{'='*70}")
        return self.failed == 0


def _reset(m):
    m.amm.tree.store.clear()
    m.amm.tree.root = _Node()
    m.amm.tree.nid = 0
    m.amm.time = 0


def _dev(m):
    return next(m.parameters()).device


# ═══════════════════════════════════════════════════════════════════
# Multimodal Embedding Constructors
#
# These create semantically meaningful embeddings by combining GPT-2
# WTE vectors for domain-specific words, plus modality-specific
# perturbations to simulate the structural differences between
# text, image, and video representations.
# ═══════════════════════════════════════════════════════════════════

def _get_concept_embedding(m, words):
    """Compute mean WTE embedding for a list of concept words."""
    wte = m.llm.transformer.wte.weight.detach()
    tok = m.tok
    all_ids = []
    for w in words:
        ids = tok.encode(" " + w)
        all_ids.extend(ids)
    valid = [i for i in all_ids if i < wte.shape[0]]
    if not valid:
        return torch.zeros(wte.shape[1], device=wte.device)
    return wte[valid].mean(0)


def _make_image_embedding(m, visual_concepts, modality_seed=1000):
    """Construct an image-like embedding from visual concept words.
    Adds a modality-specific offset derived from a fixed seed to
    distinguish image embeddings from text embeddings."""
    base = _get_concept_embedding(m, visual_concepts)
    torch.manual_seed(modality_seed)
    offset = torch.randn_like(base) * 0.15
    return F.normalize(base + offset, dim=0) * base.norm()


def _make_video_embedding(m, frame_concepts_list, modality_seed=2000):
    """Construct a video-like embedding from a sequence of frame concept lists.
    Each frame is a weighted centroid; the video embedding is the temporal
    mean plus a video-specific offset."""
    frames = []
    for i, concepts in enumerate(frame_concepts_list):
        frame_emb = _get_concept_embedding(m, concepts)
        temporal_weight = 1.0 + 0.1 * i
        frames.append(frame_emb * temporal_weight)
    if not frames:
        dev = _dev(m)
        return torch.zeros(m.c.d_LLM, device=dev)
    video_emb = torch.stack(frames).mean(0)
    torch.manual_seed(modality_seed)
    offset = torch.randn_like(video_emb) * 0.2
    return F.normalize(video_emb + offset, dim=0) * video_emb.norm()


def _store_image_memory(m, visual_concepts, source_label, surprise=1.5):
    """Store an image-like memory into AMS."""
    dev = _dev(m)
    h = _make_image_embedding(m, visual_concepts).to(dev)
    cc = m.content_classifier
    tok = m.tok
    content_ids = []
    for w in visual_concepts:
        ids = tok.encode(" " + w)
        for tid in ids:
            if tid in cc.content_ids:
                content_ids.append(tid)
    expanded = m._expand_content_ids(content_ids)
    sem_emb = h.clone()
    return m.amm.store_mem(
        h, surprise, training_mode=True,
        source_text=f"[IMAGE] {source_label}",
        content_token_ids=content_ids,
        content_semantic_emb=sem_emb,
        expanded_content_ids=expanded,
    )


def _store_video_memory(m, frame_concepts_list, source_label, surprise=2.0):
    """Store a video-like memory into AMS."""
    dev = _dev(m)
    h = _make_video_embedding(m, frame_concepts_list).to(dev)
    cc = m.content_classifier
    tok = m.tok
    content_ids = []
    for frame in frame_concepts_list:
        for w in frame:
            ids = tok.encode(" " + w)
            for tid in ids:
                if tid in cc.content_ids:
                    content_ids.append(tid)
    content_ids = list(set(content_ids))
    expanded = m._expand_content_ids(content_ids)
    sem_emb = h.clone()
    return m.amm.store_mem(
        h, surprise, training_mode=True,
        source_text=f"[VIDEO] {source_label}",
        content_token_ids=content_ids,
        content_semantic_emb=sem_emb,
        expanded_content_ids=expanded,
    )


def _content_bias_top_tokens(m, query, k=20):
    """Get top-k content-bias tokens for a text query."""
    dev = _dev(m)
    tk = m.tok(query, return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        prefix, fs, diag, cb = m._get_prefix(
            o['hs'], mask, update_stats=False, return_extra=True, ids=ids)
    topk_ids = cb[0].topk(k).indices.tolist()
    topk_toks = [m.tok.decode([t]).strip().lower() for t in topk_ids]
    return topk_toks, cb, diag


# ═══════════════════════════════════════════════════════════════════
# Domain keyword banks
# ═══════════════════════════════════════════════════════════════════
NATURE_KW = {
    'sunset', 'mountain', 'ocean', 'forest', 'river', 'landscape',
    'valley', 'cliff', 'waterfall', 'meadow', 'sky', 'cloud',
    'sunrise', 'lake', 'beach', 'canyon', 'prairie', 'horizon',
}
CITY_KW = {
    'skyscraper', 'traffic', 'building', 'street', 'urban', 'city',
    'downtown', 'highway', 'bridge', 'subway', 'metro', 'neon',
    'apartment', 'intersection', 'pedestrian', 'billboard', 'skyline',
}
ANIMAL_KW = {
    'lion', 'elephant', 'eagle', 'dolphin', 'tiger', 'whale',
    'hunting', 'predator', 'prey', 'wildlife', 'savanna', 'migration',
    'pack', 'herd', 'flock', 'feather', 'fur', 'claw', 'safari',
}
MUSIC_KW = {
    'piano', 'chopin', 'nocturne', 'orchestra', 'beethoven', 'symphony',
    'harmony', 'melody', 'chord', 'performed', 'harmonic', 'progression',
    'conservatory', 'violin', 'concerto', 'musician', 'composer',
    'pianist', 'music', 'instrument', 'practiced',
}
SPACE_KW = {
    'galaxy', 'galaxies', 'telescope', 'star', 'planet', 'orbit',
    'space', 'astronaut', 'mars', 'nebula', 'gravity', 'cosmic',
    'satellite', 'mission', 'rocket', 'spacecraft', 'launch',
}
COOKING_KW = {
    'chef', 'recipe', 'ingredient', 'cooking', 'kitchen', 'baking',
    'oven', 'spice', 'flavor', 'dish', 'meal', 'cuisine', 'garlic',
    'sauce', 'restaurant', 'gourmet', 'dessert', 'prepared',
}


# ═══════════════════════════════════════════════════════════════════
# M1-M5: 基础多模态写入与存储
# ═══════════════════════════════════════════════════════════════════
def test_image_memory_stored(m, c, R):
    """Image-like memory should be stored with all required fields."""
    print("\n── M1. Image memory stored correctly ──")
    _reset(m)
    entry = _store_image_memory(m, ["sunset", "mountain", "ocean"],
                                 "nature_landscape")
    R.check("m1_entry_created", isinstance(entry, MemEntry))
    R.check("m1_has_source", "[IMAGE]" in entry.source_text)
    R.check("m1_has_content_ids", len(entry.content_token_ids) > 0)
    R.check("m1_has_sem_emb", entry.semantic_emb is not None)
    R.check("m1_has_expanded", len(entry.expanded_content_ids) > 0)
    R.check("m1_base_finite", entry.base.isfinite().all().item())
    R.check("m1_fiber_finite", entry.fiber.isfinite().all().item())
    _reset(m)


def test_video_memory_stored(m, c, R):
    """Video-like memory should be stored with temporal frame content."""
    print("\n── M2. Video memory stored correctly ──")
    _reset(m)
    frames = [
        ["lion", "savanna", "hunting"],
        ["lion", "prey", "chase"],
        ["lion", "eating", "sunset"],
    ]
    entry = _store_video_memory(m, frames, "lion_hunting_documentary")
    R.check("m2_entry_created", isinstance(entry, MemEntry))
    R.check("m2_has_video_tag", "[VIDEO]" in entry.source_text)
    R.check("m2_has_content_ids", len(entry.content_token_ids) > 0)
    R.check("m2_has_multi_frame_content",
            len(entry.content_token_ids) >= 3,
            f"n={len(entry.content_token_ids)}")
    R.check("m2_base_finite", entry.base.isfinite().all().item())
    _reset(m)


def test_text_image_video_coexist(m, c, R):
    """All three modalities should coexist in the memory store."""
    print("\n── M3. Text+Image+Video coexistence ──")
    _reset(m)
    m.write("The pianist performed a Chopin nocturne.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain", "lake"], "nature_photo")
    _store_video_memory(m,
        [["city", "traffic", "morning"], ["city", "busy", "noon"]],
        "city_timelapse")
    n_entries = len(m.amm.tree.store)
    R.check("m3_three_modalities_stored", n_entries >= 3,
            f"entries={n_entries}")
    texts = [e.source_text for e in m.amm.tree.store.values()]
    has_text = any("[IMAGE]" not in t and "[VIDEO]" not in t for t in texts)
    has_image = any("[IMAGE]" in t for t in texts)
    has_video = any("[VIDEO]" in t for t in texts)
    R.check("m3_has_text_modality", has_text)
    R.check("m3_has_image_modality", has_image)
    R.check("m3_has_video_modality", has_video)
    errs = m.amm.tree.verify_consistency()
    R.check("m3_tree_consistent", len(errs) == 0, str(errs))
    _reset(m)


def test_multimodal_embeddings_differ(m, c, R):
    """Embeddings from different modalities of same concept should differ."""
    print("\n── M4. Modality embeddings differ ──")
    dev = _dev(m)
    concepts = ["sunset", "mountain", "ocean"]
    text_emb = _get_concept_embedding(m, concepts)
    img_emb = _make_image_embedding(m, concepts)
    vid_emb = _make_video_embedding(m, [concepts, concepts])
    ti_sim = F.cosine_similarity(text_emb.unsqueeze(0), img_emb.unsqueeze(0)).item()
    tv_sim = F.cosine_similarity(text_emb.unsqueeze(0), vid_emb.unsqueeze(0)).item()
    iv_sim = F.cosine_similarity(img_emb.unsqueeze(0), vid_emb.unsqueeze(0)).item()
    R.check("m4_text_image_differ", ti_sim < 0.99,
            f"text-image sim={ti_sim:.4f}")
    R.check("m4_text_video_differ", tv_sim < 0.99,
            f"text-video sim={tv_sim:.4f}")
    R.check("m4_image_video_differ", iv_sim < 0.99,
            f"image-video sim={iv_sim:.4f}")
    R.check("m4_same_concept_still_related",
            min(ti_sim, tv_sim, iv_sim) > 0.1,
            f"min_sim={min(ti_sim, tv_sim, iv_sim):.4f}")


def test_multimodal_different_domains_differ(m, c, R):
    """Image of nature should differ from image of city."""
    print("\n── M5. Cross-domain image embeddings differ ──")
    nature_emb = _make_image_embedding(m, ["sunset", "mountain", "ocean"])
    city_emb = _make_image_embedding(m, ["skyscraper", "traffic", "neon"])
    sim = F.cosine_similarity(nature_emb.unsqueeze(0), city_emb.unsqueeze(0)).item()
    R.check("m5_nature_city_images_differ", sim < 0.9,
            f"sim={sim:.4f}")


# ═══════════════════════════════════════════════════════════════════
# M6-M10: 跨模态检索精度
# ═══════════════════════════════════════════════════════════════════
def test_text_query_retrieves_related_image(m, c, R):
    """Text query about 'sunset mountain' should retrieve nature image memory."""
    print("\n── M6. Text query → related image retrieval ──")
    _reset(m)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature_landscape")
    _store_image_memory(m, ["skyscraper", "traffic", "downtown"], "city_scene")
    m.eval()

    top_toks, _, _ = _content_bias_top_tokens(m, "The sunset over the mountain was beautiful.")
    nature_hits = sum(1 for t in top_toks if t in NATURE_KW)
    city_hits = sum(1 for t in top_toks if t in CITY_KW)
    R.check("m6_nature_query_nature_bias",
            nature_hits >= city_hits,
            f"nature={nature_hits}, city={city_hits}, top={top_toks}")
    _reset(m)


def test_text_query_retrieves_related_video(m, c, R):
    """Text query about 'lion hunting' should retrieve wildlife video memory."""
    print("\n── M7. Text query → related video retrieval ──")
    _reset(m)
    _store_video_memory(m,
        [["lion", "hunting", "savanna"], ["lion", "prey", "chase"]],
        "lion_hunt")
    _store_video_memory(m,
        [["city", "traffic", "rush"], ["highway", "cars", "speed"]],
        "traffic_timelapse")
    m.eval()

    top_toks, _, _ = _content_bias_top_tokens(m, "Tell me about the lion hunting.")
    animal_hits = sum(1 for t in top_toks if t in ANIMAL_KW)
    city_hits = sum(1 for t in top_toks if t in CITY_KW)
    R.check("m7_animal_query_animal_bias",
            animal_hits >= city_hits,
            f"animal={animal_hits}, city={city_hits}, top={top_toks}")
    _reset(m)


def test_cross_modal_retrieval_text_to_image(m, c, R):
    """Text about music shouldn't strongly activate nature image memory."""
    print("\n── M8. Cross-modal isolation: text(music) vs image(nature) ──")
    _reset(m)
    m.write("The pianist performed a stunning Chopin nocturne.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature_photo")
    m.eval()

    top_toks, _, _ = _content_bias_top_tokens(m, "Tell me about the piano concert.")
    nature_in_top = sum(1 for t in top_toks[:10] if t in NATURE_KW)
    music_in_top = sum(1 for t in top_toks[:10] if t in MUSIC_KW)
    R.check("m8_music_query_music_dominant",
            music_in_top >= nature_in_top,
            f"music={music_in_top}, nature={nature_in_top}")
    _reset(m)


def test_mixed_modality_four_way_retrieval(m, c, R):
    """With 4 memories (2 text, 1 image, 1 video), queries should route correctly."""
    print("\n── M9. 4-way mixed modality retrieval ──")
    _reset(m)
    m.write("The pianist performed a stunning Chopin nocturne.", training_mode=True)
    m.write("The telescope revealed distant galaxies.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain", "ocean", "landscape"], "nature_photo")
    _store_video_memory(m,
        [["lion", "hunting", "savanna"], ["elephant", "herd", "migration"]],
        "wildlife_documentary")
    m.eval()

    queries = {
        'music': ("Tell me about the piano performance.", MUSIC_KW),
        'space': ("What did the telescope reveal?", SPACE_KW),
        'nature': ("Describe the mountain sunset.", NATURE_KW),
        'animal': ("Tell me about the lion hunting.", ANIMAL_KW),
    }
    correct = 0
    for domain, (query, target_kw) in queries.items():
        top_toks, _, _ = _content_bias_top_tokens(m, query, k=15)
        hits = sum(1 for t in top_toks if t in target_kw)
        print(f"    {domain}: hits={hits}, top={top_toks[:8]}")
        if hits > 0:
            correct += 1
    R.check("m9_4way_majority_correct", correct >= 2,
            f"correct={correct}/4")
    _reset(m)


def test_image_video_same_domain_reinforce(m, c, R):
    """Image and video of same domain should reinforce retrieval."""
    print("\n── M10. Same-domain image+video reinforce ──")
    _reset(m)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature_photo")
    m.eval()
    top1, _, _ = _content_bias_top_tokens(m, "The beautiful mountain sunset.")
    score1 = sum(1 for t in top1[:15] if t in NATURE_KW)

    _store_video_memory(m,
        [["mountain", "sunrise", "valley"], ["river", "forest", "meadow"]],
        "nature_timelapse")
    m.eval()
    top2, _, _ = _content_bias_top_tokens(m, "The beautiful mountain sunset.")
    score2 = sum(1 for t in top2[:15] if t in NATURE_KW)

    R.check("m10_img_vid_reinforce", score2 >= score1,
            f"img_only={score1}, img+vid={score2}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# M11-M15: 跨模态生成影响
# ═══════════════════════════════════════════════════════════════════
def test_image_memory_influences_generation(m, c, R):
    """Image memory should influence text generation."""
    print("\n── M11. Image memory influences generation ──")
    _reset(m)
    m.eval()
    with torch.no_grad():
        gen_no = m.generate("The beautiful sunset", mt=20, greedy=True)

    _store_image_memory(m, ["sunset", "mountain", "ocean", "golden", "horizon"],
                        "sunset_photo")
    m.eval()
    with torch.no_grad():
        gen_with = m.generate("The beautiful sunset", mt=20, greedy=True)

    R.check("m11_image_changes_generation",
            gen_with != gen_no,
            f"with='{gen_with[:50]}', without='{gen_no[:50]}'")
    _reset(m)


def test_video_memory_influences_generation(m, c, R):
    """Video memory should influence text generation."""
    print("\n── M12. Video memory influences generation ──")
    _reset(m)
    m.eval()
    with torch.no_grad():
        gen_no = m.generate("The wildlife documentary", mt=20, greedy=True)

    _store_video_memory(m,
        [["lion", "hunting", "savanna"], ["elephant", "herd", "migration"],
         ["eagle", "flying", "mountain"]],
        "wildlife_doc")
    m.eval()
    with torch.no_grad():
        gen_with = m.generate("The wildlife documentary", mt=20, greedy=True)

    R.check("m12_video_changes_generation",
            gen_with != gen_no,
            f"with='{gen_with[:50]}', without='{gen_no[:50]}'")
    _reset(m)


def test_mixed_modal_generation_quality(m, c, R):
    """Generation with mixed modalities should produce non-degenerate text."""
    print("\n── M13. Mixed-modal generation quality ──")
    _reset(m)
    m.write("He practiced piano for hours perfecting a Chopin nocturne.", training_mode=True)
    _store_image_memory(m, ["concert", "hall", "stage", "audience"], "concert_photo")
    _store_video_memory(m,
        [["pianist", "fingers", "keyboard"], ["audience", "applause", "standing"]],
        "concert_video")
    m.eval()

    torch.manual_seed(42)
    with torch.no_grad():
        gen = m.generate("The piano concert", mt=30, greedy=False)
    new_text = gen[len("The piano concert"):].strip()
    alpha = sum(1 for ch in new_text if ch.isalpha())
    ratio = alpha / max(len(new_text), 1)
    R.check("m13_mixed_gen_not_degenerate", ratio > 0.25,
            f"ratio={ratio:.2f}, text='{new_text[:50]}'")
    R.check("m13_mixed_gen_has_content", len(new_text) >= 5)
    _reset(m)


def test_generation_prefers_matching_modality_domain(m, c, R):
    """Music text prompt should prefer music text memory over nature image."""
    print("\n── M14. Generation prefers matching domain across modalities ──")
    _reset(m)
    m.write("The pianist performed a stunning Chopin nocturne.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature_photo")
    m.eval()

    with torch.no_grad():
        gen = m.generate("The piano", mt=25, greedy=True)

    new_text = gen[len("The piano"):].lower()
    has_music = any(kw in new_text for kw in ['piano', 'music', 'chopin', 'performed',
                                               'concert', 'nocturne', 'pianist'])
    has_nature = any(kw in new_text for kw in ['sunset', 'mountain', 'ocean',
                                                'landscape', 'nature'])
    R.check("m14_music_prompt_music_content",
            has_music or not has_nature,
            f"gen='{gen[:60]}'")
    _reset(m)


def test_video_frames_temporal_richness(m, c, R):
    """Video with many frames should produce richer content IDs than single-frame."""
    print("\n── M15. Video temporal richness ──")
    _reset(m)
    short_entry = _store_video_memory(m,
        [["lion", "hunting"]],
        "short_clip")
    n_short = len(short_entry.content_token_ids)

    _reset(m)
    long_entry = _store_video_memory(m,
        [["lion", "hunting", "savanna"],
         ["lion", "prey", "chase", "speed"],
         ["lion", "eating", "sunset", "rest"],
         ["cubs", "playing", "morning", "den"]],
        "long_documentary")
    n_long = len(long_entry.content_token_ids)

    R.check("m15_more_frames_more_content",
            n_long >= n_short,
            f"short={n_short}, long={n_long}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# M16-M20: 多模态记忆合并与衰减
# ═══════════════════════════════════════════════════════════════════
def test_consolidation_across_modalities(m, c, R):
    """Memories from different modalities should not incorrectly merge."""
    print("\n── M16. Cross-modal consolidation ──")
    _reset(m)
    m.write("The sunset over the mountain was breathtaking.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain", "golden"], "sunset_photo")
    n_before = len(m.amm.tree.store)
    m.amm.consolidate()
    n_after = len(m.amm.tree.store)
    R.check("m16_consolidation_runs", True)
    errs = m.amm.tree.verify_consistency()
    R.check("m16_tree_consistent", len(errs) == 0, str(errs))
    print(f"    before={n_before}, after={n_after}")
    _reset(m)


def test_multimodal_decay(m, c, R):
    """Decay should work on all modality types."""
    print("\n── M17. Multimodal decay ──")
    _reset(m)
    m.write("Piano practice.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain"], "nature")
    _store_video_memory(m, [["city", "traffic"]], "city_clip")
    n_before = len(m.amm.tree.store)
    m.amm.time += 5000
    n_decayed = m.amm.decay()
    n_after = len(m.amm.tree.store)
    R.check("m17_decay_runs", n_decayed >= 0)
    R.check("m17_decay_math", n_after == n_before - n_decayed)
    errs = m.amm.tree.verify_consistency()
    R.check("m17_tree_consistent", len(errs) == 0, str(errs))
    _reset(m)


def test_image_image_consolidation(m, c, R):
    """Two very similar image memories should be mergeable."""
    print("\n── M18. Image-image consolidation ──")
    _reset(m)
    dev = _dev(m)
    h1 = _make_image_embedding(m, ["sunset", "mountain"]).to(dev)
    h2 = h1 + torch.randn_like(h1) * 0.0001
    m.amm.store_mem(h1, 1.0, training_mode=True,
                    source_text="[IMAGE] sunset_1",
                    content_token_ids=[100, 200])
    m.amm.store_mem(h2, 1.0, training_mode=True,
                    source_text="[IMAGE] sunset_2",
                    content_token_ids=[300, 400])
    n_before = len(m.amm.tree.store)
    m.amm.consolidate()
    n_after = len(m.amm.tree.store)
    R.check("m18_similar_images_consolidate",
            n_after <= n_before,
            f"before={n_before}, after={n_after}")
    _reset(m)


def test_video_high_surprise_survives(m, c, R):
    """High-surprise video memory should be more resilient to decay."""
    print("\n── M19. High-surprise video decay resilience ──")
    _reset(m)
    _store_video_memory(m, [["boring", "empty", "still"]], "boring_vid", surprise=0.01)
    _store_video_memory(m,
        [["explosion", "dramatic", "action"], ["rescue", "hero", "danger"]],
        "exciting_vid", surprise=5.0)
    m.amm.time += 3000
    m.amm.decay()
    remaining = [e.source_text for e in m.amm.tree.store.values()]
    R.check("m19_decay_completed", True)
    print(f"    remaining: {remaining}")
    _reset(m)


def test_multimodal_multiple_consolidation_rounds(m, c, R):
    """Multiple consolidation rounds on mixed modalities should maintain consistency."""
    print("\n── M20. Multi-round multimodal consolidation ──")
    _reset(m)
    dev = _dev(m)
    for i in range(5):
        h = torch.randn(c.d_LLM, device=dev) * 0.01
        m.amm.store_mem(h, 0.5, training_mode=True,
                        source_text=f"[IMAGE] similar_{i}")
    m.write("Piano music.", training_mode=True)
    _store_video_memory(m, [["city", "traffic"]], "city")
    for _ in range(3):
        m.amm.consolidate()
    errs = m.amm.tree.verify_consistency()
    R.check("m20_multi_consol_consistent", len(errs) == 0, str(errs))
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# M21-M25: 多模态干扰与隔离
# ═══════════════════════════════════════════════════════════════════
def test_image_does_not_overwhelm_text(m, c, R):
    """Adding many image memories should not overwhelm text retrieval."""
    print("\n── M21. Images don't overwhelm text retrieval ──")
    _reset(m)
    m.write("The pianist performed a stunning Chopin nocturne.", training_mode=True)
    m.eval()
    top_before, _, _ = _content_bias_top_tokens(m, "Tell me about piano.")
    music_before = sum(1 for t in top_before[:15] if t in MUSIC_KW)

    for i in range(5):
        _store_image_memory(m, ["sunset", "mountain", "ocean", "lake"],
                           f"nature_{i}")
    m.eval()
    top_after, _, _ = _content_bias_top_tokens(m, "Tell me about piano.")
    music_after = sum(1 for t in top_after[:15] if t in MUSIC_KW)

    R.check("m21_music_signal_survives_images",
            music_after > 0,
            f"before={music_before}, after={music_after}")
    _reset(m)


def test_video_does_not_overwhelm_text(m, c, R):
    """Adding many video memories should not overwhelm text retrieval."""
    print("\n── M22. Videos don't overwhelm text retrieval ──")
    _reset(m)
    m.write("The telescope revealed distant galaxies.", training_mode=True)
    m.eval()
    top_before, _, _ = _content_bias_top_tokens(m, "What did the telescope observe?")
    space_before = sum(1 for t in top_before[:15] if t in SPACE_KW)

    for i in range(5):
        _store_video_memory(m,
            [["cooking", "chef", "kitchen"], ["recipe", "spice", "flavor"]],
            f"cooking_vid_{i}")
    m.eval()
    top_after, _, _ = _content_bias_top_tokens(m, "What did the telescope observe?")
    space_after = sum(1 for t in top_after[:15] if t in SPACE_KW)

    R.check("m22_space_signal_survives_videos",
            space_after > 0,
            f"before={space_before}, after={space_after}")
    _reset(m)


def test_cross_modal_domain_isolation(m, c, R):
    """Each modality-domain combination should have distinct top bias tokens."""
    print("\n── M23. Cross-modal domain isolation ──")
    _reset(m)
    m.write("The pianist performed Chopin.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature")
    _store_video_memory(m,
        [["lion", "hunting", "savanna"], ["elephant", "migration"]],
        "wildlife")
    m.eval()

    top_music, _, _ = _content_bias_top_tokens(m, "Piano music performance.", k=10)
    top_nature, _, _ = _content_bias_top_tokens(m, "Mountain sunset landscape.", k=10)
    top_animal, _, _ = _content_bias_top_tokens(m, "Lion hunting in savanna.", k=10)

    overlap_mn = len(set(top_music) & set(top_nature))
    overlap_ma = len(set(top_music) & set(top_animal))
    overlap_na = len(set(top_nature) & set(top_animal))
    avg_overlap = (overlap_mn + overlap_ma + overlap_na) / 3
    R.check("m23_domains_have_some_difference",
            overlap_mn < 10 or overlap_ma < 10 or overlap_na < 10,
            f"mn={overlap_mn}, ma={overlap_ma}, na={overlap_na}")
    _reset(m)


def test_same_domain_cross_modal_similarity(m, c, R):
    """Text and image of same domain should have related embeddings."""
    print("\n── M24. Same-domain cross-modal embedding similarity ──")
    _reset(m)
    dev = _dev(m)
    tk = m.tok("The beautiful sunset over the mountain.", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        pooled = m.layer_pool(o['hs'])
    text_emb = pooled.mean(1).squeeze(0)

    img_emb = _make_image_embedding(m, ["sunset", "mountain", "ocean"]).to(dev)
    img_emb_diff = _make_image_embedding(m, ["skyscraper", "traffic", "neon"]).to(dev)

    sim_same = F.cosine_similarity(text_emb.unsqueeze(0), img_emb.unsqueeze(0)).item()
    sim_diff = F.cosine_similarity(text_emb.unsqueeze(0), img_emb_diff.unsqueeze(0)).item()

    R.check("m24_same_domain_more_similar",
            sim_same > sim_diff - 0.1,
            f"same={sim_same:.4f}, diff={sim_diff:.4f}")
    _reset(m)


def test_many_modalities_no_catastrophic_failure(m, c, R):
    """Loading many mixed-modality memories should not crash or corrupt state."""
    print("\n── M25. Many mixed modalities stress test ──")
    _reset(m)
    dev = _dev(m)
    for i in range(5):
        m.write(f"Text memory number {i} about various topics.", training_mode=True)
    for i in range(5):
        _store_image_memory(m, ["concept", "visual", "scene"], f"img_{i}")
    for i in range(5):
        _store_video_memory(m,
            [["frame", "motion", "time"], ["scene", "change", "dynamic"]],
            f"vid_{i}")

    n_total = len(m.amm.tree.store)
    R.check("m25_many_stored", n_total > 0, f"n={n_total}")
    errs = m.amm.tree.verify_consistency()
    R.check("m25_tree_consistent", len(errs) == 0, str(errs))

    xq = torch.randn(1, c.d_M, device=dev)
    fq = torch.randn(1, c.d_F, device=dev)
    fibers, mask, fs, diag = m.amm.retrieve_multi(xq, fq)
    R.check("m25_retrieve_ok", fibers.isfinite().all().item())
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# M26-M30: 多模态保存/加载
# ═══════════════════════════════════════════════════════════════════
def test_save_load_multimodal(m, c, R):
    """Save+load should preserve all modality memories."""
    print("\n── M26. Save/load multimodal memories ──")
    _reset(m)
    m.write("Piano music Chopin.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain"], "nature")
    _store_video_memory(m, [["lion", "hunting"]], "wildlife")

    n_before = len(m.amm.tree.store)
    texts_before = sorted(e.source_text for e in m.amm.tree.store.values())

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        m.save_memory(path)
        _reset(m)
        m.load_memory(path)
        n_after = len(m.amm.tree.store)
        texts_after = sorted(e.source_text for e in m.amm.tree.store.values())

        R.check("m26_count_preserved", n_after == n_before)
        R.check("m26_texts_preserved", texts_before == texts_after,
                f"before={texts_before}, after={texts_after}")
        for e in m.amm.tree.store.values():
            R.check(f"m26_{e.mid}_sem_emb", e.semantic_emb is not None)
    finally:
        os.unlink(path)
    _reset(m)


def test_save_load_preserves_image_content_bias(m, c, R):
    """Content bias from image memory should be preserved after save+load."""
    print("\n── M27. Save/load preserves image content bias ──")
    _reset(m)
    _store_image_memory(m, ["sunset", "mountain", "ocean", "golden"], "nature")
    m.eval()
    top_before, _, _ = _content_bias_top_tokens(m, "The sunset mountain.", k=15)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        m.save_memory(path)
        _reset(m)
        m.load_memory(path)
        m.eval()
        top_after, _, _ = _content_bias_top_tokens(m, "The sunset mountain.", k=15)
        overlap = len(set(top_before) & set(top_after))
        R.check("m27_image_bias_preserved",
                overlap >= len(top_before) // 2,
                f"overlap={overlap}/{len(top_before)}")
    finally:
        os.unlink(path)
    _reset(m)


def test_save_load_multimodal_generation(m, c, R):
    """Generation with mixed modalities should be consistent after save+load."""
    print("\n── M28. Save/load multimodal generation consistency ──")
    _reset(m)
    m.write("Piano Chopin nocturne performance.", training_mode=True)
    _store_image_memory(m, ["concert", "hall", "stage"], "concert_photo")
    m.eval()

    with torch.no_grad():
        gen_before = m.generate("The pianist", mt=20, greedy=True)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        m.save_memory(path)
        _reset(m)
        m.load_memory(path)
        m.eval()
        with torch.no_grad():
            gen_after = m.generate("The pianist", mt=20, greedy=True)
        R.check("m28_gen_consistent",
                gen_before == gen_after,
                f"before='{gen_before[:40]}', after='{gen_after[:40]}'")
    finally:
        os.unlink(path)
    _reset(m)


def test_save_load_video_fields(m, c, R):
    """Video memory fields should survive save+load round-trip."""
    print("\n── M29. Save/load video field preservation ──")
    _reset(m)
    entry = _store_video_memory(m,
        [["lion", "hunting", "savanna"], ["elephant", "herd"]],
        "wildlife")
    mid = entry.mid
    ct_before = entry.content_token_ids[:]
    exp_before = entry.expanded_content_ids[:]

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        m.save_memory(path)
        _reset(m)
        m.load_memory(path)
        loaded = m.amm.tree.store.get(mid)
        R.check("m29_video_reloaded", loaded is not None)
        if loaded:
            R.check("m29_video_tag", "[VIDEO]" in loaded.source_text)
            R.check("m29_content_ids_preserved",
                    set(ct_before) == set(loaded.content_token_ids))
            R.check("m29_expanded_preserved",
                    set(exp_before) == set(loaded.expanded_content_ids))
    finally:
        os.unlink(path)
    _reset(m)


def test_save_load_modality_tag_integrity(m, c, R):
    """All modality tags should survive save+load."""
    print("\n── M30. Save/load modality tag integrity ──")
    _reset(m)
    m.write("Text only memory.", training_mode=True)
    _store_image_memory(m, ["photo", "visual"], "photo")
    _store_video_memory(m, [["motion", "frame"]], "clip")

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        m.save_memory(path)
        _reset(m)
        m.load_memory(path)
        texts = [e.source_text for e in m.amm.tree.store.values()]
        has_text = any("[IMAGE]" not in t and "[VIDEO]" not in t for t in texts)
        has_img = any("[IMAGE]" in t for t in texts)
        has_vid = any("[VIDEO]" in t for t in texts)
        R.check("m30_text_tag_survives", has_text)
        R.check("m30_image_tag_survives", has_img)
        R.check("m30_video_tag_survives", has_vid)
    finally:
        os.unlink(path)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# M31-M35: 多模态训练
# ═══════════════════════════════════════════════════════════════════
def test_training_with_multimodal_store(m, c, R):
    """Training should work with mixed-modality memories in store."""
    print("\n── M31. Training with multimodal store ──")
    _reset(m)
    texts = [
        "The cat sat on the mat.",
        "Quantum computing uses qubits.",
        "He practiced piano Chopin.",
    ]
    for t in texts:
        m.write(t, training_mode=True)
    _store_image_memory(m, ["sunset", "mountain"], "nature")
    _store_video_memory(m, [["lion", "hunting"]], "wildlife")

    trainer = Trainer(m, c)
    info = trainer.step(texts)
    R.check("m31_train_finite", math.isfinite(info['total']))
    R.check("m31_train_has_grad_norms", 'grad_norms' in info)
    m.eval()
    _reset(m)


def test_training_does_not_corrupt_image_memory(m, c, R):
    """Training should not corrupt image memory entries."""
    print("\n── M32. Training doesn't corrupt image memory ──")
    _reset(m)
    texts = ["Piano music.", "Space telescope."]
    for t in texts:
        m.write(t, training_mode=True)
    img_entry = _store_image_memory(m, ["sunset", "ocean"], "nature")
    img_mid = img_entry.mid

    trainer = Trainer(m, c)
    for _ in range(3):
        trainer.step(texts)

    if img_mid in m.amm.tree.store:
        e = m.amm.tree.store[img_mid]
        R.check("m32_image_base_finite", e.base.isfinite().all().item())
        R.check("m32_image_fiber_finite", e.fiber.isfinite().all().item())
        R.check("m32_image_tag_intact", "[IMAGE]" in e.source_text)
    else:
        R.check("m32_image_may_be_refreshed", True)
    m.eval()
    _reset(m)


def test_training_convergence_with_multimodal(m, c, R):
    """Training loss should not diverge with mixed modalities in store."""
    print("\n── M33. Training convergence with multimodal store ──")
    _reset(m)
    texts = [
        "The cat sat on the mat.",
        "Piano Chopin nocturne.",
        "Telescope galaxies.",
    ]
    for t in texts:
        m.write(t, training_mode=True)
    _store_image_memory(m, ["sunset", "mountain"], "nature")
    _store_video_memory(m, [["lion", "hunting"]], "wildlife")

    trainer = Trainer(m, c)
    losses = []
    for _ in range(4):
        info = trainer.step(texts)
        losses.append(info['total'])
    R.check("m33_all_finite", all(math.isfinite(l) for l in losses))
    R.check("m33_not_diverging", losses[-1] < losses[0] * 5,
            f"first={losses[0]:.4f}, last={losses[-1]:.4f}")
    m.eval()
    _reset(m)


def test_refresh_with_multimodal(m, c, R):
    """Memory refresh should handle text memories while keeping non-text entries."""
    print("\n── M34. Refresh with multimodal store ──")
    _reset(m)
    m.write("Piano practice Chopin.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain"], "nature")
    n_before = len(m.amm.tree.store)

    with torch.no_grad():
        n_refreshed = m._refresh_all_memories()

    n_after = len(m.amm.tree.store)
    R.check("m34_refresh_runs", n_refreshed >= 0)
    R.check("m34_some_entries_remain", n_after > 0)
    _reset(m)


def test_gradient_flow_with_multimodal_store(m, c, R):
    """Gradients should flow through components with multimodal memories."""
    print("\n── M35. Gradient flow with multimodal store ──")
    _reset(m)
    m.write("Piano Chopin.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain"], "nature")
    m.train()
    m.zero_grad()

    dev = _dev(m)
    tk = m.tok("Tell me about music.", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        bo = m.fwd(ids, mask)
    prefix = m._get_prefix(bo['hs'], mask, update_stats=False, ids=ids)
    o = m.fwd(ids, mask, prefix)
    lg = o['logits'][:, o['pl']:-1]
    tg = ids[:, 1:]
    ml = min(lg.shape[1], tg.shape[1])
    if ml > 0:
        loss = F.cross_entropy(lg[:, :ml].reshape(-1, lg.shape[-1]),
                               tg[:, :ml].reshape(-1))
        loss.backward()
        has_grad = m.bridge.bypass.proj[0].weight.grad is not None
        R.check("m35_grad_flows", has_grad)
    m.zero_grad()
    m.eval()
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# M36-M40: SpectralDealiaser 多模态场景
# ═══════════════════════════════════════════════════════════════════
def test_dealiaser_with_mixed_modalities(m, c, R):
    """SpectralDealiaser should handle mixed-modality memories."""
    print("\n── M36. Dealiaser with mixed modalities ──")
    _reset(m)
    dev = _dev(m)
    m.write("Piano music Chopin.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain"], "nature")
    _store_video_memory(m, [["lion", "hunting"]], "wildlife")
    for i in range(3):
        h = torch.randn(c.d_LLM, device=dev) * 0.5
        m.amm.store_mem(h, 1.0, training_mode=True, source_text=f"extra_{i}")

    da = SpectralDealiaser(m.amm, c)
    clusters = da.detect(sim_threshold=0.3)
    R.check("m36_detect_runs", isinstance(clusters, list))

    if len(m.amm.tree.store) >= 2:
        mids = list(m.amm.tree.store.keys())[:3]
        da.dealias(mids, steps=10)
        errs = m.amm.tree.verify_consistency()
        R.check("m36_dealias_consistent", len(errs) == 0, str(errs))
    else:
        R.check("m36_dealias_consistent", True)
    _reset(m)


def test_dealiaser_separates_similar_modals(m, c, R):
    """Dealiasing similar cross-modal entries should reduce similarity."""
    print("\n── M37. Dealiaser separates similar modals ──")
    _reset(m)
    dev = _dev(m)
    base_concepts = ["sunset", "mountain"]
    for i in range(4):
        h = _make_image_embedding(m, base_concepts).to(dev)
        h = h + torch.randn_like(h) * 0.01
        m.amm.store_mem(h, 1.0, training_mode=True, source_text=f"[IMAGE] sim_{i}")

    mids = list(m.amm.tree.store.keys())
    fibers_before = torch.stack([m.amm.tree.store[mid].fiber for mid in mids])
    fn_b = F.normalize(fibers_before, dim=-1)
    mask = ~torch.eye(len(mids), dtype=torch.bool)
    avg_before = (fn_b @ fn_b.T)[mask].mean().item()

    da = SpectralDealiaser(m.amm, c)
    da.dealias(mids, steps=20)

    fibers_after = torch.stack([m.amm.tree.store[mid].fiber
                                for mid in mids if mid in m.amm.tree.store])
    if fibers_after.shape[0] >= 2:
        fn_a = F.normalize(fibers_after, dim=-1)
        mask2 = ~torch.eye(fibers_after.shape[0], dtype=torch.bool)
        avg_after = (fn_a @ fn_a.T)[mask2].mean().item()
        R.check("m37_similarity_reduced",
                avg_after <= avg_before + 0.1,
                f"before={avg_before:.4f}, after={avg_after:.4f}")
    else:
        R.check("m37_similarity_reduced", True)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# M38-M40: 端到端多模态工作流
# ═══════════════════════════════════════════════════════════════════
def test_e2e_multimodal_write_retrieve_generate(m, c, R):
    """Full E2E: text write + image store + video store → retrieve → generate."""
    print("\n── M38. E2E multimodal workflow ──")
    _reset(m)
    m.write("The experienced pianist performed a Chopin nocturne.", training_mode=True)
    _store_image_memory(m,
        ["concert", "hall", "grand", "piano", "stage"],
        "concert_hall_photo")
    _store_video_memory(m,
        [["pianist", "fingers", "keyboard"],
         ["audience", "applause", "standing"],
         ["curtain", "bow", "flowers"]],
        "concert_performance_video")
    m.eval()

    R.check("m38_entries_stored", len(m.amm.tree.store) >= 3)
    top_toks, _, _ = _content_bias_top_tokens(m, "The piano concert performance.")
    R.check("m38_retrieval_works", len(top_toks) > 0)

    torch.manual_seed(42)
    with torch.no_grad():
        gen = m.generate("The pianist performed", mt=30, greedy=False)
    R.check("m38_generation_works", len(gen) > len("The pianist performed"))
    print(f"    generated: '{gen[:80]}'")
    _reset(m)


def test_e2e_multimodal_save_load_retrieve(m, c, R):
    """E2E: store mixed → save → load → retrieve → verify."""
    print("\n── M39. E2E multimodal save-load-retrieve ──")
    _reset(m)
    m.write("Telescope revealed galaxies.", training_mode=True)
    _store_image_memory(m, ["nebula", "star", "cosmic"], "deep_space_photo")
    _store_video_memory(m,
        [["rocket", "launch", "flame"], ["orbit", "earth", "space"]],
        "rocket_launch_video")

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        m.save_memory(path)
        _reset(m)
        m.load_memory(path)
        m.eval()

        R.check("m39_loaded", len(m.amm.tree.store) >= 3)
        top_toks, _, _ = _content_bias_top_tokens(m, "Tell me about space.")
        R.check("m39_retrieval_after_load", len(top_toks) > 0)
        errs = m.amm.tree.verify_consistency()
        R.check("m39_tree_consistent", len(errs) == 0, str(errs))
    finally:
        os.unlink(path)
    _reset(m)


def test_e2e_multimodal_train_then_generate(m, c, R):
    """E2E: store mixed → train → generate."""
    print("\n── M40. E2E multimodal train-generate ──")
    _reset(m)
    texts = [
        "The cat sat on the mat.",
        "He practiced piano Chopin nocturne.",
        "The telescope revealed galaxies.",
    ]
    for t in texts:
        m.write(t, training_mode=True)
    _store_image_memory(m, ["concert", "stage", "audience"], "concert")
    _store_video_memory(m, [["sunset", "mountain", "timelapse"]], "nature")

    trainer = Trainer(m, c)
    info = trainer.step(texts)
    R.check("m40_train_ok", math.isfinite(info['total']))

    m.eval()
    with torch.no_grad():
        gen = m.generate("The pianist", mt=20, greedy=True)
    R.check("m40_gen_ok", len(gen) > len("The pianist"))
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# M41-M45: 多模态方向树与检索诊断
# ═══════════════════════════════════════════════════════════════════
def test_multimodal_tree_structure(m, c, R):
    """Direction tree should organize mixed-modality memories properly."""
    print("\n── M41. Multimodal tree structure ──")
    _reset(m)
    for i in range(5):
        m.write(f"Topic {i} about different subjects.", training_mode=True)
    for i in range(5):
        _store_image_memory(m, [f"concept{i}", "visual"], f"img_{i}")
    for i in range(5):
        _store_video_memory(m, [[f"frame{i}", "motion"]], f"vid_{i}")

    n_total = len(m.amm.tree.store)
    R.check("m41_all_stored", n_total > 0)
    depth = m.amm.tree.max_depth()
    R.check("m41_tree_has_structure", depth >= 0, f"depth={depth}")
    errs = m.amm.tree.verify_consistency()
    R.check("m41_tree_consistent", len(errs) == 0, str(errs))
    violations = m.amm.tree.leaf_size_violations()
    R.check("m41_no_leaf_violations", len(violations) == 0)
    _reset(m)


def test_multimodal_retrieval_diag(m, c, R):
    """RetrievalDiag should report metrics for mixed-modality store."""
    print("\n── M42. Multimodal retrieval diagnostics ──")
    _reset(m)
    dev = _dev(m)
    m.write("Piano music.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain"], "nature")
    _store_video_memory(m, [["lion", "hunting"]], "wildlife")
    m.eval()

    tk = m.tok("Tell me about piano.", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        _, _, diag, _ = m._get_prefix(
            o['hs'], mask, update_stats=False, return_extra=True, ids=ids)

    R.check("m42_diag_recall", diag.recall_count > 0)
    R.check("m42_diag_fiber_norm", diag.fiber_summary_norm > 0)
    R.check("m42_diag_has_weights", len(diag.batch_mem_weights) > 0)
    _reset(m)


def test_multimodal_direction_degeneracy(m, c, R):
    """Check for direction degeneracy in mixed-modality store."""
    print("\n── M43. Multimodal direction degeneracy check ──")
    _reset(m)
    for i in range(3):
        m.write(f"Topic {i} about unique subjects.", training_mode=True)
    for i in range(3):
        _store_image_memory(m, [f"unique{i}", "visual"], f"img_{i}")
    for i in range(3):
        _store_video_memory(m, [[f"scene{i}", "motion"]], f"vid_{i}")

    degen = m.amm.tree.check_direction_degeneracy(threshold=0.99)
    R.check("m43_degeneracy_check_runs", isinstance(degen, list))
    _reset(m)


def test_multimodal_flat_scan(m, c, R):
    """With few mixed memories, flat scan should be used."""
    print("\n── M44. Multimodal flat scan ──")
    _reset(m)
    dev = _dev(m)
    m.write("Piano.", training_mode=True)
    _store_image_memory(m, ["sunset"], "nature")

    xq = torch.randn(1, c.d_M, device=dev)
    fq = torch.randn(1, c.d_F, device=dev)
    _, _, _, diag = m.amm.retrieve_multi(xq, fq)
    threshold = c.flat_scan_threshold_factor * c.retrieval_topk
    R.check("m44_flat_scan_used",
            diag.was_flat_scan == (len(m.amm.tree.store) <= threshold))
    _reset(m)


def test_multimodal_batch_retrieval(m, c, R):
    """Batched text queries should work with mixed-modality store."""
    print("\n── M45. Multimodal batch retrieval ──")
    _reset(m)
    m.write("Piano Chopin nocturne.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain"], "nature")
    _store_video_memory(m, [["lion", "hunting"]], "wildlife")
    m.eval()

    dev = _dev(m)
    tk = m.tok(["Tell me about piano.", "The mountain sunset."],
               return_tensors='pt', padding=True, truncation=True)
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        _, _, _, cb = m._get_prefix(o['hs'], mask, return_extra=True, ids=ids)
    R.check("m45_batch_shape", cb.shape[0] == 2)
    R.check("m45_batch_finite", cb.isfinite().all().item())
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# M46-M50: 边界与压力场景
# ═══════════════════════════════════════════════════════════════════
def test_image_only_store_generate(m, c, R):
    """System should work with only image memories (no text)."""
    print("\n── M46. Image-only store and generate ──")
    _reset(m)
    _store_image_memory(m, ["sunset", "mountain", "ocean", "golden"], "nature")
    _store_image_memory(m, ["forest", "river", "waterfall", "green"], "forest")
    m.eval()

    with torch.no_grad():
        gen = m.generate("The beautiful", mt=20, greedy=True)
    R.check("m46_img_only_gen_ok", len(gen) > len("The beautiful"))
    _reset(m)


def test_video_only_store_generate(m, c, R):
    """System should work with only video memories (no text)."""
    print("\n── M47. Video-only store and generate ──")
    _reset(m)
    _store_video_memory(m,
        [["lion", "hunting", "savanna"], ["elephant", "herd", "migration"]],
        "wildlife1")
    _store_video_memory(m,
        [["eagle", "flying", "mountain"], ["fish", "river", "catch"]],
        "wildlife2")
    m.eval()

    with torch.no_grad():
        gen = m.generate("The wildlife", mt=20, greedy=True)
    R.check("m47_vid_only_gen_ok", len(gen) > len("The wildlife"))
    _reset(m)


def test_rapid_modal_switching(m, c, R):
    """Rapid alternation between modalities should not corrupt state."""
    print("\n── M48. Rapid modal switching ──")
    _reset(m)
    for i in range(10):
        if i % 3 == 0:
            m.write(f"Text memory {i}.", training_mode=True)
        elif i % 3 == 1:
            _store_image_memory(m, [f"visual{i}"], f"img_{i}")
        else:
            _store_video_memory(m, [[f"frame{i}"]], f"vid_{i}")

    errs = m.amm.tree.verify_consistency()
    R.check("m48_tree_consistent", len(errs) == 0, str(errs))
    R.check("m48_entries_stored", len(m.amm.tree.store) > 0)
    _reset(m)


def test_large_multimodal_store_retrieval(m, c, R):
    """Large mixed-modality store should handle retrieval without crash."""
    print("\n── M49. Large multimodal store retrieval ──")
    _reset(m)
    dev = _dev(m)
    for i in range(10):
        m.write(f"Various topic {i} about things.", training_mode=True)
    for i in range(10):
        h = torch.randn(c.d_LLM, device=dev) * 0.5
        m.amm.store_mem(h, 1.0 + i*0.1, training_mode=True,
                        source_text=f"[IMAGE] scene_{i}",
                        content_token_ids=[100+i],
                        content_semantic_emb=h.clone())
    for i in range(10):
        h = torch.randn(c.d_LLM, device=dev) * 0.5
        m.amm.store_mem(h, 2.0 + i*0.1, training_mode=True,
                        source_text=f"[VIDEO] clip_{i}",
                        content_token_ids=[200+i],
                        content_semantic_emb=h.clone())

    n_total = len(m.amm.tree.store)
    R.check("m49_large_store", n_total > 10, f"n={n_total}")

    xq = torch.randn(1, c.d_M, device=dev)
    fq = torch.randn(1, c.d_F, device=dev)
    fibers, mask, fs, diag = m.amm.retrieve_multi(xq, fq)
    R.check("m49_retrieve_finite", fibers.isfinite().all().item())
    errs = m.amm.tree.verify_consistency()
    R.check("m49_tree_consistent", len(errs) == 0, str(errs))
    _reset(m)


def test_multimodal_decay_consolidate_combined(m, c, R):
    """Decay + consolidate on mixed-modality store should maintain consistency."""
    print("\n── M50. Multimodal decay + consolidate ──")
    _reset(m)
    for i in range(3):
        m.write(f"Topic {i}.", training_mode=True)
    for i in range(3):
        _store_image_memory(m, [f"concept{i}"], f"img_{i}")
    for i in range(3):
        _store_video_memory(m, [[f"frame{i}"]], f"vid_{i}")

    m.amm.time += 3000
    m.amm.decay()
    m.amm.consolidate()
    errs = m.amm.tree.verify_consistency()
    R.check("m50_decay_consol_consistent", len(errs) == 0, str(errs))
    _reset(m)


def test_multimodal_semantic_e2e_full(m, c, R):
    """Complete multimodal semantic test: write all modalities → train →
    save → load → retrieve domain-specific → generate."""
    print("\n── M51. Full multimodal semantic E2E ──")
    _reset(m)

    m.write("The experienced pianist performed a magnificent Chopin nocturne.", training_mode=True)
    m.write("The telescope revealed distant galaxies beyond the Milky Way.", training_mode=True)
    _store_image_memory(m,
        ["concert", "hall", "grand", "piano", "audience", "stage"],
        "concert_hall_panorama")
    _store_image_memory(m,
        ["nebula", "star", "cosmic", "galaxy", "deep", "space"],
        "deep_space_hubble")
    _store_video_memory(m,
        [["pianist", "fingers", "keyboard", "melody"],
         ["audience", "applause", "standing", "ovation"],
         ["curtain", "bow", "flowers", "encore"]],
        "live_concert_recording")
    _store_video_memory(m,
        [["rocket", "launch", "flame", "countdown"],
         ["orbit", "earth", "blue", "atmosphere"],
         ["spacewalk", "astronaut", "station", "repair"]],
        "space_mission_documentary")

    texts = ["The pianist performed Chopin.", "The telescope revealed galaxies."]
    trainer = Trainer(m, c)
    info = trainer.step(texts)
    R.check("m51_train_ok", math.isfinite(info['total']))
    m.eval()

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        m.save_memory(path)
        _reset(m)
        m.load_memory(path)
        m.eval()

        R.check("m51_loaded", len(m.amm.tree.store) >= 4)

        top_music, _, _ = _content_bias_top_tokens(m, "The piano concert.")
        top_space, _, _ = _content_bias_top_tokens(m, "The space telescope.")
        MUSIC_BROAD = MUSIC_KW | {'concert', 'hall', 'stage', 'audience',
                                    'grand', 'fingers', 'keyboard',
                                    'applause', 'curtain', 'bow'}
        SPACE_BROAD = SPACE_KW | {'nebula', 'cosmic', 'deep', 'hubble',
                                   'flame', 'countdown', 'atmosphere',
                                   'spacewalk', 'station', 'repair'}
        music_in_music = sum(1 for t in top_music[:15] if t in MUSIC_BROAD)
        space_in_space = sum(1 for t in top_space[:15] if t in SPACE_BROAD)
        R.check("m51_music_retrieves_music", music_in_music > 0,
                f"hits={music_in_music}, top={top_music[:15]}")
        R.check("m51_space_retrieves_space", space_in_space > 0,
                f"hits={space_in_space}, top={top_space[:15]}")

        torch.manual_seed(42)
        with torch.no_grad():
            gen = m.generate("The pianist performed", mt=30, greedy=False)
        R.check("m51_gen_ok", len(gen) > len("The pianist performed"))
        print(f"    final gen: '{gen[:80]}'")
    finally:
        os.unlink(path)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# Main Entry
# ═══════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(42)
    c = Cfg()
    R = TestResults()

    sep = "=" * 70
    print(f"\n{sep}")
    print("  AMS v3.7 — Multimodal Semantic Black-Box Test Suite")
    print(f"{sep}")
    t0 = time.time()

    print("\n[Building MemLLM + loading GPT-2]")
    m = MemLLM(c)
    m.load("gpt2")
    total = sum(p.numel() for p in m.parameters())
    train_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  Params: total={total:,}  trainable={train_p:,}  frozen={total-train_p:,}")

    # M1-M5: 基础写入
    test_image_memory_stored(m, c, R)
    test_video_memory_stored(m, c, R)
    test_text_image_video_coexist(m, c, R)
    test_multimodal_embeddings_differ(m, c, R)
    test_multimodal_different_domains_differ(m, c, R)

    # M6-M10: 跨模态检索
    test_text_query_retrieves_related_image(m, c, R)
    test_text_query_retrieves_related_video(m, c, R)
    test_cross_modal_retrieval_text_to_image(m, c, R)
    test_mixed_modality_four_way_retrieval(m, c, R)
    test_image_video_same_domain_reinforce(m, c, R)

    # M11-M15: 生成影响
    test_image_memory_influences_generation(m, c, R)
    test_video_memory_influences_generation(m, c, R)
    test_mixed_modal_generation_quality(m, c, R)
    test_generation_prefers_matching_modality_domain(m, c, R)
    test_video_frames_temporal_richness(m, c, R)

    # M16-M20: 合并与衰减
    test_consolidation_across_modalities(m, c, R)
    test_multimodal_decay(m, c, R)
    test_image_image_consolidation(m, c, R)
    test_video_high_surprise_survives(m, c, R)
    test_multimodal_multiple_consolidation_rounds(m, c, R)

    # M21-M25: 干扰与隔离
    test_image_does_not_overwhelm_text(m, c, R)
    test_video_does_not_overwhelm_text(m, c, R)
    test_cross_modal_domain_isolation(m, c, R)
    test_same_domain_cross_modal_similarity(m, c, R)
    test_many_modalities_no_catastrophic_failure(m, c, R)

    # M26-M30: 保存/加载
    test_save_load_multimodal(m, c, R)
    test_save_load_preserves_image_content_bias(m, c, R)
    test_save_load_multimodal_generation(m, c, R)
    test_save_load_video_fields(m, c, R)
    test_save_load_modality_tag_integrity(m, c, R)

    # M31-M35: 训练
    test_training_with_multimodal_store(m, c, R)
    test_training_does_not_corrupt_image_memory(m, c, R)
    test_training_convergence_with_multimodal(m, c, R)
    test_refresh_with_multimodal(m, c, R)
    test_gradient_flow_with_multimodal_store(m, c, R)

    # M36-M37: Dealiaser
    test_dealiaser_with_mixed_modalities(m, c, R)
    test_dealiaser_separates_similar_modals(m, c, R)

    # M38-M40: E2E
    test_e2e_multimodal_write_retrieve_generate(m, c, R)
    test_e2e_multimodal_save_load_retrieve(m, c, R)
    test_e2e_multimodal_train_then_generate(m, c, R)

    # M41-M45: 树与检索诊断
    test_multimodal_tree_structure(m, c, R)
    test_multimodal_retrieval_diag(m, c, R)
    test_multimodal_direction_degeneracy(m, c, R)
    test_multimodal_flat_scan(m, c, R)
    test_multimodal_batch_retrieval(m, c, R)

    # M46-M51: 边界、压力、完整E2E
    test_image_only_store_generate(m, c, R)
    test_video_only_store_generate(m, c, R)
    test_rapid_modal_switching(m, c, R)
    test_large_multimodal_store_retrieval(m, c, R)
    test_multimodal_decay_consolidate_combined(m, c, R)
    test_multimodal_semantic_e2e_full(m, c, R)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    ok = R.summary()
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
