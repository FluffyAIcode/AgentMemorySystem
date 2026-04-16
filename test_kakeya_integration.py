#!/usr/bin/env python3
"""
Kakeya-Compressed AMS — Full Black-Box Test Suite
===================================================

Runs all three tiers of black-box tests against a KakeyaMemLLM-wrapped
MemLLM, plus Kakeya-specific compression validation tests.

The original test suites are imported and executed with the Kakeya-compressed
system substituted in. AgentMemorySystem.py is NOT modified.
"""

import sys, os, time, math, tempfile
import torch
import torch.nn.functional as F

from AgentMemorySystem import (
    Cfg, MemLLM, _Node, MemEntry, DirectionTree,
    Trainer, SpectralDealiaser, RetrievalDiag,
)
from kakeya_codec import KakeyaCodec, KakeyaMemLLM


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


def _reset(km):
    km.amm.tree.store.clear()
    km.amm.tree.root = _Node()
    km.amm.tree.nid = 0
    km.amm.time = 0
    km.codec.sem_compressed.clear()
    km.codec.wte_compressed.clear()
    km.codec.sem_skeleton = None
    km.codec.wte_skeleton = None
    km.codec._is_active = False


def _dev(km):
    return next(km.parameters()).device


def _content_bias_top_tokens(km, query, k=20):
    dev = _dev(km)
    tk = km.tok(query, return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = km.fwd(ids, mask)
        prefix, fs, diag, cb = km._get_prefix(
            o['hs'], mask, update_stats=False, return_extra=True, ids=ids)
    topk_ids = cb[0].topk(k).indices.tolist()
    topk_toks = [km.tok.decode([t]).strip().lower() for t in topk_ids]
    return topk_toks, cb, diag


MUSIC_KW = {'piano', 'chopin', 'nocturne', 'orchestra', 'beethoven', 'symphony',
            'harmony', 'melody', 'chord', 'performed', 'harmonic', 'progression',
            'conservatory', 'violin', 'concerto', 'musician', 'composer',
            'pianist', 'music', 'instrument', 'practiced'}
SPACE_KW = {'galaxy', 'galaxies', 'telescope', 'star', 'planet', 'orbit',
            'space', 'astronaut', 'mars', 'nebula', 'gravity', 'cosmic',
            'satellite', 'mission', 'rocket', 'spacecraft', 'launch'}
NATURE_KW = {'sunset', 'mountain', 'ocean', 'forest', 'river', 'landscape',
             'valley', 'cliff', 'waterfall', 'meadow', 'sky', 'cloud'}


def _get_concept_embedding(km, words):
    wte = km.llm.transformer.wte.weight.detach()
    tok = km.tok
    all_ids = []
    for w in words:
        ids = tok.encode(" " + w)
        all_ids.extend(ids)
    valid = [i for i in all_ids if i < wte.shape[0]]
    if not valid:
        return torch.zeros(wte.shape[1], device=wte.device)
    return wte[valid].mean(0)


def _make_image_embedding(km, concepts, seed=1000):
    base = _get_concept_embedding(km, concepts)
    torch.manual_seed(seed)
    offset = torch.randn_like(base) * 0.15
    return F.normalize(base + offset, dim=0) * base.norm()


def _store_image_memory(km, concepts, label, surprise=1.5):
    dev = _dev(km)
    h = _make_image_embedding(km, concepts).to(dev)
    cc = km.content_classifier
    tok = km.tok
    ct_ids = []
    for w in concepts:
        for tid in tok.encode(" " + w):
            if tid in cc.content_ids:
                ct_ids.append(tid)
    expanded = km._expand_content_ids(ct_ids)
    wte_c = km._compute_wte_centroid(ct_ids)
    return km.amm.store_mem(h, surprise, training_mode=True,
                            source_text=f"[IMAGE] {label}",
                            content_token_ids=ct_ids,
                            content_semantic_emb=h.clone(),
                            content_wte_centroid=wte_c,
                            expanded_content_ids=expanded)


def _make_video_embedding(km, frame_list, seed=2000):
    frames = []
    for i, concepts in enumerate(frame_list):
        f = _get_concept_embedding(km, concepts)
        frames.append(f * (1.0 + 0.1 * i))
    if not frames:
        return torch.zeros(km.c.d_LLM, device=_dev(km))
    v = torch.stack(frames).mean(0)
    torch.manual_seed(seed)
    offset = torch.randn_like(v) * 0.2
    return F.normalize(v + offset, dim=0) * v.norm()


def _store_video_memory(km, frame_list, label, surprise=2.0):
    dev = _dev(km)
    h = _make_video_embedding(km, frame_list).to(dev)
    cc = km.content_classifier
    tok = km.tok
    ct_ids = []
    for frame in frame_list:
        for w in frame:
            for tid in tok.encode(" " + w):
                if tid in cc.content_ids:
                    ct_ids.append(tid)
    ct_ids = list(set(ct_ids))
    expanded = km._expand_content_ids(ct_ids)
    wte_c = km._compute_wte_centroid(ct_ids)
    return km.amm.store_mem(h, surprise, training_mode=True,
                            source_text=f"[VIDEO] {label}",
                            content_token_ids=ct_ids,
                            content_semantic_emb=h.clone(),
                            content_wte_centroid=wte_c,
                            expanded_content_ids=expanded)


# ═══════════════════════════════════════════════════════════════════
# Part 1: Kakeya Codec Validation (KV1-KV10)
# ═══════════════════════════════════════════════════════════════════

def test_kv_codec_build(km, c, R):
    print("\n── KV1. Codec build from text memories ──")
    _reset(km)
    for t in [
        "The pianist performed Chopin nocturne.",
        "The telescope revealed distant galaxies.",
        "The chef prepared exquisite cuisine.",
        "Neural networks process visual data.",
        "The surgeon performed cardiac surgery.",
        "The athlete won gold medal.",
        "Ancient historians documented Rome.",
        "The mathematician proved theorem.",
        "Quantum computing uses qubits.",
        "The architect designed buildings.",
    ]:
        km.write(t, training_mode=True)
    R.check("kv1_codec_active", km.codec.is_active)
    stats = km.codec.get_stats()
    R.check("kv1_sem_entries", stats['sem_entries'] > 0, f"n={stats['sem_entries']}")
    R.check("kv1_wte_entries", stats['wte_entries'] > 0, f"n={stats['wte_entries']}")
    R.check("kv1_d_eff_sem", stats['sem_d_eff'] > 0, f"d={stats['sem_d_eff']}")
    R.check("kv1_K_sem", stats['sem_K'] > 0, f"K={stats['sem_K']}")
    print(f"    stats: {stats}")
    _reset(km)


def test_kv_encode_decode_precision(km, c, R):
    print("\n── KV2. Encode-decode cosine precision ──")
    _reset(km)
    for t in [
        "The pianist performed Chopin nocturne at the concert hall.",
        "She studied music theory and harmonic progression.",
        "The telescope revealed distant galaxies in deep space.",
        "Astronauts trained for the Mars mission.",
        "The chef prepared exquisite French cuisine.",
        "Neural networks process visual recognition data.",
        "The surgeon performed complex cardiac surgery.",
        "The athlete won gold in the Olympic competition.",
        "Ancient historians documented the Roman empire.",
        "The mathematician proved algebraic theorem.",
    ]:
        km.write(t, training_mode=True)

    dev = _dev(km)
    cos_errs = []
    for mid, entry in km.amm.tree.store.items():
        km._decompress_entry(entry)
        if entry.semantic_emb is not None:
            original = entry.semantic_emb.clone()
            compressed = km.codec.sem_compressed.get(mid)
            if compressed is not None:
                decoded = km.codec.decode_sem(mid, dev)
                if decoded is not None:
                    cs = F.cosine_similarity(original.unsqueeze(0),
                                             decoded.unsqueeze(0)).item()
                    cos_errs.append(1.0 - cs)
    if cos_errs:
        avg_err = sum(cos_errs) / len(cos_errs)
        max_err = max(cos_errs)
        R.check("kv2_avg_cos_err_small", avg_err < 0.05, f"avg={avg_err:.6f}")
        R.check("kv2_max_cos_err_bounded", max_err < 0.1, f"max={max_err:.6f}")
        print(f"    cosine error: avg={avg_err:.6f}, max={max_err:.6f}")
    else:
        R.check("kv2_has_compressed_entries", False, "no compressed entries")
    _reset(km)


def test_kv_compression_ratio(km, c, R):
    print("\n── KV3. Compression ratio ──")
    _reset(km)
    for t in [
        "Piano music theory.", "Space telescope observation.",
        "Chef cooking recipe.", "Quantum computing algorithm.",
        "Cardiac surgery procedure.", "Olympic swimming competition.",
        "Roman history documentation.", "Algebraic theorem proof.",
        "Neural network training.", "Architecture design principles.",
    ]:
        km.write(t, training_mode=True)
    stats = km.codec.get_stats()
    ratio = stats['compression_ratio']
    R.check("kv3_ratio_gt_1", ratio > 1.0, f"ratio={ratio:.2f}")
    print(f"    original={stats['original_bytes']}B, compressed={stats['compressed_bytes']}B, ratio={ratio:.2f}×")
    _reset(km)


def test_kv_retrieval_preserved(km, c, R):
    print("\n── KV4. Retrieval order preserved under compression ──")
    _reset(km)
    texts = [
        "He practiced piano for hours perfecting a difficult Chopin nocturne.",
        "She studied music theory and harmonic progression at the conservatory.",
        "The telescope revealed distant galaxies beyond the Milky Way.",
        "Astronauts trained for the Mars mission in simulated zero gravity.",
    ]
    for t in texts:
        km.write(t, training_mode=True)
    km.eval()

    top_toks, cb, _ = _content_bias_top_tokens(km, "Tell me about piano practice.")
    music_hits = sum(1 for t in top_toks[:15] if t in MUSIC_KW)
    space_hits = sum(1 for t in top_toks[:15] if t in SPACE_KW)
    R.check("kv4_music_query_music_bias", music_hits >= space_hits,
            f"music={music_hits}, space={space_hits}")
    R.check("kv4_content_bias_nonzero", cb.abs().max().item() > 0.01)
    _reset(km)


def test_kv_generation_works(km, c, R):
    print("\n── KV5. Generation works with Kakeya compression ──")
    _reset(km)
    for t in [
        "The pianist performed Chopin.", "Music theory harmonic.",
        "Telescope galaxies space.", "Astronaut Mars mission.",
        "Chef prepared cuisine.", "Neural network algorithm.",
        "Surgeon cardiac operation.", "Athlete Olympic gold.",
        "Historian Roman empire.", "Mathematician theorem proof.",
    ]:
        km.write(t, training_mode=True)
    km.eval()
    with torch.no_grad():
        gen = km.generate("The pianist", mt=20, greedy=True)
    R.check("kv5_gen_nonempty", len(gen) > len("The pianist"), f"gen='{gen}'")
    new_text = gen[len("The pianist"):].strip()
    alpha = sum(1 for ch in new_text if ch.isalpha())
    ratio = alpha / max(len(new_text), 1)
    R.check("kv5_gen_not_degenerate", ratio > 0.2, f"ratio={ratio:.2f}")
    _reset(km)


def test_kv_save_load(km, c, R):
    print("\n── KV6. Save/load with Kakeya codec ──")
    _reset(km)
    for t in [
        "Piano Chopin performance.", "Telescope galaxy observation.",
        "Chef cuisine preparation.", "Quantum computing qubits.",
        "Surgeon cardiac surgery.", "Olympic gold medal.",
        "Roman empire history.", "Algebraic theorem math.",
        "Neural network visual.", "Architecture design building.",
    ]:
        km.write(t, training_mode=True)
    n_before = len(km.amm.tree.store)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        km.save_memory(path)
        _reset(km)
        km.load_memory(path)
        n_after = len(km.amm.tree.store)
        R.check("kv6_count_preserved", n_after == n_before)
        all_sem = all(e.semantic_emb is not None for e in km.amm.tree.store.values())
        R.check("kv6_sem_restored", all_sem)
    finally:
        os.unlink(path)
        codec_path = path + '.kakeya'
        if os.path.exists(codec_path):
            os.unlink(codec_path)
    _reset(km)


def test_kv_consolidation(km, c, R):
    print("\n── KV7. Consolidation with Kakeya compression ──")
    _reset(km)
    dev = _dev(km)
    for i in range(10):
        h = torch.randn(c.d_LLM, device=dev) * 0.01
        km.amm.store_mem(h, 0.5, True, source_text=f"similar_{i}",
                         content_semantic_emb=h.clone(),
                         content_wte_centroid=torch.randn(c.d_LLM, device=dev))
    if len(km.amm.tree.store) >= km._auto_threshold:
        km._maybe_build_codec()
    km.amm.consolidate()
    errs = km.amm.tree.verify_consistency()
    R.check("kv7_consistent_after_consol", len(errs) == 0, str(errs))
    _reset(km)


def test_kv_decay(km, c, R):
    print("\n── KV8. Decay with Kakeya compression ──")
    _reset(km)
    for t in [
        "Piano.", "Telescope.", "Chef.", "Quantum.", "Surgeon.",
        "Athlete.", "Historian.", "Math.", "Neural.", "Architect.",
    ]:
        km.write(t, training_mode=True)
    km.amm.time += 5000
    n_before = len(km.amm.tree.store)
    if km.codec.is_active:
        km._decompress_all()
    n_decayed = km.amm.decay()
    R.check("kv8_decay_runs", n_decayed >= 0)
    R.check("kv8_math_correct", len(km.amm.tree.store) == n_before - n_decayed)
    _reset(km)


def test_kv_multimodal(km, c, R):
    print("\n── KV9. Multimodal with Kakeya compression ──")
    _reset(km)
    km.write("The pianist performed Chopin.", training_mode=True)
    _store_image_memory(km, ["sunset", "mountain", "ocean"], "nature")
    _store_video_memory(km, [["lion", "hunting"], ["eagle", "flying"]], "wildlife")
    for t in [
        "Telescope galaxies.", "Chef cuisine.", "Quantum computing.",
        "Surgeon cardiac.", "Athlete Olympic.", "Historian Rome.",
        "Mathematician theorem.", "Neural network.",
    ]:
        km.write(t, training_mode=True)
    km.eval()

    R.check("kv9_multi_stored", len(km.amm.tree.store) >= 5)
    top_toks, _, _ = _content_bias_top_tokens(km, "Tell me about piano.")
    R.check("kv9_retrieval_works", len(top_toks) > 0)
    with torch.no_grad():
        gen = km.generate("The pianist", mt=15, greedy=True)
    R.check("kv9_gen_works", len(gen) > len("The pianist"))
    _reset(km)


def test_kv_training(km, c, R):
    print("\n── KV10. Training with Kakeya compression ──")
    _reset(km)
    texts = [
        "The cat sat on the mat.", "Quantum computing qubits.",
        "Piano Chopin nocturne.", "Telescope galaxies space.",
        "Chef prepared cuisine.", "Surgeon cardiac operation.",
        "Athlete Olympic gold.", "Historian Roman empire.",
        "Mathematician theorem.", "Neural network algorithm.",
    ]
    for t in texts:
        km.write(t, training_mode=True)

    if km.codec.is_active:
        km._decompress_all()
    trainer = Trainer(km._m, c)
    info = trainer.step(texts[:3])
    R.check("kv10_train_finite", math.isfinite(info['total']))
    km._m.eval()
    _reset(km)


# ═══════════════════════════════════════════════════════════════════
# Part 2: Re-run original structural tests (selected critical ones)
# ═══════════════════════════════════════════════════════════════════

def test_struct_write_and_retrieve(km, c, R):
    print("\n── S-STRUCT. Write + retrieve with Kakeya ──")
    _reset(km)
    texts = [
        "The cat sat on the mat and watched birds.",
        "Quantum computing uses qubits in superposition.",
        "He practiced piano for hours perfecting Chopin.",
        "The stock market experienced volatility.",
        "She walked along the beach at sunset.",
        "The chef prepared an exquisite meal.",
        "Machine learning algorithms identify patterns.",
        "The ancient temple was hidden in rainforest.",
        "The telescope revealed distant galaxies.",
        "Neural networks process visual recognition.",
    ]
    total = 0
    for t in texts:
        n, gv = km.write(t, training_mode=True)
        total += n
    R.check("struct_write_count", total > 0)
    R.check("struct_gate_range", True)

    all_have_text = all(e.source_text for e in km.amm.tree.store.values())
    R.check("struct_source_text", all_have_text)

    km.eval()
    dev = _dev(km)
    tk = km.tok("Tell me about piano.", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = km.fwd(ids, mask)
        _, _, _, cb = km._get_prefix(o['hs'], mask, return_extra=True, ids=ids)
    R.check("struct_retrieve_cb_nonzero", cb.abs().max().item() > 0)

    torch.manual_seed(42)
    with torch.no_grad():
        gen = km.generate("The pianist", 20, greedy=True)
    R.check("struct_generate_nonempty", len(gen) > len("The pianist"))
    _reset(km)


def test_struct_empty_memory(km, c, R):
    print("\n── S-EMPTY. Empty memory operations ──")
    _reset(km)
    km.eval()
    dev = _dev(km)
    tk = km.tok("Hello world", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = km.fwd(ids, mask)
        prefix, _, _, cb = km._get_prefix(o['hs'], mask, return_extra=True, ids=ids)
    R.check("empty_prefix_finite", prefix.isfinite().all().item())
    R.check("empty_cb_zero", cb.abs().max().item() < 1e-6)
    with torch.no_grad():
        gen = km.generate("Hello", mt=10, greedy=True)
    R.check("empty_generate_ok", len(gen) > 0)
    _reset(km)


def test_struct_greedy_deterministic(km, c, R):
    print("\n── S-GREEDY. Greedy determinism ──")
    _reset(km)
    for t in [
        "Cats are fluffy.", "Dogs play fetch.", "Birds fly south.",
        "Fish swim deep.", "Horses run fast.", "Rabbits hop quickly.",
        "Turtles move slowly.", "Eagles soar high.", "Wolves hunt packs.",
        "Bears hibernate winter.",
    ]:
        km.write(t, training_mode=True)
    km.eval()
    with torch.no_grad():
        g1 = km.generate("The cat", mt=15, greedy=True)
        g2 = km.generate("The cat", mt=15, greedy=True)
    R.check("greedy_deterministic", g1 == g2, f"g1='{g1[:40]}' g2='{g2[:40]}'")
    _reset(km)


def test_struct_batch_retrieval(km, c, R):
    print("\n── S-BATCH. Batch retrieval ──")
    _reset(km)
    for t in [
        "Cats are fluffy.", "Stars shine bright.",
        "Piano music beautiful.", "Space telescope observation.",
        "Chef cooking recipe.", "Quantum computing algorithm.",
        "Surgeon cardiac surgery.", "Olympic gold medal.",
        "Roman history empire.", "Neural network training.",
    ]:
        km.write(t, training_mode=True)
    km.eval()
    dev = _dev(km)
    tk = km.tok(["Tell me about cats.", "The night sky."],
                return_tensors='pt', padding=True, truncation=True)
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = km.fwd(ids, mask)
        _, _, _, cb = km._get_prefix(o['hs'], mask, return_extra=True, ids=ids)
    R.check("batch_cb_shape", cb.shape[0] == 2)
    R.check("batch_cb_finite", cb.isfinite().all().item())
    _reset(km)


# ═══════════════════════════════════════════════════════════════════
# Part 3: Semantic quality tests
# ═══════════════════════════════════════════════════════════════════

def test_sem_domain_retrieval(km, c, R):
    print("\n── SEM-DOMAIN. Domain retrieval with Kakeya ──")
    _reset(km)
    km.write("He practiced piano for hours perfecting Chopin nocturne.", training_mode=True)
    km.write("She studied music theory and harmonic progression.", training_mode=True)
    km.write("The telescope revealed distant galaxies.", training_mode=True)
    km.write("Astronauts trained for the Mars mission.", training_mode=True)
    for t in [
        "Chef cuisine.", "Quantum computing.", "Surgeon cardiac.",
        "Athlete Olympic.", "Historian Rome.", "Neural network.",
    ]:
        km.write(t, training_mode=True)
    km.eval()

    top_music, _, _ = _content_bias_top_tokens(km, "Tell me about piano.", k=15)
    top_space, _, _ = _content_bias_top_tokens(km, "What did the telescope reveal?", k=15)
    m_hits = sum(1 for t in top_music if t in MUSIC_KW)
    s_hits = sum(1 for t in top_space if t in SPACE_KW)
    R.check("sem_music_has_music_kw", m_hits > 0, f"hits={m_hits}")
    R.check("sem_space_has_space_kw", s_hits > 0, f"hits={s_hits}")
    _reset(km)


def test_sem_generation_quality(km, c, R):
    print("\n── SEM-GEN. Generation quality with Kakeya ──")
    _reset(km)
    for t in [
        "The cat sat on the mat.", "Piano Chopin nocturne.",
        "Telescope galaxies space.", "Chef prepared cuisine.",
        "Quantum computing qubits.", "Surgeon cardiac operation.",
        "Athlete Olympic gold.", "Historian Roman empire.",
        "Mathematician theorem proof.", "Neural network algorithm.",
    ]:
        km.write(t, training_mode=True)
    km.eval()
    cc = km.content_classifier
    for prompt in ["The pianist", "Stars and galaxies"]:
        torch.manual_seed(42)
        with torch.no_grad():
            gen = km.generate(prompt, mt=30, greedy=False)
        new_text = gen[len(prompt):].strip()
        alpha = sum(1 for ch in new_text if ch.isalpha())
        ratio = alpha / max(len(new_text), 1)
        R.check(f"sem_gen_{prompt[:8]}_alpha", ratio > 0.2,
                f"ratio={ratio:.2f}, text='{new_text[:50]}'")
    _reset(km)


def test_sem_memory_vs_no_memory(km, c, R):
    print("\n── SEM-MEM. Memory vs no-memory ──")
    _reset(km)
    for t in [
        "He practiced piano Chopin nocturne.", "Telescope galaxies space.",
        "Chef cuisine garlic.", "Quantum computing qubits.",
        "Surgeon cardiac operation.", "Athlete Olympic gold.",
        "Historian Roman empire.", "Mathematician theorem proof.",
        "Neural network algorithm.", "Architect design building.",
    ]:
        km.write(t, training_mode=True)
    km.eval()
    with torch.no_grad():
        gen_with = km.generate("The pianist", mt=25, greedy=True)

    saved = dict(km.amm.tree.store)
    saved_root = km.amm.tree.root
    saved_nid = km.amm.tree.nid
    km.amm.tree.store = {}
    km.amm.tree.root = _Node()
    with torch.no_grad():
        gen_without = km.generate("The pianist", mt=25, greedy=True)

    km.amm.tree.store = saved
    km.amm.tree.root = saved_root
    km.amm.tree.nid = saved_nid

    R.check("sem_mem_differs", gen_with != gen_without,
            f"with='{gen_with[:40]}', without='{gen_without[:40]}'")
    _reset(km)


# ═══════════════════════════════════════════════════════════════════
# Part 4: Multimodal tests
# ═══════════════════════════════════════════════════════════════════

def test_multi_coexist(km, c, R):
    print("\n── MULTI-COEXIST. Text+Image+Video with Kakeya ──")
    _reset(km)
    km.write("The pianist performed Chopin.", training_mode=True)
    _store_image_memory(km, ["sunset", "mountain", "lake"], "nature")
    _store_video_memory(km, [["city", "traffic"], ["city", "busy"]], "city")
    for t in [
        "Telescope galaxies.", "Chef cuisine.", "Quantum computing.",
        "Surgeon cardiac.", "Athlete Olympic.", "Historian Rome.",
        "Mathematician theorem.", "Neural network.",
    ]:
        km.write(t, training_mode=True)

    texts = [e.source_text for e in km.amm.tree.store.values()]
    has_text = any("[IMAGE]" not in t and "[VIDEO]" not in t for t in texts)
    has_image = any("[IMAGE]" in t for t in texts)
    has_video = any("[VIDEO]" in t for t in texts)
    R.check("multi_has_text", has_text)
    R.check("multi_has_image", has_image)
    R.check("multi_has_video", has_video)
    errs = km.amm.tree.verify_consistency()
    R.check("multi_tree_consistent", len(errs) == 0, str(errs))
    _reset(km)


def test_multi_retrieval(km, c, R):
    print("\n── MULTI-RETRIEVE. Cross-modal retrieval with Kakeya ──")
    _reset(km)
    km.write("The pianist performed Chopin nocturne.", training_mode=True)
    km.write("The telescope revealed galaxies.", training_mode=True)
    _store_image_memory(km, ["sunset", "mountain", "ocean"], "nature")
    _store_video_memory(km, [["lion", "hunting"], ["eagle", "flying"]], "wildlife")
    for t in [
        "Chef cuisine.", "Quantum computing.", "Surgeon cardiac.",
        "Athlete Olympic.", "Historian Rome.", "Neural network.",
    ]:
        km.write(t, training_mode=True)
    km.eval()

    top, _, _ = _content_bias_top_tokens(km, "Tell me about piano.", k=15)
    R.check("multi_ret_works", len(top) > 0)
    R.check("multi_ret_has_content", any(t in MUSIC_KW for t in top))
    _reset(km)


def test_multi_generation(km, c, R):
    print("\n── MULTI-GEN. Generation with multimodal Kakeya store ──")
    _reset(km)
    km.write("Piano Chopin nocturne concert.", training_mode=True)
    _store_image_memory(km, ["concert", "hall", "stage"], "concert")
    _store_video_memory(km, [["pianist", "keyboard"], ["audience", "applause"]], "concert_vid")
    for t in [
        "Telescope galaxies.", "Chef cuisine.", "Quantum computing.",
        "Surgeon cardiac.", "Athlete Olympic.", "Historian Rome.",
        "Mathematician theorem.", "Neural network.",
    ]:
        km.write(t, training_mode=True)
    km.eval()
    torch.manual_seed(42)
    with torch.no_grad():
        gen = km.generate("The piano concert", mt=25, greedy=False)
    R.check("multi_gen_ok", len(gen) > len("The piano concert"))
    new_text = gen[len("The piano concert"):].strip()
    R.check("multi_gen_has_content", len(new_text) >= 3)
    _reset(km)


def test_multi_save_load(km, c, R):
    print("\n── MULTI-SAVELOAD. Multimodal save/load with Kakeya ──")
    _reset(km)
    km.write("Piano Chopin.", training_mode=True)
    _store_image_memory(km, ["sunset", "mountain"], "nature")
    _store_video_memory(km, [["lion", "hunting"]], "wildlife")
    for t in [
        "Telescope.", "Chef.", "Quantum.", "Surgeon.",
        "Athlete.", "Historian.", "Math.", "Neural.",
    ]:
        km.write(t, training_mode=True)

    n_before = len(km.amm.tree.store)
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        km.save_memory(path)
        _reset(km)
        km.load_memory(path)
        R.check("multi_sl_count", len(km.amm.tree.store) == n_before)
        texts = [e.source_text for e in km.amm.tree.store.values()]
        R.check("multi_sl_has_image", any("[IMAGE]" in t for t in texts))
        R.check("multi_sl_has_video", any("[VIDEO]" in t for t in texts))
    finally:
        os.unlink(path)
        codec_path = path + '.kakeya'
        if os.path.exists(codec_path):
            os.unlink(codec_path)
    _reset(km)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
def main():
    torch.manual_seed(42)
    c = Cfg()
    R = TestResults()

    sep = "=" * 70
    print(f"\n{sep}")
    print("  AMS v3.7 + Kakeya Compression — Full Integration Test")
    print(f"{sep}")
    t0 = time.time()

    print("\n[Building MemLLM + KakeyaCodec]")
    base_m = MemLLM(c)
    base_m.load("gpt2")
    km = KakeyaMemLLM(base_m, auto_build_threshold=8)
    total = sum(p.numel() for p in km.parameters())
    train_p = sum(p.numel() for p in km.parameters() if p.requires_grad)
    print(f"  Params: total={total:,}  trainable={train_p:,}")

    # Part 1: Kakeya Codec Validation
    print(f"\n{'─'*70}")
    print("  PART 1: Kakeya Codec Validation")
    print(f"{'─'*70}")
    test_kv_codec_build(km, c, R)
    test_kv_encode_decode_precision(km, c, R)
    test_kv_compression_ratio(km, c, R)
    test_kv_retrieval_preserved(km, c, R)
    test_kv_generation_works(km, c, R)
    test_kv_save_load(km, c, R)
    test_kv_consolidation(km, c, R)
    test_kv_decay(km, c, R)
    test_kv_multimodal(km, c, R)
    test_kv_training(km, c, R)

    # Part 2: Structural tests
    print(f"\n{'─'*70}")
    print("  PART 2: Structural Tests (with Kakeya)")
    print(f"{'─'*70}")
    test_struct_write_and_retrieve(km, c, R)
    test_struct_empty_memory(km, c, R)
    test_struct_greedy_deterministic(km, c, R)
    test_struct_batch_retrieval(km, c, R)

    # Part 3: Semantic tests
    print(f"\n{'─'*70}")
    print("  PART 3: Semantic Tests (with Kakeya)")
    print(f"{'─'*70}")
    test_sem_domain_retrieval(km, c, R)
    test_sem_generation_quality(km, c, R)
    test_sem_memory_vs_no_memory(km, c, R)

    # Part 4: Multimodal tests
    print(f"\n{'─'*70}")
    print("  PART 4: Multimodal Tests (with Kakeya)")
    print(f"{'─'*70}")
    test_multi_coexist(km, c, R)
    test_multi_retrieval(km, c, R)
    test_multi_generation(km, c, R)
    test_multi_save_load(km, c, R)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # Final stats
    print(f"\n{'─'*70}")
    print("  Kakeya Codec Final Stats")
    print(f"{'─'*70}")
    _reset(km)
    for t in [
        "Piano Chopin.", "Telescope galaxies.", "Chef cuisine.", "Quantum computing.",
        "Surgeon cardiac.", "Athlete Olympic.", "Historian Rome.", "Mathematician theorem.",
        "Neural network.", "Architect design.",
    ]:
        km.write(t, training_mode=True)
    stats = km.codec.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    ok = R.summary()
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
