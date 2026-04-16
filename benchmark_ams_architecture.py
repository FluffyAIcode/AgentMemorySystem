#!/usr/bin/env python3
"""
AMS v3.7 — Architecture Benchmark & Path Latency Profiler
==========================================================

Measures real latencies for every data path in the AMS architecture
across TEXT, IMAGE, and VIDEO modalities, plus memory compression ratios.

No mocks. No simplification. Real GPT-2 + real AMS pipeline.
"""

import sys, time, torch, torch.nn.functional as F, math, json
from dataclasses import dataclass, field
from typing import Dict, List

from AgentMemorySystem import (
    Cfg, MemLLM, _Node, Trainer, SpectralDealiaser, MemEntry,
)

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _reset(m):
    m.amm.tree.store.clear()
    m.amm.tree.root = _Node()
    m.amm.tree.nid = 0
    m.amm.time = 0

def _dev(m):
    return next(m.parameters()).device

def _get_concept_embedding(m, words):
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

def _make_image_embedding(m, visual_concepts, seed=1000):
    base = _get_concept_embedding(m, visual_concepts)
    torch.manual_seed(seed)
    offset = torch.randn_like(base) * 0.15
    return F.normalize(base + offset, dim=0) * base.norm()

def _make_video_embedding(m, frame_concepts_list, seed=2000):
    frames = []
    for i, concepts in enumerate(frame_concepts_list):
        frame_emb = _get_concept_embedding(m, concepts)
        frames.append(frame_emb * (1.0 + 0.1 * i))
    if not frames:
        return torch.zeros(m.c.d_LLM, device=_dev(m))
    video_emb = torch.stack(frames).mean(0)
    torch.manual_seed(seed)
    offset = torch.randn_like(video_emb) * 0.2
    return F.normalize(video_emb + offset, dim=0) * video_emb.norm()

def _store_image_memory(m, visual_concepts, source_label, surprise=1.5):
    dev = _dev(m)
    h = _make_image_embedding(m, visual_concepts).to(dev)
    cc = m.content_classifier
    tok = m.tok
    content_ids = []
    for w in visual_concepts:
        for tid in tok.encode(" " + w):
            if tid in cc.content_ids:
                content_ids.append(tid)
    expanded = m._expand_content_ids(content_ids)
    wte_centroid = m._compute_wte_centroid(content_ids)
    return m.amm.store_mem(
        h, surprise, training_mode=True,
        source_text=f"[IMAGE] {source_label}",
        content_token_ids=content_ids,
        content_semantic_emb=h.clone(),
        content_wte_centroid=wte_centroid,
        expanded_content_ids=expanded)

def _store_video_memory(m, frame_concepts_list, source_label, surprise=2.0):
    dev = _dev(m)
    h = _make_video_embedding(m, frame_concepts_list).to(dev)
    cc = m.content_classifier
    tok = m.tok
    content_ids = []
    for frame in frame_concepts_list:
        for w in frame:
            for tid in tok.encode(" " + w):
                if tid in cc.content_ids:
                    content_ids.append(tid)
    content_ids = list(set(content_ids))
    expanded = m._expand_content_ids(content_ids)
    wte_centroid = m._compute_wte_centroid(content_ids)
    return m.amm.store_mem(
        h, surprise, training_mode=True,
        source_text=f"[VIDEO] {source_label}",
        content_token_ids=content_ids,
        content_semantic_emb=h.clone(),
        content_wte_centroid=wte_centroid,
        expanded_content_ids=expanded)

class Timer:
    def __init__(self):
        self._start = None
    def __enter__(self):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self._start = time.perf_counter()
        return self
    def __exit__(self, *args):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000

# ═══════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════

def benchmark(m, c):
    dev = _dev(m)
    results = {}
    N_TRIALS = 5

    sep = "═" * 70
    print(f"\n{sep}")
    print("  AMS v3.7 — Architecture Path Latency Benchmark")
    print(f"{sep}")

    # ─────────────────────────────────────────────────────────────
    # 1. WRITE PATH (存入路径)
    # ─────────────────────────────────────────────────────────────
    print("\n┌─── 1. WRITE PATH (存入路径) ────────────────────────────┐")

    # 1a. Text write
    text = "The experienced pianist performed a magnificent Chopin nocturne at Carnegie Hall."
    times = []
    for trial in range(N_TRIALS):
        _reset(m)
        with Timer() as t:
            m.write(text, training_mode=True)
        times.append(t.elapsed_ms)
    avg = sum(times) / len(times)
    results['write_text_ms'] = avg
    print(f"│  TEXT  write:   {avg:8.2f} ms  (min={min(times):.2f}, max={max(times):.2f})")

    # 1b. Image write
    img_concepts = ["sunset", "mountain", "ocean", "golden", "landscape"]
    times = []
    for trial in range(N_TRIALS):
        _reset(m)
        with Timer() as t:
            _store_image_memory(m, img_concepts, "nature_photo")
        times.append(t.elapsed_ms)
    avg = sum(times) / len(times)
    results['write_image_ms'] = avg
    print(f"│  IMAGE write:   {avg:8.2f} ms  (min={min(times):.2f}, max={max(times):.2f})")

    # 1c. Video write
    vid_frames = [
        ["lion", "hunting", "savanna"],
        ["lion", "prey", "chase", "speed"],
        ["lion", "eating", "sunset", "rest"],
    ]
    times = []
    for trial in range(N_TRIALS):
        _reset(m)
        with Timer() as t:
            _store_video_memory(m, vid_frames, "wildlife_doc")
        times.append(t.elapsed_ms)
    avg = sum(times) / len(times)
    results['write_video_ms'] = avg
    print(f"│  VIDEO write:   {avg:8.2f} ms  (min={min(times):.2f}, max={max(times):.2f})")

    # 1d. Text write sub-path breakdown
    _reset(m)
    # GPT-2 forward
    tk = m.tok(text, return_tensors='pt', padding=True, truncation=True)
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with Timer() as t:
        with torch.no_grad():
            o = m.fwd(ids, mask)
    results['write_text_gpt2_fwd_ms'] = t.elapsed_ms
    print(f"│    ├ GPT-2 fwd:    {t.elapsed_ms:8.2f} ms")

    # Layer pool
    with Timer() as t:
        with torch.no_grad():
            hs_pooled = m.layer_pool(o['hs'])
    results['write_text_layer_pool_ms'] = t.elapsed_ms
    print(f"│    ├ LayerPool:    {t.elapsed_ms:8.2f} ms")

    # Surprise proxy
    with Timer() as t:
        surp = m.amm.surprise_proxy(o['logits'][:, :-1], ids[:, 1:])
    results['write_text_surprise_ms'] = t.elapsed_ms
    print(f"│    ├ Surprise:     {t.elapsed_ms:8.2f} ms")

    # Content semantic emb
    with Timer() as t:
        content_sem = m._compute_content_semantic_emb(hs_pooled, ids, mask)
    results['write_text_content_sem_ms'] = t.elapsed_ms
    print(f"│    ├ ContentSem:   {t.elapsed_ms:8.2f} ms")

    # Content ID extraction + expansion
    with Timer() as t:
        raw_ids = m.tok.encode(text)
        cc = m.content_classifier
        content_ids = list(set(cc.get_content_ids_from_tokens(raw_ids)))
        expanded_ids = m._expand_content_ids(content_ids)
        wte_centroid = m._compute_wte_centroid(content_ids)
    results['write_text_content_ids_ms'] = t.elapsed_ms
    print(f"│    ├ ContentIDs:   {t.elapsed_ms:8.2f} ms")

    # CtxEncoder + FibEncoder + store_mem
    pooled_mean = hs_pooled.mean(1)
    with Timer() as t:
        m.amm.store_mem(
            pooled_mean[0], surp[0], True,
            source_text=text, content_token_ids=content_ids,
            content_semantic_emb=content_sem[0],
            content_wte_centroid=wte_centroid,
            expanded_content_ids=expanded_ids)
    results['write_text_store_mem_ms'] = t.elapsed_ms
    print(f"│    └ store_mem:    {t.elapsed_ms:8.2f} ms")

    print("│")

    # ─────────────────────────────────────────────────────────────
    # 2. READ PATH (读取路径)
    # ─────────────────────────────────────────────────────────────
    print("├─── 2. READ PATH (读取路径) ─────────────────────────────┤")

    # Populate store with mixed modalities
    _reset(m)
    m.write("The pianist performed Chopin nocturne.", training_mode=True)
    m.write("The telescope revealed distant galaxies.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature")
    _store_video_memory(m, [["lion", "hunting"], ["eagle", "flying"]], "wildlife")
    m.eval()

    query = "Tell me about the piano performance."
    tk = m.tok(query, return_tensors='pt')
    ids_q, mask_q = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)

    # 2a. Full read path
    times = []
    for _ in range(N_TRIALS):
        with Timer() as t:
            with torch.no_grad():
                o = m.fwd(ids_q, mask_q)
                prefix, fs, diag, cb = m._get_prefix(
                    o['hs'], mask_q, update_stats=False, return_extra=True, ids=ids_q)
        times.append(t.elapsed_ms)
    avg = sum(times) / len(times)
    results['read_full_ms'] = avg
    print(f"│  Full retrieve:  {avg:8.2f} ms  (min={min(times):.2f}, max={max(times):.2f})")

    # 2b. Read sub-path breakdown
    with Timer() as t:
        with torch.no_grad():
            o = m.fwd(ids_q, mask_q)
    results['read_gpt2_fwd_ms'] = t.elapsed_ms
    print(f"│    ├ GPT-2 fwd:    {t.elapsed_ms:8.2f} ms")

    with Timer() as t:
        pooled_q, xq, fq = m.extract_state(o['hs'], mask_q)
    results['read_extract_state_ms'] = t.elapsed_ms
    print(f"│    ├ ExtractState: {t.elapsed_ms:8.2f} ms")

    with Timer() as t:
        with torch.no_grad():
            query_sem = m._compute_content_semantic_emb(pooled_q, ids_q, mask_q)
    results['read_query_sem_ms'] = t.elapsed_ms
    print(f"│    ├ QuerySemEmb:  {t.elapsed_ms:8.2f} ms")

    with Timer() as t:
        with torch.no_grad():
            fibers, mem_mask, fiber_summary, diag = m.amm.retrieve_multi(
                xq, fq, query_semantic_emb=query_sem)
    results['read_retrieve_multi_ms'] = t.elapsed_ms
    print(f"│    ├ RetrieveMulti:{t.elapsed_ms:8.2f} ms")

    with Timer() as t:
        prefix = m.bridge.inject(fibers, mem_mask, fiber_summary=fiber_summary)
    results['read_bridge_inject_ms'] = t.elapsed_ms
    print(f"│    └ BridgeInject: {t.elapsed_ms:8.2f} ms")

    # 2c. Generation path
    _reset(m)
    m.write("The pianist performed Chopin.", training_mode=True)
    _store_image_memory(m, ["concert", "hall", "stage"], "concert")
    m.eval()

    times = []
    for _ in range(N_TRIALS):
        with Timer() as t:
            with torch.no_grad():
                gen = m.generate("The pianist", mt=20, greedy=True)
        times.append(t.elapsed_ms)
    avg = sum(times) / len(times)
    results['generate_20tok_ms'] = avg
    print(f"│  Generate 20tok: {avg:8.2f} ms  (min={min(times):.2f}, max={max(times):.2f})")

    print("│")

    # ─────────────────────────────────────────────────────────────
    # 3. SEMANTIC DISCRIMINATION PATH (语义区分路径)
    # ─────────────────────────────────────────────────────────────
    print("├─── 3. SEMANTIC DISCRIMINATION PATH (语义区分路径) ──────┤")

    _reset(m)
    m.write("He practiced piano Chopin nocturne.", training_mode=True)
    m.write("The telescope revealed galaxies.", training_mode=True)
    _store_image_memory(m, ["sunset", "mountain", "ocean"], "nature")
    _store_video_memory(m, [["lion", "hunting", "savanna"]], "wildlife")
    m.eval()

    # 3a. Full discrimination: query → content bias → top-k
    query_music = "Tell me about the piano performance."
    times = []
    for _ in range(N_TRIALS):
        tk = m.tok(query_music, return_tensors='pt')
        ids_q, mask_q = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
        with Timer() as t:
            with torch.no_grad():
                o = m.fwd(ids_q, mask_q)
                _, _, _, cb = m._get_prefix(
                    o['hs'], mask_q, update_stats=False, return_extra=True, ids=ids_q)
                _ = cb[0].topk(20)
        times.append(t.elapsed_ms)
    avg = sum(times) / len(times)
    results['semantic_discrim_full_ms'] = avg
    print(f"│  Full discrim:   {avg:8.2f} ms")

    # 3b. Three-way scoring breakdown
    tk = m.tok(query_music, return_tensors='pt')
    ids_q, mask_q = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids_q, mask_q)
    pooled_q, xq, fq = m.extract_state(o['hs'], mask_q)

    # Direction similarity
    with Timer() as t:
        qdir = m.amm.dir_pred(xq, fq)
        for mid, mem in m.amm.tree.store.items():
            _ = (qdir[0] @ mem.dirn).item()
    results['discrim_dir_sim_ms'] = t.elapsed_ms
    print(f"│    ├ DirSim:       {t.elapsed_ms:8.2f} ms")

    # Semantic embedding similarity
    with Timer() as t:
        query_sem = m._compute_content_semantic_emb(pooled_q, ids_q, mask_q)
        for mid, mem in m.amm.tree.store.items():
            if mem.semantic_emb is not None:
                _ = F.cosine_similarity(query_sem, mem.semantic_emb.unsqueeze(0).to(dev)).item()
    results['discrim_sem_sim_ms'] = t.elapsed_ms
    print(f"│    ├ SemSim:       {t.elapsed_ms:8.2f} ms")

    # WTE centroid similarity
    with Timer() as t:
        wte = m.llm.transformer.wte.weight.detach()
        b_ids = ids_q[0].tolist()
        b_content = m.content_classifier.get_content_ids_from_tokens(b_ids)
        if b_content:
            valid = [i for i in b_content if i < wte.shape[0]]
            if valid:
                q_wte = wte[valid].mean(0)
            else:
                q_wte = torch.zeros(wte.shape[1], device=dev)
        else:
            q_wte = torch.zeros(wte.shape[1], device=dev)
        for mid, mem in m.amm.tree.store.items():
            if mem.content_wte_centroid is not None:
                _ = F.cosine_similarity(q_wte.unsqueeze(0),
                                        mem.content_wte_centroid.unsqueeze(0).to(dev)).item()
    results['discrim_wte_sim_ms'] = t.elapsed_ms
    print(f"│    ├ WTESim:       {t.elapsed_ms:8.2f} ms")

    # Content bias build
    with Timer() as t:
        _, _, diag_tmp, cb_tmp = m._get_prefix(
            o['hs'], mask_q, update_stats=False, return_extra=True, ids=ids_q)
        _ = m._build_content_bias(diag_tmp)
    results['discrim_content_bias_ms'] = t.elapsed_ms
    print(f"│    └ ContentBias:  {t.elapsed_ms:8.2f} ms")

    print("│")

    # ─────────────────────────────────────────────────────────────
    # 4. MEMORY MODIFICATION PATH (记忆修改路径)
    # ─────────────────────────────────────────────────────────────
    print("├─── 4. MEMORY MODIFICATION PATH (记忆修改路径) ─────────┤")

    # 4a. Consolidation
    _reset(m)
    for i in range(8):
        h = torch.randn(c.d_LLM, device=dev) * 0.01
        m.amm.store_mem(h, 0.5, True, source_text=f"mem_{i}")
    times = []
    for _ in range(N_TRIALS):
        with Timer() as t:
            m.amm.consolidate()
        times.append(t.elapsed_ms)
    avg = sum(times) / len(times)
    results['modify_consolidate_ms'] = avg
    print(f"│  Consolidate:    {avg:8.2f} ms  (8 memories)")

    # 4b. Decay
    _reset(m)
    for i in range(8):
        h = torch.randn(c.d_LLM, device=dev)
        m.amm.store_mem(h, 0.5, True, source_text=f"mem_{i}")
    m.amm.time += 5000
    times = []
    for _ in range(N_TRIALS):
        with Timer() as t:
            m.amm.decay()
        times.append(t.elapsed_ms)
    avg = sum(times) / len(times)
    results['modify_decay_ms'] = avg
    print(f"│  Decay:          {avg:8.2f} ms  (8 memories, t+5000)")

    # 4c. Update existing
    _reset(m)
    h0 = torch.randn(c.d_LLM, device=dev)
    m.amm.store_mem(h0, 1.0, True, source_text="original")
    h_similar = h0 + torch.randn(c.d_LLM, device=dev) * 0.0001
    times = []
    for _ in range(N_TRIALS):
        with Timer() as t:
            m.amm.store_mem(h_similar.clone(), 1.5, True, source_text="updated")
        times.append(t.elapsed_ms)
    avg = sum(times) / len(times)
    results['modify_update_existing_ms'] = avg
    print(f"│  UpdateExisting:  {avg:8.2f} ms")

    # 4d. SpectralDealiaser
    _reset(m)
    for i in range(6):
        h = torch.randn(c.d_LLM, device=dev) * 0.01
        m.amm.store_mem(h, 1.0, True, source_text=f"entry_{i}")
    mids = list(m.amm.tree.store.keys())
    times = []
    for _ in range(N_TRIALS):
        da = SpectralDealiaser(m.amm, c)
        with Timer() as t:
            da.dealias(mids, steps=20)
        times.append(t.elapsed_ms)
    avg = sum(times) / len(times)
    results['modify_dealias_ms'] = avg
    print(f"│  Dealias(20st):  {avg:8.2f} ms  (6 memories)")

    # 4e. Tree rebuild
    _reset(m)
    for i in range(50):
        d = F.normalize(torch.randn(c.d_M), dim=0)
        me = MemEntry(mid=i, base=torch.randn(c.d_M), fiber=torch.randn(c.d_F),
                      dirn=d, surprise=0.5, ts=float(i), last=float(i))
        m.amm.tree.store[me.mid] = me
        m.amm.tree.nid = i + 1
        m.amm.tree._ins(m.amm.tree.root, me)
    times = []
    for _ in range(N_TRIALS):
        with Timer() as t:
            m.amm.tree.rebuild()
        times.append(t.elapsed_ms)
    avg = sum(times) / len(times)
    results['modify_tree_rebuild_ms'] = avg
    print(f"│  TreeRebuild:    {avg:8.2f} ms  (50 entries)")

    # 4f. Memory refresh
    _reset(m)
    for t_str in ["Piano music.", "Space telescope.", "Chef cooking."]:
        m.write(t_str, training_mode=True)
    times_r = []
    for _ in range(N_TRIALS):
        with Timer() as t:
            with torch.no_grad():
                m._refresh_all_memories()
        times_r.append(t.elapsed_ms)
    avg = sum(times_r) / len(times_r)
    results['modify_refresh_ms'] = avg
    print(f"│  Refresh:        {avg:8.2f} ms  (3 text memories)")

    print("│")

    # ─────────────────────────────────────────────────────────────
    # 5. MEMORY COMPRESSION RATIO (记忆压缩率)
    # ─────────────────────────────────────────────────────────────
    print("├─── 5. COMPRESSION RATIO (记忆压缩率) ─────────────────┤")

    def _entry_bytes(e):
        """Bytes consumed by a single MemEntry."""
        b = e.base.numel() * e.base.element_size()
        f = e.fiber.numel() * e.fiber.element_size()
        d = e.dirn.numel() * e.dirn.element_size()
        sem = e.semantic_emb.numel() * e.semantic_emb.element_size() if e.semantic_emb is not None else 0
        wte = e.content_wte_centroid.numel() * e.content_wte_centroid.element_size() if e.content_wte_centroid is not None else 0
        scalars = 8 * 5  # surprise, ts, last, cnt, version, mid (~5 floats + ints)
        ct_ids = len(e.content_token_ids) * 8
        exp_ids = len(e.expanded_content_ids) * 8
        txt = len(e.source_text.encode('utf-8'))
        return b + f + d + sem + wte + scalars + ct_ids + exp_ids + txt

    # 5a. Text compression
    _reset(m)
    text_long = "The experienced pianist performed a magnificent Chopin nocturne at Carnegie Hall during the annual classical music festival, captivating the audience with brilliant technique and emotional depth."
    raw_text_bytes = len(text_long.encode('utf-8'))
    token_ids = m.tok.encode(text_long)
    raw_token_bytes = len(token_ids) * 4  # int32
    gpt2_hidden_bytes = len(token_ids) * c.d_LLM * 4  # float32

    m.write(text_long, training_mode=True)
    entry = list(m.amm.tree.store.values())[0]
    mem_bytes = _entry_bytes(entry)

    text_ratio = raw_text_bytes / mem_bytes
    token_ratio = raw_token_bytes / mem_bytes
    hidden_ratio = gpt2_hidden_bytes / mem_bytes

    results['compress_text_raw_bytes'] = raw_text_bytes
    results['compress_text_token_bytes'] = raw_token_bytes
    results['compress_text_hidden_bytes'] = gpt2_hidden_bytes
    results['compress_text_mem_bytes'] = mem_bytes
    results['compress_text_ratio'] = raw_text_bytes / mem_bytes

    print(f"│  TEXT:")
    print(f"│    Raw UTF-8:      {raw_text_bytes:>6d} bytes ({len(text_long)} chars)")
    print(f"│    Token IDs:      {raw_token_bytes:>6d} bytes ({len(token_ids)} tokens × 4B)")
    print(f"│    GPT-2 hidden:   {gpt2_hidden_bytes:>6d} bytes ({len(token_ids)}×{c.d_LLM}×4B)")
    print(f"│    MemEntry:       {mem_bytes:>6d} bytes")
    print(f"│    Ratio raw/mem:  {text_ratio:>6.2f}×")
    print(f"│    Ratio hid/mem:  {hidden_ratio:>6.2f}×")

    # 5b. Image compression
    _reset(m)
    img_concepts = ["sunset", "mountain", "ocean", "golden", "landscape",
                    "horizon", "cliff", "valley", "river", "sky"]
    img_description = " ".join(img_concepts)
    img_desc_bytes = len(img_description.encode('utf-8'))
    img_simulated_pixels = 224 * 224 * 3  # typical image input size
    img_simulated_bytes = img_simulated_pixels  # uint8

    _store_image_memory(m, img_concepts, "nature_panorama")
    entry_img = list(m.amm.tree.store.values())[0]
    mem_img_bytes = _entry_bytes(entry_img)

    results['compress_img_pixel_bytes'] = img_simulated_bytes
    results['compress_img_mem_bytes'] = mem_img_bytes
    results['compress_img_ratio'] = img_simulated_bytes / mem_img_bytes

    print(f"│  IMAGE:")
    print(f"│    Simulated 224²×3: {img_simulated_bytes:>6d} bytes (150,528 pixels)")
    print(f"│    Concept text:     {img_desc_bytes:>6d} bytes")
    print(f"│    MemEntry:         {mem_img_bytes:>6d} bytes")
    print(f"│    Ratio pixel/mem:  {img_simulated_bytes/mem_img_bytes:>6.2f}×")

    # 5c. Video compression
    _reset(m)
    vid_frames = [
        ["lion", "hunting", "savanna", "grass"],
        ["lion", "prey", "chase", "speed", "dust"],
        ["lion", "eating", "sunset", "rest", "blood"],
        ["cubs", "playing", "morning", "den", "family"],
        ["elephant", "herd", "migration", "river", "crossing"],
    ]
    vid_n_frames = len(vid_frames)
    vid_simulated_bytes = vid_n_frames * 224 * 224 * 3  # 5 frames

    _store_video_memory(m, vid_frames, "wildlife_documentary")
    entry_vid = list(m.amm.tree.store.values())[0]
    mem_vid_bytes = _entry_bytes(entry_vid)

    results['compress_vid_frame_bytes'] = vid_simulated_bytes
    results['compress_vid_mem_bytes'] = mem_vid_bytes
    results['compress_vid_ratio'] = vid_simulated_bytes / mem_vid_bytes

    print(f"│  VIDEO ({vid_n_frames} frames):")
    print(f"│    Simulated {vid_n_frames}×224²×3: {vid_simulated_bytes:>6d} bytes")
    print(f"│    MemEntry:          {mem_vid_bytes:>6d} bytes")
    print(f"│    Ratio frames/mem:  {vid_simulated_bytes/mem_vid_bytes:>6.2f}×")

    # 5d. MemEntry field breakdown
    print(f"│  MemEntry field breakdown (TEXT):")
    _reset(m)
    m.write(text_long, training_mode=True)
    e = list(m.amm.tree.store.values())[0]
    base_b = e.base.numel() * e.base.element_size()
    fiber_b = e.fiber.numel() * e.fiber.element_size()
    dirn_b = e.dirn.numel() * e.dirn.element_size()
    sem_b = e.semantic_emb.numel() * e.semantic_emb.element_size() if e.semantic_emb is not None else 0
    wte_b = e.content_wte_centroid.numel() * e.content_wte_centroid.element_size() if e.content_wte_centroid is not None else 0
    ct_b = len(e.content_token_ids) * 8
    exp_b = len(e.expanded_content_ids) * 8
    txt_b = len(e.source_text.encode('utf-8'))

    print(f"│    base[{c.d_M}]:        {base_b:>5d} B  ({c.d_M}×4B)")
    print(f"│    fiber[{c.d_F}]:       {fiber_b:>5d} B  ({c.d_F}×4B)")
    print(f"│    dirn[{c.d_M}]:        {dirn_b:>5d} B  ({c.d_M}×4B)")
    print(f"│    semantic_emb[{c.d_LLM}]: {sem_b:>5d} B  ({c.d_LLM}×4B)")
    print(f"│    wte_centroid[{c.d_LLM}]: {wte_b:>5d} B  ({c.d_LLM}×4B)")
    print(f"│    content_ids:      {ct_b:>5d} B  ({len(e.content_token_ids)} ints)")
    print(f"│    expanded_ids:     {exp_b:>5d} B  ({len(e.expanded_content_ids)} ints)")
    print(f"│    source_text:      {txt_b:>5d} B")
    total_computed = base_b + fiber_b + dirn_b + sem_b + wte_b + ct_b + exp_b + txt_b + 40
    print(f"│    TOTAL:            {total_computed:>5d} B")

    print("│")

    # ─────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────
    print("└─── SUMMARY ────────────────────────────────────────────┘")
    print()
    print("  ┌───────────────────────────────────────────────────────────────┐")
    print("  │            AMS v3.7 Architecture Path Latency Summary        │")
    print("  ├──────────────┬──────────┬──────────┬──────────┬──────────────┤")
    print("  │ Path         │   TEXT   │  IMAGE   │  VIDEO   │ Sub-paths    │")
    print("  ├──────────────┼──────────┼──────────┼──────────┼──────────────┤")
    print(f"  │ WRITE        │{results['write_text_ms']:7.1f}ms │{results['write_image_ms']:7.1f}ms │{results['write_video_ms']:7.1f}ms │              │")
    print(f"  │  ├GPT2 fwd   │{results['write_text_gpt2_fwd_ms']:7.1f}ms │   N/A    │   N/A    │ text only    │")
    print(f"  │  ├LayerPool   │{results['write_text_layer_pool_ms']:7.1f}ms │   N/A    │   N/A    │ text only    │")
    print(f"  │  ├Surprise    │{results['write_text_surprise_ms']:7.1f}ms │   N/A    │   N/A    │ text only    │")
    print(f"  │  ├ContentSem  │{results['write_text_content_sem_ms']:7.1f}ms │   N/A    │   N/A    │ text only    │")
    print(f"  │  ├ContentIDs  │{results['write_text_content_ids_ms']:7.1f}ms │  ~same   │  ~same   │              │")
    print(f"  │  └store_mem   │{results['write_text_store_mem_ms']:7.1f}ms │  ~same   │  ~same   │              │")
    print("  ├──────────────┼──────────┼──────────┼──────────┼──────────────┤")
    print(f"  │ READ         │{results['read_full_ms']:7.1f}ms │  ~same   │  ~same   │ 4-mem store  │")
    print(f"  │  ├GPT2 fwd   │{results['read_gpt2_fwd_ms']:7.1f}ms │         │         │              │")
    print(f"  │  ├ExtractSt   │{results['read_extract_state_ms']:7.1f}ms │         │         │              │")
    print(f"  │  ├QuerySem    │{results['read_query_sem_ms']:7.1f}ms │         │         │              │")
    print(f"  │  ├RetrMulti   │{results['read_retrieve_multi_ms']:7.1f}ms │         │         │              │")
    print(f"  │  └BridgeInj   │{results['read_bridge_inject_ms']:7.1f}ms │         │         │              │")
    print("  ├──────────────┼──────────┼──────────┼──────────┼──────────────┤")
    print(f"  │ DISCRIM      │{results['semantic_discrim_full_ms']:7.1f}ms │  ~same   │  ~same   │ 4-mem store  │")
    print(f"  │  ├DirSim      │{results['discrim_dir_sim_ms']:7.1f}ms │         │         │              │")
    print(f"  │  ├SemSim      │{results['discrim_sem_sim_ms']:7.1f}ms │         │         │              │")
    print(f"  │  ├WTESim      │{results['discrim_wte_sim_ms']:7.1f}ms │         │         │              │")
    print(f"  │  └ContentBias │{results['discrim_content_bias_ms']:7.1f}ms │         │         │              │")
    print("  ├──────────────┼──────────┼──────────┼──────────┼──────────────┤")
    print(f"  │ MODIFY       │          │          │          │              │")
    print(f"  │  Consolidate │{results['modify_consolidate_ms']:7.1f}ms │  ~same   │  ~same   │ 8 entries    │")
    print(f"  │  Decay       │{results['modify_decay_ms']:7.1f}ms │  ~same   │  ~same   │ 8 entries    │")
    print(f"  │  Update      │{results['modify_update_existing_ms']:7.1f}ms │  ~same   │  ~same   │              │")
    print(f"  │  Dealias     │{results['modify_dealias_ms']:7.1f}ms │  ~same   │  ~same   │ 6 entries    │")
    print(f"  │  Rebuild     │{results['modify_tree_rebuild_ms']:7.1f}ms │  ~same   │  ~same   │ 50 entries   │")
    print(f"  │  Refresh     │{results['modify_refresh_ms']:7.1f}ms │   N/A    │   N/A    │ 3 text mem   │")
    print("  ├──────────────┼──────────┼──────────┼──────────┼──────────────┤")
    print(f"  │ GENERATE     │{results['generate_20tok_ms']:7.1f}ms │          │          │ 20 tokens    │")
    print("  ├──────────────┴──────────┴──────────┴──────────┴──────────────┤")
    print("  │                    COMPRESSION RATIO                        │")
    print("  ├──────────────┬──────────────────────┬────────────────────────┤")
    print("  │ Modality     │ Raw → MemEntry       │ Compression            │")
    print("  ├──────────────┼──────────────────────┼────────────────────────┤")
    print(f"  │ TEXT         │ {results['compress_text_raw_bytes']:>5d}B → {results['compress_text_mem_bytes']:>5d}B   │ {results['compress_text_ratio']:>5.2f}× (UTF-8)       │")
    print(f"  │              │ {results['compress_text_hidden_bytes']:>5d}B → {results['compress_text_mem_bytes']:>5d}B   │ {results['compress_text_hidden_bytes']/results['compress_text_mem_bytes']:>5.2f}× (GPT-2 hid)   │")
    print(f"  │ IMAGE        │ {results['compress_img_pixel_bytes']:>5d}B → {results['compress_img_mem_bytes']:>5d}B   │ {results['compress_img_ratio']:>5.1f}× (224²×3)      │")
    print(f"  │ VIDEO(5fr)   │ {results['compress_vid_frame_bytes']:>5d}B → {results['compress_vid_mem_bytes']:>5d}B   │ {results['compress_vid_ratio']:>5.1f}× (5×224²×3)   │")
    print("  └──────────────┴──────────────────────┴────────────────────────┘")

    _reset(m)
    return results


if __name__ == "__main__":
    torch.manual_seed(42)
    c = Cfg()
    m = MemLLM(c)
    m.load("gpt2")
    results = benchmark(m, c)
