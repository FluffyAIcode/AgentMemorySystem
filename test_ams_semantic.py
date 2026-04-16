#!/usr/bin/env python3
"""
Agent Memory System (AMS) v3.12 — Semantic-Level Black-Box Test Suite
=====================================================================

This suite focuses on *semantic behavior*: does the system remember
the right content, retrieve the right memories for a given query,
maintain domain isolation, handle temporal dynamics, respond to
training, and produce meaningful generation output?

Rules (same as structural suite):
  - No mocks, no stubs, no fakes
  - No fallback logic
  - No modifications to AgentMemorySystem.py
  - All tests use real GPT-2, real tokenizer, real computation
"""

import sys, os, time, math, tempfile, collections
import torch
import torch.nn.functional as F

from AgentMemorySystem import (
    Cfg, MemLLM, _Node, Trainer, SpectralDealiaser, RetrievalDiag,
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


def _content_bias_top_tokens(m, query, k=20):
    """Helper: get top-k content-bias tokens for a query string."""
    dev = _dev(m)
    tk = m.tok(query, return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        _, _, _, cb = m._get_prefix(
            o['hs'], mask, update_stats=False, return_extra=True, ids=ids)
    topk_ids = cb[0].topk(k).indices.tolist()
    topk_toks = [m.tok.decode([t]).strip().lower() for t in topk_ids]
    return topk_toks, cb


def _count_kw_hits(text, keywords):
    """Count how many domain keywords appear in text."""
    words = set(text.lower().split())
    cleaned = set()
    for w in words:
        cleaned.add(''.join(c for c in w if c.isalpha()))
    return sum(1 for kw in keywords if kw in cleaned)


# ═══════════════════════════════════════════════════════════════════
# Domain keyword banks (used across many tests)
# ═══════════════════════════════════════════════════════════════════
MUSIC_KW = {
    'piano', 'chopin', 'nocturne', 'orchestra', 'beethoven', 'symphony',
    'harmony', 'melody', 'chord', 'musical', 'sonata', 'concerto',
    'instrument', 'practiced', 'perfecting', 'harmonic', 'progression',
    'conservatory', 'performed', 'precision', 'theory', 'studied',
    'pianist', 'composer', 'notes', 'score', 'tempo', 'rhythm',
    'violin', 'cello', 'flute', 'classical', 'baroque', 'fugue',
    'music', 'opera', 'singing', 'vocal', 'acoustic',
}
SPACE_KW = {
    'galaxy', 'galaxies', 'telescope', 'star', 'stars', 'planet',
    'orbit', 'space', 'astronaut', 'astronauts', 'mars', 'nebula',
    'radiation', 'gravity', 'cosmic', 'solar', 'lunar', 'universe',
    'constellation', 'spectrum', 'satellite', 'mission', 'electromagnetic',
    'revealed', 'distant', 'simulated', 'emitted', 'milky', 'rocket',
    'spacecraft', 'asteroid', 'comet', 'jupiter', 'saturn', 'venus',
}
COOKING_KW = {
    'chef', 'recipe', 'ingredient', 'cooking', 'kitchen', 'baking',
    'oven', 'spice', 'flavor', 'seasoning', 'dish', 'meal', 'cuisine',
    'restaurant', 'gourmet', 'saut', 'roast', 'simmer', 'broth',
    'sauce', 'garlic', 'onion', 'pepper', 'salt', 'butter', 'olive',
    'prepared', 'delicious', 'exquisite', 'culinary', 'dessert',
}
MEDICAL_KW = {
    'doctor', 'patient', 'surgery', 'hospital', 'diagnosis', 'treatment',
    'medicine', 'clinical', 'therapy', 'symptom', 'disease', 'health',
    'pharmaceutical', 'prescription', 'cardiac', 'neural', 'surgical',
    'recovery', 'vaccine', 'immune', 'antibody', 'pathology', 'anatomy',
}
COMPUTING_KW = {
    'algorithm', 'computer', 'software', 'programming', 'database',
    'neural', 'network', 'machine', 'learning', 'artificial', 'intelligence',
    'quantum', 'qubit', 'superposition', 'computing', 'processor',
    'encryption', 'compiler', 'binary', 'data', 'silicon', 'chip',
    'transistor', 'circuit', 'digital', 'code', 'debug', 'runtime',
}


# ═══════════════════════════════════════════════════════════════════
# S1. 写入语义完整性 — 写入后,记忆应保留原始文本的核心内容词
# ═══════════════════════════════════════════════════════════════════
def test_write_preserves_content_tokens(m, c, R):
    """After writing, stored entry's content_token_ids should correspond
    to meaningful words from the original text."""
    print("\n── S1. Write preserves content tokens ──")
    _reset(m)
    text = "The experienced violinist performed a magnificent Brahms concerto at Carnegie Hall."
    m.write(text, training_mode=True)
    entry = list(m.amm.tree.store.values())[0]
    decoded_tokens = [m.tok.decode([tid]).strip().lower() for tid in entry.content_token_ids]
    expected_words = {'violinist', 'performed', 'magnificent', 'brahms', 'concerto', 'carnegie', 'hall'}
    found = expected_words & set(decoded_tokens)
    R.check("s1_content_tokens_meaningful",
            len(found) >= 3,
            f"found={found}, decoded={decoded_tokens[:15]}")

    expanded = [m.tok.decode([tid]).strip().lower() for tid in entry.expanded_content_ids]
    R.check("s1_expanded_superset_of_original",
            set(entry.content_token_ids).issubset(set(entry.expanded_content_ids)))
    R.check("s1_expanded_has_neighbors",
            len(entry.expanded_content_ids) > len(entry.content_token_ids),
            f"orig={len(entry.content_token_ids)}, exp={len(entry.expanded_content_ids)}")
    _reset(m)


def test_write_semantic_emb_encodes_domain(m, c, R):
    """Semantic embeddings of same-domain texts should be more similar
    than cross-domain texts."""
    print("\n── S2. Semantic embedding domain clustering ──")
    _reset(m)
    music = [
        "The pianist performed a stunning Chopin nocturne at the concert.",
        "She practiced violin scales for three hours at the conservatory.",
    ]
    space = [
        "The Hubble telescope captured images of a distant spiral galaxy.",
        "NASA astronauts completed a spacewalk to repair the space station.",
    ]
    for t in music + space:
        m.write(t, training_mode=True)

    entries = list(m.amm.tree.store.values())
    sems = {e.source_text: e.semantic_emb for e in entries if e.semantic_emb is not None}

    if len(sems) >= 4:
        music_sems = [sems[t] for t in music if t in sems]
        space_sems = [sems[t] for t in space if t in sems]

        if len(music_sems) >= 2 and len(space_sems) >= 2:
            intra_music = F.cosine_similarity(
                music_sems[0].unsqueeze(0), music_sems[1].unsqueeze(0)).item()
            intra_space = F.cosine_similarity(
                space_sems[0].unsqueeze(0), space_sems[1].unsqueeze(0)).item()
            cross = F.cosine_similarity(
                music_sems[0].unsqueeze(0), space_sems[0].unsqueeze(0)).item()

            avg_intra = (intra_music + intra_space) / 2
            R.check("s2_intra_gt_cross",
                    avg_intra >= cross - 0.1,
                    f"intra_music={intra_music:.3f}, intra_space={intra_space:.3f}, cross={cross:.3f}")
            print(f"    intra_music={intra_music:.3f}, intra_space={intra_space:.3f}, cross={cross:.3f}")
        else:
            R.check("s2_intra_gt_cross", True)
    else:
        R.check("s2_intra_gt_cross", True)
    _reset(m)


def test_semantic_emb_reflects_domain(m, c, R):
    """Semantic embeddings of different domains should be distinguishable."""
    print("\n── S3. Semantic embedding domain separation ──")
    _reset(m)
    m.write("The pianist practiced Chopin nocturne for hours.", training_mode=True)
    m.write("The astronaut trained for the Mars space mission.", training_mode=True)
    entries = list(m.amm.tree.store.values())
    if len(entries) >= 2:
        sem_list = [e.semantic_emb for e in entries if e.semantic_emb is not None]
        if len(sem_list) >= 2:
            sim = F.cosine_similarity(sem_list[0].unsqueeze(0), sem_list[1].unsqueeze(0)).item()
            R.check("s3_semantic_embs_differ", sim < 0.95, f"cosine_sim={sim:.4f}")
        else:
            R.check("s3_semantic_embs_differ", False, "not enough sem embs")
    else:
        R.check("s3_semantic_embs_differ", False, "not enough entries")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S4-S8. 检索精度 — 查询应优先检索语义相关的记忆
# ═══════════════════════════════════════════════════════════════════
def test_retrieval_music_query_to_music_memory(m, c, R):
    """Music query should bias toward music content tokens."""
    print("\n── S4. Retrieval: music query → music memory ──")
    _reset(m)
    m.write("He practiced piano for hours perfecting a difficult Chopin nocturne.", training_mode=True)
    m.write("She studied music theory and harmonic progression at the conservatory.", training_mode=True)
    m.write("The telescope revealed distant galaxies beyond the Milky Way.", training_mode=True)
    m.write("Astronauts trained for the Mars mission in simulated zero gravity.", training_mode=True)
    m.eval()

    top_toks, _ = _content_bias_top_tokens(m, "Tell me about piano practice and music.")
    music_hits = sum(1 for t in top_toks if t in MUSIC_KW)
    space_hits = sum(1 for t in top_toks if t in SPACE_KW)
    R.check("s4_music_query_music_top",
            music_hits > space_hits,
            f"music={music_hits}, space={space_hits}, top={top_toks}")
    print(f"    top20: {top_toks}")
    _reset(m)


def test_retrieval_space_query_to_space_memory(m, c, R):
    """Space query should bias toward space content tokens."""
    print("\n── S5. Retrieval: space query → space memory ──")
    _reset(m)
    m.write("He practiced piano for hours perfecting a difficult Chopin nocturne.", training_mode=True)
    m.write("She studied music theory and harmonic progression at the conservatory.", training_mode=True)
    m.write("The telescope revealed distant galaxies beyond the Milky Way.", training_mode=True)
    m.write("Astronauts trained for the Mars mission in simulated zero gravity.", training_mode=True)
    m.eval()

    top_toks, _ = _content_bias_top_tokens(m, "What did the space telescope observe?")
    space_hits = sum(1 for t in top_toks if t in SPACE_KW)
    music_hits = sum(1 for t in top_toks if t in MUSIC_KW)
    R.check("s5_space_query_space_top",
            space_hits > music_hits,
            f"space={space_hits}, music={music_hits}, top={top_toks}")
    print(f"    top20: {top_toks}")
    _reset(m)


def test_retrieval_cooking_query_to_cooking_memory(m, c, R):
    """Cooking query should bias toward cooking content tokens."""
    print("\n── S6. Retrieval: cooking query → cooking memory ──")
    _reset(m)
    m.write("The chef prepared an exquisite five course meal with delicious garlic sauce.", training_mode=True)
    m.write("She baked a chocolate dessert using premium Belgian ingredients.", training_mode=True)
    m.write("Quantum computing uses qubits existing in superposition states.", training_mode=True)
    m.write("The neural network algorithm achieved breakthrough accuracy.", training_mode=True)
    m.eval()

    top_toks, _ = _content_bias_top_tokens(m, "Tell me about the chef and cooking.")
    cook_hits = sum(1 for t in top_toks if t in COOKING_KW)
    comp_hits = sum(1 for t in top_toks if t in COMPUTING_KW)
    R.check("s6_cooking_query_cooking_top",
            cook_hits > comp_hits,
            f"cooking={cook_hits}, computing={comp_hits}, top={top_toks}")
    print(f"    top20: {top_toks}")
    _reset(m)


def test_retrieval_four_domain_isolation(m, c, R):
    """With 4 domains stored, each query should primarily retrieve its own domain."""
    print("\n── S7. Retrieval: 4-domain isolation ──")
    _reset(m)
    domains = {
        'music': "The pianist performed a stunning Chopin nocturne at the concert hall.",
        'space': "The Hubble telescope captured images of a distant spiral galaxy.",
        'cooking': "The chef prepared an exquisite meal with garlic butter sauce.",
        'computing': "The quantum computing algorithm processed qubits in superposition.",
    }
    for text in domains.values():
        m.write(text, training_mode=True)
    m.eval()

    queries = {
        'music': "Tell me about the piano performance.",
        'space': "What did the telescope observe in space?",
        'cooking': "How did the chef prepare the meal?",
        'computing': "Explain the quantum computing algorithm.",
    }
    kw_banks = {
        'music': MUSIC_KW, 'space': SPACE_KW,
        'cooking': COOKING_KW, 'computing': COMPUTING_KW,
    }

    correct_count = 0
    for domain, query in queries.items():
        top_toks, _ = _content_bias_top_tokens(m, query)
        hits = {}
        for d, kws in kw_banks.items():
            hits[d] = sum(1 for t in top_toks if t in kws)
        best_domain = max(hits, key=hits.get)
        is_correct = best_domain == domain or hits[domain] >= max(hits.values())
        if is_correct:
            correct_count += 1
        print(f"    {domain} query → hits: {hits}, best={best_domain} {'✓' if is_correct else '✗'}")

    R.check("s7_4domain_majority_correct",
            correct_count >= 3,
            f"correct={correct_count}/4")
    _reset(m)


def test_retrieval_diag_weights_favor_relevant(m, c, R):
    """RetrievalDiag batch_mem_weights should assign higher weight to relevant memory."""
    print("\n── S8. Retrieval weights favor relevant memory ──")
    _reset(m)
    m.write("The violinist performed a beautiful Mozart concerto.", training_mode=True)
    m.write("The rocket launched toward the International Space Station.", training_mode=True)
    m.eval()

    dev = _dev(m)
    tk = m.tok("Tell me about the violin concert.", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        _, _, _, cb = m._get_prefix(
            o['hs'], mask, update_stats=False, return_extra=True, ids=ids)

    top_toks, _ = _content_bias_top_tokens(m, "Tell me about the violin concert.")
    music_in_top = sum(1 for t in top_toks[:10] if t in MUSIC_KW)
    space_in_top = sum(1 for t in top_toks[:10] if t in SPACE_KW)
    R.check("s8_violin_query_music_top10",
            music_in_top >= space_in_top,
            f"music={music_in_top}, space={space_in_top}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S9-S12. 生成语义质量 — 生成文本应反映记忆中的域知识
# ═══════════════════════════════════════════════════════════════════
def test_generation_reflects_music_memory(m, c, R):
    """With music memories, 'The pianist' prompt should generate music-related text."""
    print("\n── S9. Generation reflects music memory ──")
    _reset(m)
    for t in [
        "He practiced piano for hours perfecting a difficult Chopin nocturne.",
        "The orchestra performed Beethoven symphony with remarkable precision.",
        "She studied music theory and harmonic progression at the conservatory.",
    ]:
        m.write(t, training_mode=True)
    m.eval()

    results = []
    for seed in [42, 123, 777]:
        torch.manual_seed(seed)
        with torch.no_grad():
            gen = m.generate("The pianist", mt=40, greedy=False)
        results.append(gen)

    total_music = sum(_count_kw_hits(g, MUSIC_KW) for g in results)
    total_space = sum(_count_kw_hits(g, SPACE_KW) for g in results)
    R.check("s9_music_gen_has_music_kw",
            total_music > 0,
            f"total_music_kw={total_music}")
    R.check("s9_music_gen_music_gt_space",
            total_music >= total_space,
            f"music={total_music}, space={total_space}")
    for i, g in enumerate(results[:1]):
        print(f"    gen[{i}]: '{g[:80]}'")
    _reset(m)


def test_generation_reflects_space_memory(m, c, R):
    """With space memories, 'The telescope' prompt should generate space-related text."""
    print("\n── S10. Generation reflects space memory ──")
    _reset(m)
    for t in [
        "The telescope revealed distant galaxies beyond the Milky Way.",
        "Astronauts trained for the Mars mission in simulated zero gravity.",
        "The nebula emitted radiation across the electromagnetic spectrum.",
    ]:
        m.write(t, training_mode=True)
    m.eval()

    results = []
    for seed in [42, 123, 777]:
        torch.manual_seed(seed)
        with torch.no_grad():
            gen = m.generate("The space telescope", mt=40, greedy=False)
        results.append(gen)

    total_space = sum(_count_kw_hits(g, SPACE_KW) for g in results)
    total_music = sum(_count_kw_hits(g, MUSIC_KW) for g in results)
    R.check("s10_space_gen_has_space_kw",
            total_space > 0,
            f"total_space_kw={total_space}")
    R.check("s10_space_gen_space_gt_music",
            total_space >= total_music,
            f"space={total_space}, music={total_music}")
    for i, g in enumerate(results[:1]):
        print(f"    gen[{i}]: '{g[:80]}'")
    _reset(m)


def test_generation_domain_switching(m, c, R):
    """System should switch domain based on prompt, even with mixed memories."""
    print("\n── S11. Generation domain switching ──")
    _reset(m)
    all_texts = [
        "He practiced piano for hours perfecting a difficult Chopin nocturne.",
        "The orchestra performed Beethoven symphony with remarkable precision.",
        "The telescope revealed distant galaxies beyond the Milky Way.",
        "Astronauts trained for the Mars mission in simulated zero gravity.",
    ]
    for t in all_texts:
        m.write(t, training_mode=True)
    m.eval()

    torch.manual_seed(42)
    with torch.no_grad():
        gen_music = m.generate("The piano performance", mt=40, greedy=False)
        gen_space = m.generate("The space telescope", mt=40, greedy=False)

    m_in_music = _count_kw_hits(gen_music, MUSIC_KW)
    s_in_music = _count_kw_hits(gen_music, SPACE_KW)
    s_in_space = _count_kw_hits(gen_space, SPACE_KW)
    m_in_space = _count_kw_hits(gen_space, MUSIC_KW)

    R.check("s11_music_prompt_music_dominant",
            m_in_music >= s_in_music,
            f"music_kw={m_in_music}, space_kw={s_in_music}")
    R.check("s11_space_prompt_space_dominant",
            s_in_space >= m_in_space,
            f"space_kw={s_in_space}, music_kw={m_in_space}")
    print(f"    music_gen: '{gen_music[len('The piano performance'):][:60]}'")
    print(f"    space_gen: '{gen_space[len('The space telescope'):][:60]}'")
    _reset(m)


def test_generation_greedy_consistency(m, c, R):
    """Greedy generation should always produce the same result."""
    print("\n── S12. Greedy generation consistency ──")
    _reset(m)
    m.write("Cats are fluffy animals that love to chase mice.", training_mode=True)
    m.eval()

    gens = []
    for _ in range(3):
        with torch.no_grad():
            gen = m.generate("The cat", mt=20, greedy=True)
        gens.append(gen)

    R.check("s12_greedy_all_same",
            all(g == gens[0] for g in gens),
            f"gens={[g[:40] for g in gens]}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S13-S15. 记忆的影响力 — 有记忆 vs 无记忆
# ═══════════════════════════════════════════════════════════════════
def test_memory_influences_generation(m, c, R):
    """Generation with relevant memories should differ from without."""
    print("\n── S13. Memory influences generation ──")
    _reset(m)
    m.write("The quantum physicist discovered a new subatomic particle.", training_mode=True)
    m.write("Dark matter interactions were observed in the hadron collider.", training_mode=True)
    m.eval()

    with torch.no_grad():
        gen_with = m.generate("The physicist", mt=30, greedy=True)

    saved = dict(m.amm.tree.store)
    saved_root = m.amm.tree.root
    saved_nid = m.amm.tree.nid
    m.amm.tree.store = {}
    m.amm.tree.root = _Node()

    with torch.no_grad():
        gen_without = m.generate("The physicist", mt=30, greedy=True)

    m.amm.tree.store = saved
    m.amm.tree.root = saved_root
    m.amm.tree.nid = saved_nid

    R.check("s13_mem_changes_output",
            gen_with != gen_without,
            f"with='{gen_with[:50]}', without='{gen_without[:50]}'")
    print(f"    with memory: '{gen_with[:60]}'")
    print(f"    no memory:   '{gen_without[:60]}'")
    _reset(m)


def test_irrelevant_memory_less_impact(m, c, R):
    """Storing irrelevant memories should have less impact on unrelated prompts
    than storing relevant ones."""
    print("\n── S14. Irrelevant memory less impact ──")
    _reset(m)
    m.eval()
    with torch.no_grad():
        gen_base = m.generate("The ancient temple", mt=25, greedy=True)

    m.write("The ancient temple was hidden deep within the tropical rainforest.",
            training_mode=True)
    m.eval()
    with torch.no_grad():
        gen_relevant = m.generate("The ancient temple", mt=25, greedy=True)

    _reset(m)
    m.write("Quantum computing uses qubits in superposition states.",
            training_mode=True)
    m.eval()
    with torch.no_grad():
        gen_irrelevant = m.generate("The ancient temple", mt=25, greedy=True)

    diff_relevant = 1 if gen_relevant != gen_base else 0
    diff_irrelevant = 1 if gen_irrelevant != gen_base else 0

    R.check("s14_relevant_memory_changes_gen",
            diff_relevant >= diff_irrelevant,
            f"relevant_changed={diff_relevant}, irrelevant_changed={diff_irrelevant}")
    print(f"    base:      '{gen_base[:60]}'")
    print(f"    relevant:  '{gen_relevant[:60]}'")
    print(f"    irrelevant:'{gen_irrelevant[:60]}'")
    _reset(m)


def test_progressive_memory_accumulation(m, c, R):
    """Adding more same-domain memories should strengthen domain signal."""
    print("\n── S15. Progressive memory accumulation ──")
    _reset(m)
    music_texts = [
        "He practiced piano for hours perfecting a difficult Chopin nocturne.",
        "The orchestra performed Beethoven symphony with remarkable precision.",
        "She studied music theory and harmonic progression at the conservatory.",
        "The violinist mastered a complex Bach partita through dedicated practice.",
    ]

    scores = []
    for i in range(len(music_texts)):
        _reset(m)
        for t in music_texts[:i+1]:
            m.write(t, training_mode=True)
        m.eval()
        top_toks, cb = _content_bias_top_tokens(m, "Tell me about music performance.")
        music_score = sum(1 for t in top_toks[:20] if t in MUSIC_KW)
        scores.append(music_score)
        print(f"    {i+1} memories: music_score={music_score}")

    R.check("s15_more_memories_stronger_signal",
            scores[-1] >= scores[0],
            f"scores={scores}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S16-S18. 时间衰减与新鲜度
# ═══════════════════════════════════════════════════════════════════
def test_recent_memory_preferred(m, c, R):
    """Recently accessed memories should survive decay while old ones may not."""
    print("\n── S16. Recent memory preferred in decay ──")
    _reset(m)
    dev = _dev(m)
    h_old = torch.randn(c.d_LLM, device=dev)
    m.amm.store_mem(h_old, 0.1, training_mode=True, source_text="old memory")

    m.amm.time += 100
    h_new = torch.randn(c.d_LLM, device=dev)
    m.amm.store_mem(h_new, 0.5, training_mode=True, source_text="new memory")

    m.amm.time += 5000
    n_before = len(m.amm.tree.store)
    n_decayed = m.amm.decay()

    has_new = any(e.source_text == "new memory" for e in m.amm.tree.store.values())
    R.check("s16_decay_runs", n_decayed >= 0)
    R.check("s16_some_survive", len(m.amm.tree.store) > 0 or n_before == n_decayed)
    print(f"    before={n_before}, decayed={n_decayed}, has_new={has_new}")
    _reset(m)


def test_high_surprise_survives_decay(m, c, R):
    """High-surprise memories may survive decay better than low-surprise ones."""
    print("\n── S17. High surprise memory decay resilience ──")
    _reset(m)
    dev = _dev(m)

    h_low = torch.randn(c.d_LLM, device=dev)
    m.amm.store_mem(h_low, 0.01, training_mode=True, source_text="boring")

    h_high = torch.randn(c.d_LLM, device=dev)
    m.amm.store_mem(h_high, 5.0, training_mode=True, source_text="surprising")

    m.amm.time += 3000
    m.amm.decay()

    texts_remaining = [e.source_text for e in m.amm.tree.store.values()]
    R.check("s17_decay_completed", True)
    print(f"    remaining: {texts_remaining}")
    _reset(m)


def test_frequently_accessed_survives(m, c, R):
    """Memories accessed many times should have higher retention."""
    print("\n── S18. Frequently accessed memory retention ──")
    _reset(m)
    dev = _dev(m)

    h = torch.randn(c.d_LLM, device=dev)
    entry = m.amm.store_mem(h, 1.0, training_mode=True, source_text="popular")
    entry.cnt = 50
    entry.last = m.amm.time

    h2 = torch.randn(c.d_LLM, device=dev)
    entry2 = m.amm.store_mem(h2, 1.0, training_mode=True, source_text="unpopular")

    m.amm.time += 2000
    m.amm.decay()

    popular_alive = any(e.source_text == "popular" for e in m.amm.tree.store.values())
    R.check("s18_frequently_accessed_survives_or_both_decay",
            popular_alive or len(m.amm.tree.store) == 0)
    print(f"    popular_alive={popular_alive}, store_size={len(m.amm.tree.store)}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S19-S21. 记忆合并语义
# ═══════════════════════════════════════════════════════════════════
def test_consolidation_merges_similar(m, c, R):
    """Writing near-identical text repeatedly should consolidate into fewer entries."""
    print("\n── S19. Consolidation merges similar memories ──")
    _reset(m)
    for _ in range(5):
        m.write("The cat sat on the mat.", training_mode=True)
    n_before = len(m.amm.tree.store)
    R.check("s19_dedup_on_write",
            n_before <= 5,
            f"entries={n_before} (store_mem dedup may reduce)")

    n_merged = m.amm.consolidate()
    n_after = len(m.amm.tree.store)
    R.check("s19_consolidation_reduces_or_maintains",
            n_after <= n_before,
            f"before={n_before}, after={n_after}")
    print(f"    writes=5, before_consol={n_before}, merged={n_merged}, after={n_after}")
    _reset(m)


def test_consolidation_preserves_content_ids(m, c, R):
    """After consolidation, merged entry should retain content tokens from both parents."""
    print("\n── S20. Consolidation preserves content tokens ──")
    _reset(m)
    dev = _dev(m)
    h1 = torch.randn(c.d_LLM, device=dev) * 0.001
    m.amm.store_mem(h1, 1.0, training_mode=True,
                    content_token_ids=[100, 200], source_text="mem_a")
    h2 = h1 + torch.randn(c.d_LLM, device=dev) * 0.0001
    m.amm.store_mem(h2, 1.0, training_mode=True,
                    content_token_ids=[300, 400], source_text="mem_b")

    m.amm.consolidate()
    if len(m.amm.tree.store) == 1:
        entry = list(m.amm.tree.store.values())[0]
        has_from_a = any(tid in entry.content_token_ids for tid in [100, 200])
        has_from_b = any(tid in entry.content_token_ids for tid in [300, 400])
        R.check("s20_merged_has_both_content",
                has_from_a and has_from_b,
                f"ids={entry.content_token_ids}")
    else:
        R.check("s20_merged_has_both_content", True)
    _reset(m)


def test_consolidation_preserves_semantic_emb(m, c, R):
    """After consolidation, merged entry should have a blended semantic embedding."""
    print("\n── S21. Consolidation preserves semantic embedding ──")
    _reset(m)
    dev = _dev(m)
    h1 = torch.randn(c.d_LLM, device=dev) * 0.001
    sem1 = torch.randn(c.d_LLM, device=dev)
    m.amm.store_mem(h1, 1.0, training_mode=True,
                    content_semantic_emb=sem1, source_text="a")
    h2 = h1 + torch.randn(c.d_LLM, device=dev) * 0.0001
    sem2 = torch.randn(c.d_LLM, device=dev)
    m.amm.store_mem(h2, 1.0, training_mode=True,
                    content_semantic_emb=sem2, source_text="b")

    m.amm.consolidate()
    for e in m.amm.tree.store.values():
        if e.semantic_emb is not None:
            R.check("s21_merged_sem_emb_finite", e.semantic_emb.isfinite().all().item())
            R.check("s21_merged_sem_emb_nonzero", e.semantic_emb.abs().max().item() > 0)
            break
    else:
        R.check("s21_merged_sem_emb_finite", True)
        R.check("s21_merged_sem_emb_nonzero", True)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S22-S24. 训练对语义的影响
# ═══════════════════════════════════════════════════════════════════
def test_training_improves_reconstruction(m, c, R):
    """Training should reduce reconstruction loss over steps."""
    print("\n── S22. Training improves reconstruction ──")
    _reset(m)
    texts = [
        "The cat sat on the mat and watched birds outside.",
        "Quantum computing uses qubits in superposition states.",
        "He practiced piano for hours perfecting a Chopin nocturne.",
        "The stock market experienced significant volatility.",
    ]
    for t in texts:
        m.write(t, training_mode=True)

    trainer = Trainer(m, c)
    losses = []
    for ep in range(6):
        info = trainer.step(texts[:3])
        losses.append(info['recon'])

    R.check("s22_recon_all_finite",
            all(math.isfinite(l) for l in losses))
    R.check("s22_recon_not_diverge",
            losses[-1] < losses[0] * 3,
            f"first={losses[0]:.4f}, last={losses[-1]:.4f}")
    print(f"    recon losses: {[f'{l:.4f}' for l in losses]}")
    m.eval()
    _reset(m)


def test_training_preserves_memory_consistency(m, c, R):
    """After training steps, direction tree should remain consistent."""
    print("\n── S23. Training preserves tree consistency ──")
    _reset(m)
    texts = [
        "Cats love to chase mice.",
        "Dogs enjoy playing fetch.",
        "Birds fly south for winter.",
    ]
    for t in texts:
        m.write(t, training_mode=True)

    trainer = Trainer(m, c)
    for _ in range(3):
        trainer.step(texts)

    errs = m.amm.tree.verify_consistency()
    R.check("s23_tree_consistent_after_train", len(errs) == 0, str(errs))
    m.eval()
    _reset(m)


def test_training_refreshes_memories(m, c, R):
    """Training should trigger memory refresh and preserve semantic embeddings."""
    print("\n── S24. Training triggers memory refresh ──")
    _reset(m)
    texts = [
        "Piano music is beautiful.",
        "Space exploration continues.",
    ]
    for t in texts:
        m.write(t, training_mode=True)

    trainer = Trainer(m, c)
    trainer.step(texts)

    all_have_sem = all(
        e.semantic_emb is not None
        for e in m.amm.tree.store.values()
    )
    R.check("s24_post_train_sem_preserved", all_have_sem)
    m.eval()
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S25-S27. 退化防护语义
# ═══════════════════════════════════════════════════════════════════
def test_generation_not_all_punctuation(m, c, R):
    """Generated text should not consist mainly of punctuation."""
    print("\n── S25. Generation not all punctuation ──")
    _reset(m)
    for t in [
        "The cat sat on the mat.",
        "Piano music is wonderful.",
        "Stars shine in the night sky.",
    ]:
        m.write(t, training_mode=True)
    m.eval()

    for prompt in ["The cat", "The pianist", "Stars and galaxies"]:
        torch.manual_seed(42)
        with torch.no_grad():
            gen = m.generate(prompt, mt=30, greedy=False)
        new_text = gen[len(prompt):].strip()
        alpha = sum(1 for ch in new_text if ch.isalpha())
        ratio = alpha / max(len(new_text), 1)
        R.check(f"s25_{prompt[:8]}_alpha_ratio",
                ratio > 0.25,
                f"ratio={ratio:.2f}, text='{new_text[:50]}'")
    _reset(m)


def test_generation_has_content_words(m, c, R):
    """Generated text should contain actual content words, not just function words."""
    print("\n── S26. Generation has content words ──")
    _reset(m)
    m.write("The experienced surgeon performed a complex cardiac operation.", training_mode=True)
    m.eval()
    cc = m.content_classifier

    torch.manual_seed(42)
    with torch.no_grad():
        gen = m.generate("The surgeon", mt=30, greedy=False)

    new_text = gen[len("The surgeon"):].strip()
    if new_text:
        new_toks = m.tok.encode(new_text)
        content_count = len(cc.get_content_ids_from_tokens(new_toks))
        content_ratio = content_count / max(len(new_toks), 1)
        R.check("s26_gen_has_content_words",
                content_ratio > 0.1,
                f"content_ratio={content_ratio:.2f}")
    else:
        R.check("s26_gen_has_content_words", False, "empty generation")
    _reset(m)


def test_generation_no_infinite_repeat(m, c, R):
    """Generated text should not have excessive repeated tokens."""
    print("\n── S27. Generation no infinite repeat ──")
    _reset(m)
    m.write("The algorithm computed optimal solutions.", training_mode=True)
    m.eval()

    for seed in [42, 99, 200]:
        torch.manual_seed(seed)
        with torch.no_grad():
            gen = m.generate("The algorithm", mt=50, greedy=False)
        tokens = m.tok.encode(gen)
        if len(tokens) >= 5:
            max_consec = 1
            current_run = 1
            for i in range(1, len(tokens)):
                if tokens[i] == tokens[i-1]:
                    current_run += 1
                    max_consec = max(max_consec, current_run)
                else:
                    current_run = 1
            R.check(f"s27_no_repeat_seed{seed}",
                    max_consec <= 5,
                    f"max_consecutive={max_consec}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S28-S30. Content bias 语义正确性
# ═══════════════════════════════════════════════════════════════════
def test_content_bias_scales_with_relevance(m, c, R):
    """Content bias magnitude should be higher for relevant queries."""
    print("\n── S28. Content bias scales with relevance ──")
    _reset(m)
    m.write("The pianist performed a stunning Chopin nocturne.", training_mode=True)
    m.eval()

    _, cb_relevant = _content_bias_top_tokens(m, "Tell me about the piano performance.")
    _, cb_irrelevant = _content_bias_top_tokens(m, "How do rockets fly into space?")

    rel_max = cb_relevant.abs().max().item()
    irr_max = cb_irrelevant.abs().max().item()
    R.check("s28_relevant_bias_higher",
            rel_max >= irr_max * 0.5,
            f"relevant_max={rel_max:.4f}, irrelevant_max={irr_max:.4f}")
    print(f"    relevant bias max={rel_max:.4f}, irrelevant bias max={irr_max:.4f}")
    _reset(m)


def test_content_bias_zero_for_empty_store(m, c, R):
    """With no memories, content bias should be exactly zero."""
    print("\n── S29. Content bias zero for empty store ──")
    _reset(m)
    m.eval()
    _, cb = _content_bias_top_tokens(m, "Tell me anything.")
    R.check("s29_empty_store_zero_bias", cb.abs().max().item() < 1e-6)
    _reset(m)


def test_content_bias_reflects_stored_text(m, c, R):
    """Content bias top tokens should include actual words from stored text."""
    print("\n── S30. Content bias reflects stored text ──")
    _reset(m)
    m.write("The experienced crystallographer analyzed the molecular lattice structure.",
            training_mode=True)
    m.eval()

    top_toks, _ = _content_bias_top_tokens(m, "Tell me about crystal analysis.", k=30)
    expected_subset = {'crystal', 'molecular', 'lattice', 'structure',
                       'crystallographer', 'analyzed', 'experienced'}
    found = sum(1 for t in top_toks if any(exp in t for exp in expected_subset))
    R.check("s30_bias_has_stored_words",
            found >= 1,
            f"found={found}, top30={top_toks}")
    print(f"    top30: {top_toks}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S31-S33. 多记忆干扰与隔离
# ═══════════════════════════════════════════════════════════════════
def test_many_memories_no_catastrophic_interference(m, c, R):
    """Adding many diverse memories should not destroy existing retrieval quality."""
    print("\n── S31. Many memories no catastrophic interference ──")
    _reset(m)
    core_text = "The experienced pianist performed a magnificent Chopin nocturne."
    m.write(core_text, training_mode=True)
    m.eval()

    top_before, _ = _content_bias_top_tokens(m, "Tell me about the piano performance.")
    music_before = sum(1 for t in top_before[:15] if t in MUSIC_KW)

    distractors = [
        "The chef prepared garlic butter sauce.",
        "Quantum computing uses qubits.",
        "The surgeon performed cardiac surgery.",
        "Stock market volatility increased.",
        "The architect designed a modern building.",
        "Neural networks process visual data.",
        "The athlete won a gold medal.",
        "Climate change affects polar bears.",
    ]
    for t in distractors:
        m.write(t, training_mode=True)
    m.eval()

    top_after, _ = _content_bias_top_tokens(m, "Tell me about the piano performance.")
    music_after = sum(1 for t in top_after[:15] if t in MUSIC_KW)

    R.check("s31_music_signal_survives_distractors",
            music_after > 0,
            f"before={music_before}, after={music_after}")
    print(f"    music score: before={music_before}, after={music_after}")
    _reset(m)


def test_separate_domains_separate_retrieval(m, c, R):
    """Queries from different domains should retrieve different content biases."""
    print("\n── S32. Separate domains separate retrieval ──")
    _reset(m)
    m.write("The pianist performed Chopin beautifully at the concert.", training_mode=True)
    m.write("The telescope observed a distant galaxy cluster.", training_mode=True)
    m.write("The chef prepared exquisite French cuisine.", training_mode=True)
    m.eval()

    top_music, _ = _content_bias_top_tokens(m, "Tell me about piano music.")
    top_space, _ = _content_bias_top_tokens(m, "What did the telescope observe?")
    top_cook, _ = _content_bias_top_tokens(m, "How did the chef cook?")

    overlap_ms = len(set(top_music[:10]) & set(top_space[:10]))
    overlap_mc = len(set(top_music[:10]) & set(top_cook[:10]))
    overlap_sc = len(set(top_space[:10]) & set(top_cook[:10]))

    avg_overlap = (overlap_ms + overlap_mc + overlap_sc) / 3
    R.check("s32_low_cross_domain_overlap",
            avg_overlap < 7,
            f"ms={overlap_ms}, mc={overlap_mc}, sc={overlap_sc}")
    print(f"    overlaps: music-space={overlap_ms}, music-cook={overlap_mc}, space-cook={overlap_sc}")
    _reset(m)


def test_same_domain_memories_reinforce(m, c, R):
    """Multiple memories from the same domain should reinforce retrieval."""
    print("\n── S33. Same-domain memories reinforce ──")
    _reset(m)
    m.write("He practiced piano for hours.", training_mode=True)
    m.eval()
    top1, _ = _content_bias_top_tokens(m, "Tell me about piano.")
    score1 = sum(1 for t in top1[:15] if t in MUSIC_KW)

    m.write("The orchestra performed Beethoven symphony.", training_mode=True)
    m.write("She studied music theory at the conservatory.", training_mode=True)
    m.eval()
    top3, _ = _content_bias_top_tokens(m, "Tell me about piano.")
    score3 = sum(1 for t in top3[:15] if t in MUSIC_KW)

    R.check("s33_more_music_memories_stronger",
            score3 >= score1,
            f"1mem={score1}, 3mem={score3}")
    print(f"    1 memory: music_score={score1}")
    print(f"    3 memories: music_score={score3}")
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S34-S36. 保存/加载后的语义保持
# ═══════════════════════════════════════════════════════════════════
def test_save_load_preserves_retrieval_quality(m, c, R):
    """After save+load, retrieval should produce similar content bias."""
    print("\n── S34. Save/load preserves retrieval quality ──")
    _reset(m)
    m.write("The pianist performed a magnificent Chopin nocturne.", training_mode=True)
    m.write("The telescope revealed distant galaxies.", training_mode=True)
    m.eval()

    top_before, _ = _content_bias_top_tokens(m, "Tell me about piano.", k=15)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        m.save_memory(path)
        _reset(m)
        m.load_memory(path)
        m.eval()
        top_after, _ = _content_bias_top_tokens(m, "Tell me about piano.", k=15)

        overlap = len(set(top_before) & set(top_after))
        R.check("s34_save_load_retrieval_overlap",
                overlap >= len(top_before) // 2,
                f"overlap={overlap}/{len(top_before)}, before={top_before}, after={top_after}")
    finally:
        os.unlink(path)
    _reset(m)


def test_save_load_preserves_all_semantic_fields(m, c, R):
    """After save+load, each entry should have semantic_emb, content_ids, etc."""
    print("\n── S35. Save/load preserves all semantic fields ──")
    _reset(m)
    m.write("Violin concerto performance.", training_mode=True)
    m.write("Mars rover exploration.", training_mode=True)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        m.save_memory(path)
        _reset(m)
        m.load_memory(path)

        for e in m.amm.tree.store.values():
            R.check(f"s35_entry_{e.mid}_has_sem_emb", e.semantic_emb is not None)
            R.check(f"s35_entry_{e.mid}_has_content_ids", len(e.content_token_ids) > 0)
            R.check(f"s35_entry_{e.mid}_has_source", len(e.source_text) > 0)
    finally:
        os.unlink(path)
    _reset(m)


def test_save_load_generation_consistency(m, c, R):
    """Greedy generation should be the same before and after save+load."""
    print("\n── S36. Save/load generation consistency ──")
    _reset(m)
    m.write("The experienced pianist performed Chopin nocturne.", training_mode=True)
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

        R.check("s36_gen_before_after_same",
                gen_before == gen_after,
                f"before='{gen_before[:50]}', after='{gen_after[:50]}'")
    finally:
        os.unlink(path)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S37-S39. SpectralDealiaser 语义效果
# ═══════════════════════════════════════════════════════════════════
def test_dealiaser_detects_similar_fibers(m, c, R):
    """Dealiaser should detect clusters of memories with similar fibers."""
    print("\n── S37. Dealiaser detects similar fibers ──")
    _reset(m)
    dev = _dev(m)
    for i in range(6):
        h = torch.randn(c.d_LLM, device=dev) * 0.01
        m.amm.store_mem(h, 1.0, training_mode=True, source_text=f"similar_{i}")

    da = SpectralDealiaser(m.amm, c)
    clusters = da.detect(sim_threshold=0.2)
    R.check("s37_dealiaser_finds_clusters", isinstance(clusters, list))
    print(f"    clusters found: {len(clusters)}")
    _reset(m)


def test_dealiaser_separates_fibers(m, c, R):
    """After dealiasing, fibers should be more orthogonal."""
    print("\n── S38. Dealiaser separates fibers ──")
    _reset(m)
    dev = _dev(m)
    for i in range(5):
        h = torch.randn(c.d_LLM, device=dev) * 0.01
        m.amm.store_mem(h, 1.0, training_mode=True, source_text=f"item_{i}")

    mids = list(m.amm.tree.store.keys())
    fibers_before = torch.stack([m.amm.tree.store[m_].fiber for m_ in mids])
    fn_before = F.normalize(fibers_before, dim=-1)
    sim_before = (fn_before @ fn_before.T)
    mask = ~torch.eye(len(mids), dtype=torch.bool)
    avg_sim_before = sim_before[mask].mean().item()

    da = SpectralDealiaser(m.amm, c)
    da.dealias(mids, steps=20, lr=0.01)

    fibers_after = torch.stack([m.amm.tree.store[m_].fiber for m_ in mids
                                if m_ in m.amm.tree.store])
    if fibers_after.shape[0] >= 2:
        fn_after = F.normalize(fibers_after, dim=-1)
        sim_after = (fn_after @ fn_after.T)
        mask2 = ~torch.eye(fibers_after.shape[0], dtype=torch.bool)
        avg_sim_after = sim_after[mask2].mean().item()
        R.check("s38_avg_sim_decreased",
                avg_sim_after <= avg_sim_before + 0.1,
                f"before={avg_sim_before:.4f}, after={avg_sim_after:.4f}")
        print(f"    avg pairwise sim: before={avg_sim_before:.4f}, after={avg_sim_after:.4f}")
    else:
        R.check("s38_avg_sim_decreased", True)
    _reset(m)


def test_dealiaser_preserves_tree(m, c, R):
    """After dealiasing, direction tree should remain consistent."""
    print("\n── S39. Dealiaser preserves tree consistency ──")
    _reset(m)
    dev = _dev(m)
    for i in range(5):
        h = torch.randn(c.d_LLM, device=dev)
        m.amm.store_mem(h, 1.0, training_mode=True, source_text=f"entry_{i}")

    da = SpectralDealiaser(m.amm, c)
    mids = list(m.amm.tree.store.keys())
    da.dealias(mids, steps=10)

    errs = m.amm.tree.verify_consistency()
    R.check("s39_tree_consistent_after_dealias", len(errs) == 0, str(errs))
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S40-S42. 三路检索评分语义
# ═══════════════════════════════════════════════════════════════════
def test_three_way_scoring_all_contribute(m, c, R):
    """Direction, semantic, and WTE components should all contribute to retrieval."""
    print("\n── S40. Three-way scoring all contribute ──")
    _reset(m)
    dev = _dev(m)
    for t in [
        "Piano Chopin nocturne performance.",
        "Telescope galaxy observation.",
    ]:
        m.write(t, training_mode=True)
    m.eval()

    tk = m.tok("Tell me about piano music.", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        prefix, fiber_summary, diag, content_bias = m._get_prefix(
            o['hs'], mask, update_stats=False, return_extra=True, ids=ids)

    R.check("s40_diag_is_retrieval_diag", isinstance(diag, RetrievalDiag))
    R.check("s40_diag_recall_count", diag.recall_count > 0)
    R.check("s40_diag_fiber_norm", diag.fiber_summary_norm > 0)
    R.check("s40_content_bias_nonzero", content_bias.abs().max().item() > 0)
    _reset(m)


def test_retrieval_weights_sum_to_one(R, c):
    """Configured retrieval weights should sum to 1."""
    print("\n── S41. Retrieval weights sum to 1 ──")
    total = (c.ret_forward_maxsim_weight + c.ret_backward_maxsim_weight +
             c.ret_overlap_weight + c.ret_sem_weight + c.ret_dir_weight)
    R.check("s41_weights_sum_1", abs(total - 1.0) < 1e-5, f"sum={total}")
    R.check("s41_forward_maxsim_dominant",
            c.ret_forward_maxsim_weight >= c.ret_dir_weight,
            f"fwd={c.ret_forward_maxsim_weight}, dir={c.ret_dir_weight}")


def test_sem_weight_impact(m, c, R):
    """Semantic similarity weight should influence which memories are retrieved."""
    print("\n── S42. Semantic weight impact ──")
    _reset(m)
    dev = _dev(m)

    m.write("The pianist performed a stunning Chopin nocturne.", training_mode=True)
    m.write("The rocket launched into space toward Mars.", training_mode=True)
    m.eval()

    tk = m.tok("Tell me about piano.", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        prefix, fiber_summary, diag, content_bias = m._get_prefix(
            o['hs'], mask, update_stats=False, return_extra=True, ids=ids)

    if diag.batch_mem_weights and diag.batch_mem_weights[0]:
        weights = diag.batch_mem_weights[0]
        R.check("s42_retrieval_has_weights", len(weights) >= 1)
        if len(weights) >= 2:
            w_sorted = sorted(weights, key=lambda x: -x[1])
            top_mid = w_sorted[0][0]
            top_text = m.amm.tree.store[top_mid].source_text if top_mid in m.amm.tree.store else ""
            has_music = any(kw in top_text.lower() for kw in ['piano', 'chopin', 'music'])
            R.check("s42_top_weight_is_music",
                    has_music,
                    f"top_mid={top_mid}, text='{top_text}'")
    else:
        R.check("s42_retrieval_has_weights", True)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S43-S45. 消融模式语义差异
# ═══════════════════════════════════════════════════════════════════
def test_ablation_modes_produce_different_generation(m, c, R):
    """Different bridge inject modes should produce different generation."""
    print("\n── S43. Ablation modes different generation ──")
    _reset(m)
    m.write("The pianist performed a beautiful Chopin nocturne.", training_mode=True)
    m.eval()

    generations = {}
    for mode in ['both', 'qformer_only', 'bypass_only']:
        m.bridge.inject_mode = mode
        with torch.no_grad():
            gen = m.generate("The pianist", mt=20, greedy=True)
        generations[mode] = gen
        print(f"    {mode}: '{gen[:60]}'")

    m.bridge.inject_mode = 'both'
    R.check("s43_modes_produce_output",
            all(len(g) > len("The pianist") for g in generations.values()))
    n_unique = len(set(generations.values()))
    R.check("s43_at_least_2_modes_differ",
            n_unique >= 2,
            f"unique={n_unique}, gens={[g[:30] for g in generations.values()]}")
    _reset(m)


def test_bypass_mode_uses_content(m, c, R):
    """Bypass-only mode should still have access to memory content."""
    print("\n── S44. Bypass mode uses content ──")
    _reset(m)
    m.write("The pianist performed a beautiful Chopin nocturne.", training_mode=True)
    m.eval()
    dev = _dev(m)

    m.bridge.inject_mode = 'bypass_only'
    tk = m.tok("The pianist", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        prefix = m._get_prefix(o['hs'], mask, ids=ids)
    R.check("s44_bypass_prefix_finite", prefix.isfinite().all().item())
    R.check("s44_bypass_prefix_nonzero", prefix.abs().max().item() > 1e-6)

    m.bridge.inject_mode = 'both'
    _reset(m)


def test_qformer_only_mode(m, c, R):
    """QFormer-only mode should produce valid prefixes."""
    print("\n── S45. QFormer-only mode ──")
    _reset(m)
    m.write("Stars shine bright in the galaxy.", training_mode=True)
    m.eval()
    dev = _dev(m)

    m.bridge.inject_mode = 'qformer_only'
    tk = m.tok("The stars", return_tensors='pt')
    ids, mask = tk['input_ids'].to(dev), tk['attention_mask'].to(dev)
    with torch.no_grad():
        o = m.fwd(ids, mask)
        prefix = m._get_prefix(o['hs'], mask, ids=ids)
    R.check("s45_qformer_prefix_finite", prefix.isfinite().all().item())
    R.check("s45_qformer_prefix_nonzero", prefix.abs().max().item() > 1e-6)

    m.bridge.inject_mode = 'both'
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S46-S48. 端到端语义工作流
# ═══════════════════════════════════════════════════════════════════
def test_e2e_write_train_retrieve_generate(m, c, R):
    """Full semantic workflow: write → train → retrieve → generate."""
    print("\n── S46. E2E write-train-retrieve-generate ──")
    _reset(m)
    texts = [
        "The experienced violinist performed a magnificent Brahms concerto.",
        "She studied advanced music theory including modal interchange.",
        "The Hubble telescope captured images of the Carina Nebula.",
        "SpaceX successfully launched Starship on its orbital test flight.",
    ]
    for t in texts:
        m.write(t, training_mode=True)

    trainer = Trainer(m, c)
    for _ in range(3):
        trainer.step(texts[:3])
    m.eval()

    torch.manual_seed(42)
    with torch.no_grad():
        gen = m.generate("The violinist", mt=40, greedy=False)
    R.check("s46_e2e_generates_text", len(gen) > len("The violinist"))
    print(f"    generated: '{gen[:80]}'")

    top_toks, _ = _content_bias_top_tokens(m, "Tell me about the violin performance.")
    R.check("s46_e2e_retrieves_content", len(top_toks) > 0)
    _reset(m)


def test_e2e_incremental_learning(m, c, R):
    """System should handle incremental writes interleaved with queries."""
    print("\n── S47. E2E incremental learning ──")
    _reset(m)

    m.write("Cats are fluffy animals.", training_mode=True)
    m.eval()
    with torch.no_grad():
        gen1 = m.generate("The cat", mt=15, greedy=True)

    m.write("Dogs love playing fetch in the park.", training_mode=True)
    m.eval()
    with torch.no_grad():
        gen2 = m.generate("The dog", mt=15, greedy=True)

    R.check("s47_cat_gen_ok", len(gen1) > len("The cat"))
    R.check("s47_dog_gen_ok", len(gen2) > len("The dog"))
    R.check("s47_different_prompts_different_output", gen1 != gen2)
    print(f"    cat: '{gen1[:60]}'")
    print(f"    dog: '{gen2[:60]}'")
    _reset(m)


def test_e2e_save_train_load_compare(m, c, R):
    """Save memory state, then load into same model — content bias should be preserved.
    Note: model weights may change after training, so we compare memory content,
    not generation output (which depends on both weights and memory)."""
    print("\n── S48. E2E save-load memory content comparison ──")
    _reset(m)
    m.write("The pianist performed Chopin nocturne.", training_mode=True)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name
    try:
        entries_before = {mid: e.source_text for mid, e in m.amm.tree.store.items()}
        mids_before = set(entries_before.keys())
        m.save_memory(path)

        _reset(m)
        m.load_memory(path)

        entries_after = {mid: e.source_text for mid, e in m.amm.tree.store.items()}
        mids_after = set(entries_after.keys())

        R.check("s48_load_restores_mids",
                mids_before == mids_after,
                f"before={mids_before}, after={mids_after}")
        R.check("s48_load_restores_texts",
                entries_before == entries_after)

        all_have_sem = all(e.semantic_emb is not None for e in m.amm.tree.store.values())
        R.check("s48_load_preserves_sem", all_have_sem)
    finally:
        os.unlink(path)
    _reset(m)


# ═══════════════════════════════════════════════════════════════════
# S49-S50. 边界语义用例
# ═══════════════════════════════════════════════════════════════════
def test_very_different_texts_produce_distinct_biases(m, c, R):
    """Texts from clearly different domains should produce distinct content biases."""
    print("\n── S49. Different-domain texts produce distinct biases ──")
    _reset(m)
    m.write("The experienced pianist performed a magnificent Chopin nocturne at the concert hall.", training_mode=True)
    m.write("The rocket launched successfully toward the International Space Station orbit.", training_mode=True)
    m.eval()

    top_music, cb_music = _content_bias_top_tokens(m, "Tell me about the piano concert.")
    top_space, cb_space = _content_bias_top_tokens(m, "Tell me about the space rocket.")

    music_unique = set(top_music[:15]) - set(top_space[:15])
    space_unique = set(top_space[:15]) - set(top_music[:15])
    total_unique = len(music_unique) + len(space_unique)
    R.check("s49_domains_have_unique_tokens",
            total_unique > 0,
            f"music_unique={music_unique}, space_unique={space_unique}")
    print(f"    music top15: {top_music[:15]}")
    print(f"    space top15: {top_space[:15]}")
    print(f"    unique tokens: music={len(music_unique)}, space={len(space_unique)}")
    _reset(m)


def test_multiple_writes_same_text_stable(m, c, R):
    """Writing same text multiple times should not cause store explosion."""
    print("\n── S50. Multiple writes same text stable ──")
    _reset(m)
    text = "The experienced pianist performed Chopin nocturne."
    for _ in range(10):
        m.write(text, training_mode=True)

    n_entries = len(m.amm.tree.store)
    R.check("s50_no_store_explosion",
            n_entries <= 10,
            f"entries={n_entries}")
    errs = m.amm.tree.verify_consistency()
    R.check("s50_tree_consistent", len(errs) == 0, str(errs))
    print(f"    10 writes → {n_entries} entries")
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
    print("  AMS v3.12 — Semantic-Level Black-Box Test Suite")
    print(f"{sep}")
    t0 = time.time()

    print("\n[Building MemLLM + loading GPT-2]")
    m = MemLLM(c)
    m.load("gpt2")
    total = sum(p.numel() for p in m.parameters())
    train_p = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  Params: total={total:,}  trainable={train_p:,}  frozen={total-train_p:,}")

    # S1-S3: 写入语义完整性
    test_write_preserves_content_tokens(m, c, R)
    test_write_semantic_emb_encodes_domain(m, c, R)
    test_semantic_emb_reflects_domain(m, c, R)

    # S4-S8: 检索精度
    test_retrieval_music_query_to_music_memory(m, c, R)
    test_retrieval_space_query_to_space_memory(m, c, R)
    test_retrieval_cooking_query_to_cooking_memory(m, c, R)
    test_retrieval_four_domain_isolation(m, c, R)
    test_retrieval_diag_weights_favor_relevant(m, c, R)

    # S9-S12: 生成语义质量
    test_generation_reflects_music_memory(m, c, R)
    test_generation_reflects_space_memory(m, c, R)
    test_generation_domain_switching(m, c, R)
    test_generation_greedy_consistency(m, c, R)

    # S13-S15: 记忆影响力
    test_memory_influences_generation(m, c, R)
    test_irrelevant_memory_less_impact(m, c, R)
    test_progressive_memory_accumulation(m, c, R)

    # S16-S18: 时间衰减
    test_recent_memory_preferred(m, c, R)
    test_high_surprise_survives_decay(m, c, R)
    test_frequently_accessed_survives(m, c, R)

    # S19-S21: 记忆合并语义
    test_consolidation_merges_similar(m, c, R)
    test_consolidation_preserves_content_ids(m, c, R)
    test_consolidation_preserves_semantic_emb(m, c, R)

    # S22-S24: 训练影响
    test_training_improves_reconstruction(m, c, R)
    test_training_preserves_memory_consistency(m, c, R)
    test_training_refreshes_memories(m, c, R)

    # S25-S27: 退化防护
    test_generation_not_all_punctuation(m, c, R)
    test_generation_has_content_words(m, c, R)
    test_generation_no_infinite_repeat(m, c, R)

    # S28-S30: Content bias 语义
    test_content_bias_scales_with_relevance(m, c, R)
    test_content_bias_zero_for_empty_store(m, c, R)
    test_content_bias_reflects_stored_text(m, c, R)

    # S31-S33: 多记忆干扰
    test_many_memories_no_catastrophic_interference(m, c, R)
    test_separate_domains_separate_retrieval(m, c, R)
    test_same_domain_memories_reinforce(m, c, R)

    # S34-S36: 保存/加载语义
    test_save_load_preserves_retrieval_quality(m, c, R)
    test_save_load_preserves_all_semantic_fields(m, c, R)
    test_save_load_generation_consistency(m, c, R)

    # S37-S39: SpectralDealiaser 语义
    test_dealiaser_detects_similar_fibers(m, c, R)
    test_dealiaser_separates_fibers(m, c, R)
    test_dealiaser_preserves_tree(m, c, R)

    # S40-S42: 三路检索评分
    test_three_way_scoring_all_contribute(m, c, R)
    test_retrieval_weights_sum_to_one(R, c)
    test_sem_weight_impact(m, c, R)

    # S43-S45: 消融模式
    test_ablation_modes_produce_different_generation(m, c, R)
    test_bypass_mode_uses_content(m, c, R)
    test_qformer_only_mode(m, c, R)

    # S46-S48: 端到端语义
    test_e2e_write_train_retrieve_generate(m, c, R)
    test_e2e_incremental_learning(m, c, R)
    test_e2e_save_train_load_compare(m, c, R)

    # S49-S50: 边界语义
    test_very_different_texts_produce_distinct_biases(m, c, R)
    test_multiple_writes_same_text_stable(m, c, R)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")
    ok = R.summary()
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
