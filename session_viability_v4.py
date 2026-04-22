#!/usr/bin/env python3
"""Session-layer viability spike for AMS v4 (fresh-init).

Same synthetic session + 10 targeted-recall queries as PR #29's
`session_viability.py`, but using MemLLM4 for A_ams_prefix / C_ams_hybrid.

v4.6 will add a Trainer; this harness runs FRESH-INIT only — the goal is to
prove the v4 stack composes and runs at Qwen2.5-1.5B scale on GPU, and to
produce a baseline that v4-trained numbers can be compared against.

Not expected to beat v3.46 fresh-init numbers on A_ams_prefix / C_ams_hybrid
— that is specifically what training is for.

Usage:
  python3 session_viability_v4.py --mt 30 --n-facts 10 --out reports/session_viability_v4_fresh
  python3 session_viability_v4.py --mt 30 --n-facts 20 --out reports/session_viability_v4_fresh_20facts
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ams_v4 import Cfg4, MemLLM4


# ─── Synthetic session (same as session_viability.py) ─────────────────────

@dataclass
class Turn:
    idx: int
    kind: str
    text: str
    expected_keyword: Optional[str] = None


_FACTS_20 = [
    "I love classical piano, especially Chopin nocturnes.",
    "My favorite composer is Beethoven, particularly the Ninth Symphony.",
    "Last summer I traveled to Tokyo and visited the Shibuya crossing.",
    "I work as a software engineer on distributed systems.",
    "My dog is a golden retriever named Max, he is three years old.",
    "I started learning Mandarin Chinese in January this year.",
    "I collect vinyl records; my latest is Kind of Blue by Miles Davis.",
    "I am allergic to peanuts and shellfish, so I avoid Thai food.",
    "I use a mechanical keyboard with Cherry MX Brown switches for coding.",
    "My sister is a marine biologist studying coral reefs in Australia.",
    "Chess openings like the Sicilian Defense require deep theoretical study.",
    "Sourdough bread depends on long fermentation for a complex flavor.",
    "Marathons require consistent training plans spread over several months.",
    "Film noir often uses low-key lighting and moral ambiguity.",
    "Lunar eclipses occur when Earth sits between the Sun and the Moon.",
    "Kubernetes schedules containers across a cluster using a control plane.",
    "Tea ceremonies in Kyoto follow precise, centuries-old protocols.",
    "Ancient Rome's aqueducts carried water across tens of kilometers.",
    "Sudoku puzzles are constraint-satisfaction problems solvable by backtracking.",
    "Honey crystallizes faster when stored below about ten degrees Celsius.",
]

_QUERIES_10 = [
    ("What kind of music do I love?",                         "chopin"),
    ("Who is my favorite composer?",                          "beethoven"),
    ("Where did I travel last summer?",                       "tokyo"),
    ("What is my job?",                                       "engineer"),
    ("What is my dog's name?",                                "max"),
    ("What language am I learning this year?",                "mandarin"),
    ("What is the latest record in my collection?",           "davis"),
    ("What cuisine should I avoid because of allergies?",     "thai"),
    ("What keyboard switches do I use?",                      "brown"),
    ("What does my sister study?",                            "coral"),
]


def build_session(n_facts: int = 10) -> List[Turn]:
    n_facts = max(1, min(n_facts, len(_FACTS_20)))
    facts = [Turn(i, "fact", _FACTS_20[i]) for i in range(n_facts)]
    queries = [
        Turn(n_facts + i, "query", q, expected_keyword=kw)
        for i, (q, kw) in enumerate(_QUERIES_10)
    ]
    return facts + queries


# ─── Measurement ─────────────────────────────────────────────────────────

def _sync(dev: torch.device):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _timer(dev: torch.device) -> float:
    _sync(dev)
    return time.time()


@dataclass
class TurnMetrics:
    write_ms: float = 0.0
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    answer_hit: bool = False
    answer_text: str = ""


def _contains_kw(text: str, kw: Optional[str]) -> bool:
    if not kw:
        return False
    return kw.lower() in text.lower()


# ─── Modes ───────────────────────────────────────────────────────────────

def run_mode_D(model: MemLLM4, facts: List[Turn], query: Turn, mt: int) -> TurnMetrics:
    """D_full_history: every fact in prompt, no AMS retrieval."""
    m = TurnMetrics()
    dev = model.backbone.device
    history = "\n".join(f"User: {f.text}" for f in facts)
    prompt = f"{history}\nUser: {query.text}\nAssistant:"
    ids, mask = model.backbone.tokenize(prompt)
    m.input_tokens = int(ids.shape[1])
    t0 = _timer(dev)
    with torch.no_grad():
        # Use HF's built-in generate() for baseline (no prefix)
        out = model.backbone.model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=mt, do_sample=False,
            pad_token_id=model.backbone.tok.pad_token_id or model.backbone.tok.eos_token_id,
            use_cache=True,
        )
    _sync(dev)
    m.generate_ms = (time.time() - t0) * 1000
    new_ids = out[0, ids.shape[1]:].tolist()
    m.output_tokens = len(new_ids)
    m.answer_text = model.backbone.tok.decode(new_ids, skip_special_tokens=True).strip()
    m.answer_hit = _contains_kw(m.answer_text, query.expected_keyword)
    return m


def run_mode_A(model: MemLLM4, facts: List[Turn], query: Turn, mt: int) -> TurnMetrics:
    """A_ams_prefix: AMS v4 prefix injection only, no history text."""
    m = TurnMetrics()
    dev = model.backbone.device
    prompt = f"User: {query.text}\nAssistant:"
    ids, mask = model.backbone.tokenize(prompt)
    m.input_tokens = int(ids.shape[1])
    t0 = _timer(dev)
    ctx = model.prepare_decode_context(ids, mask)
    _sync(dev)
    m.retrieve_ms = (time.time() - t0) * 1000
    t1 = _timer(dev)
    gen_text = model.generate(prompt, mt=mt, greedy=True)
    _sync(dev)
    m.generate_ms = (time.time() - t1) * 1000
    m.output_tokens = len(model.backbone.tok(gen_text, add_special_tokens=False)["input_ids"])
    m.answer_text = gen_text.strip()
    m.answer_hit = _contains_kw(m.answer_text, query.expected_keyword)
    return m


def run_mode_C(model: MemLLM4, facts: List[Turn], query: Turn, mt: int) -> TurnMetrics:
    """C_ams_hybrid: AMS v4 prefix + top-1 source_text in prompt.

    Top-1 retrieved via topic DirectionTreeV4 — this is the v4 equivalent of
    what v3.46 C_ams_hybrid did via `prepare_decode_context.diag.dominant_per_batch`.
    """
    m = TurnMetrics()
    dev = model.backbone.device

    t0 = _timer(dev)
    # Retrieve top-1 from the topic tree using the same pooled-hidden-state
    # query as the cross-bundle attention
    q_ids, q_mask = model.backbone.tokenize(query.text)
    q_hs = model.backbone.hidden_states(q_ids, q_mask)
    mq = q_mask.unsqueeze(-1).to(q_hs.dtype)
    pooled_q = ((q_hs * mq).sum(dim=1) / mq.sum(dim=1).clamp(min=1e-6)).float()
    # Project into topic bundle query (uses the canonical TopicEncoder
    # mapping — for retrieval we need a topic-space vector). Reuse the
    # topic encoder with empty content tokens so we get a base from hidden
    # projection only.
    W = model._wte_normed_cache.to(pooled_q.device)
    _, _, topic_q_dirn = model.bundle_topic.encode(
        pooled_q, content_token_ids=[[]], wte_normed=W,
    )
    hits = model.store.tree_topic.retrieve(topic_q_dirn[0].detach(),
                                           beam=model.cfg.retrieval_beam)
    _sync(dev)
    m.retrieve_ms = (time.time() - t0) * 1000

    top_text = ""
    if hits:
        top_mid = hits[0][0]
        entry = model.store.get(top_mid)
        if entry is not None:
            top_text = entry.source_text

    prompt = (f"Context: {top_text}\nUser: {query.text}\nAssistant:"
              if top_text else f"User: {query.text}\nAssistant:")
    ids, mask = model.backbone.tokenize(prompt)
    m.input_tokens = int(ids.shape[1])

    t1 = _timer(dev)
    gen_text = model.generate(prompt, mt=mt, greedy=True)
    _sync(dev)
    m.generate_ms = (time.time() - t1) * 1000

    m.output_tokens = len(model.backbone.tok(gen_text, add_special_tokens=False)["input_ids"])
    m.answer_text = gen_text.strip()
    m.answer_hit = _contains_kw(m.answer_text, query.expected_keyword)
    return m


MODE_RUNNERS: Dict[str, Callable] = {
    "D_full_history": run_mode_D,
    "A_ams_prefix":   run_mode_A,
    "C_ams_hybrid":   run_mode_C,
}


# ─── Driver ──────────────────────────────────────────────────────────────

def _build_model(seed: int, llm_name: str,
                 trained_weights: Optional[str] = None) -> MemLLM4:
    torch.manual_seed(seed)
    from transformers import AutoConfig
    ac = AutoConfig.from_pretrained(llm_name)
    cfg = Cfg4(
        llm_name=llm_name,
        d_LLM=ac.hidden_size,
        vocab_size=ac.vocab_size,
    )
    model = MemLLM4(cfg)
    model.load()
    if trained_weights:
        model.load_trained_weights(trained_weights)
    return model


def _seed_memory(model: MemLLM4, facts: List[Turn]) -> float:
    dev = model.backbone.device
    t0 = _timer(dev)
    for f in facts:
        model.write(f.text)
    _sync(dev)
    return (time.time() - t0) * 1000


def run_session_for_mode(model: MemLLM4, session: List[Turn], mode: str, mt: int):
    runner = MODE_RUNNERS[mode]
    facts = [t for t in session if t.kind == "fact"]
    queries = [t for t in session if t.kind == "query"]

    # Reset memory (fresh store) for every mode; D doesn't use it, but a clean
    # store makes numbers comparable.
    if mode != "D_full_history":
        from ams_v4.core.mem_store import MemStore
        from ams_v4.kakeya.registry import KakeyaRegistry
        model.store = MemStore(model.cfg)
        model.kakeya = KakeyaRegistry(model.cfg)
        model._session_summary = None

    write_ms_total = 0.0
    if mode != "D_full_history":
        write_ms_total = _seed_memory(model, facts)

    turn_records: List[Dict[str, Any]] = []
    for q in queries:
        try:
            tm = runner(model, facts, q, mt)
        except Exception as e:
            import traceback
            traceback.print_exc()
            tm = TurnMetrics(answer_text=f"ERROR {type(e).__name__}: {e}")
        rec = {
            "turn_idx": q.idx,
            "query": q.text,
            "expected_keyword": q.expected_keyword,
            **asdict(tm),
        }
        turn_records.append(rec)
        hit = "HIT " if tm.answer_hit else "    "
        print(
            f"  [{mode} t{q.idx:2d}] {hit} ret={tm.retrieve_ms:7.1f}ms "
            f"gen={tm.generate_ms:8.1f}ms in={tm.input_tokens:4d}t "
            f"out={tm.output_tokens:3d}t kw={q.expected_keyword!r} "
            f"ans={tm.answer_text[:70]!r}"
        )
    return {
        "mode": mode,
        "n_facts": len(facts),
        "n_queries": len(queries),
        "write_ms_total": write_ms_total,
        "turns": turn_records,
    }


def aggregate(res: Dict[str, Any]) -> Dict[str, Any]:
    turns = res["turns"]
    n = len(turns)

    def _avg(k):
        return sum(t[k] for t in turns) / n if n else 0.0

    return {
        "mode": res["mode"],
        "n_queries": n,
        "hit_rate": sum(1 for t in turns if t["answer_hit"]) / max(1, n),
        "avg_retrieve_ms": _avg("retrieve_ms"),
        "avg_generate_ms": _avg("generate_ms"),
        "avg_input_tokens": _avg("input_tokens"),
        "avg_output_tokens": _avg("output_tokens"),
        "write_ms_total": res["write_ms_total"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="reports/session_viability_v4_fresh")
    ap.add_argument("--mt", type=int, default=30)
    ap.add_argument("--n-facts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--llm-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--only-modes", type=str, default="")
    ap.add_argument("--trained-weights", type=str, default="",
                    help="path to v4 trainer checkpoint (ckpt/v4_trained.pt). "
                         "If empty, runs fresh-init.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    session = build_session(args.n_facts)

    print("=" * 70)
    mode_label = "trained" if args.trained_weights else "fresh-init"
    print(f"Session-layer viability spike · AMS v4 ({mode_label})")
    print(f"  backbone = {args.llm_name}")
    if args.trained_weights:
        print(f"  trained weights = {args.trained_weights}")
    print(f"  max_new_tokens = {args.mt}")
    print(f"  session turns = {len(session)} "
          f"({sum(1 for t in session if t.kind=='fact')} facts + "
          f"{sum(1 for t in session if t.kind=='query')} queries)")
    print("=" * 70)

    model = _build_model(
        args.seed, args.llm_name,
        trained_weights=args.trained_weights or None,
    )
    dev_name = (
        torch.cuda.get_device_name(0) if model.backbone.device.type == "cuda" else "cpu"
    )
    print(f"  device = {dev_name}")

    modes = list(MODE_RUNNERS.keys())
    if args.only_modes.strip():
        modes = [m.strip() for m in args.only_modes.split(",") if m.strip()]

    results: Dict[str, Dict[str, Any]] = {}
    for mode in modes:
        if mode not in MODE_RUNNERS:
            print(f"  [skip] unknown mode: {mode}")
            continue
        print(f"\n--- mode: {mode} ---")
        t0 = _timer(model.backbone.device)
        res = run_session_for_mode(model, session, mode, args.mt)
        _sync(model.backbone.device)
        res["elapsed_s"] = time.time() - t0
        results[mode] = res
        agg = aggregate(res)
        print(f"  [{mode}] elapsed {res['elapsed_s']:.1f}s "
              f"hit_rate={agg['hit_rate']*100:.0f}%")

    blob = {
        "generated_at_epoch": time.time(),
        "config": {
            "max_new_tokens": args.mt,
            "seed": args.seed,
            "modes": modes,
            "backbone": args.llm_name,
            "device": dev_name,
            "n_facts": args.n_facts,
            "trained_weights": args.trained_weights or None,
        },
        "session": [asdict(t) for t in session],
        "results": results,
        "aggregates": [aggregate(r) for r in results.values()],
    }
    out_json = os.path.join(args.out, "report.json")
    with open(out_json, "w") as f:
        json.dump(blob, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("Decision table (hit-rate / avg-in-tokens / avg-gen-ms):")
    print("=" * 70)
    for r in blob["aggregates"]:
        print(f"  {r['mode']:18s} hit={r['hit_rate']*100:3.0f}% "
              f"in_tok={r['avg_input_tokens']:5.0f} "
              f"ret={r['avg_retrieve_ms']:6.1f}ms "
              f"gen={r['avg_generate_ms']:7.1f}ms")
    print(f"\n[done] report.json -> {out_json}")


if __name__ == "__main__":
    main()
