#!/usr/bin/env python3
"""Session-layer viability spike for AMS v3.46-trained.

Per SPRINT_CLOSEOUT_v3.46.md \u00a710 decision framework: before continuing the
blackbox-audit climb (P0\u2013P4), quantify whether AMS is already useful as a
low-cost session layer between an application and an LLM.

Five modes compared on the SAME synthetic 20-turn session (+ optional
LongMemEval subset if present):

  D_full_history    : send ALL prior turns to the backbone (ceiling baseline)
  B_flat_cos        : flat-scan cosine over stored semantic_emb -> text inject
  B_ams_text        : full AMS retrieval (tree + gate + rerank) -> text inject
  A_ams_prefix      : AMS prefix injection ONLY (current blackbox mechanism)
  C_ams_hybrid      : AMS prefix + top-1 retrieved source_text

Metrics per mode:
  - write_ms / retrieve_ms / generate_ms (per turn)
  - input_tokens / output_tokens per turn
  - storage_bytes per memory
  - answer_hit (keyword in generated answer, case-insensitive)

Usage:
  python3 session_viability.py --out reports/session_viability --mt 40
  python3 session_viability.py --only-modes D_full_history,A_ams_prefix --mt 20

CPU-runnable (slow, ~1\u00d7 minutes per turn per mode); on H200 \u224520 turns x 5 modes
finishes in ~5 min.
"""
from __future__ import annotations
import argparse, json, os, sys, time, hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Callable

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scheme_b_v344 as sb


# =====================================================================
# 1. Synthetic session (fact-memorize + targeted-recall)
# =====================================================================

@dataclass
class Turn:
    idx: int
    kind: str      # "fact" | "query"
    text: str
    expected_keyword: Optional[str] = None  # for "query" turns

SYNTHETIC_SESSION: List[Turn] = [
    # 10 facts (will be written to memory for A/B/C; prepended as history for D)
    Turn(0,  "fact",  "I love classical piano, especially Chopin nocturnes."),
    Turn(1,  "fact",  "My favorite composer is Beethoven, particularly the Ninth Symphony."),
    Turn(2,  "fact",  "Last summer I traveled to Tokyo and visited the Shibuya crossing."),
    Turn(3,  "fact",  "I work as a software engineer on distributed systems."),
    Turn(4,  "fact",  "My dog is a golden retriever named Max, he is three years old."),
    Turn(5,  "fact",  "I started learning Mandarin Chinese in January this year."),
    Turn(6,  "fact",  "I collect vinyl records; my latest is Kind of Blue by Miles Davis."),
    Turn(7,  "fact",  "I am allergic to peanuts and shellfish, so I avoid Thai food."),
    Turn(8,  "fact",  "I use a mechanical keyboard with Cherry MX Brown switches for coding."),
    Turn(9,  "fact",  "My sister is a marine biologist studying coral reefs in Australia."),
    # 10 targeted recall queries, each with an expected keyword substring
    Turn(10, "query", "What kind of music do I love?",                          expected_keyword="chopin"),
    Turn(11, "query", "Who is my favorite composer?",                            expected_keyword="beethoven"),
    Turn(12, "query", "Where did I travel last summer?",                         expected_keyword="tokyo"),
    Turn(13, "query", "What is my job?",                                         expected_keyword="engineer"),
    Turn(14, "query", "What is my dog's name?",                                  expected_keyword="max"),
    Turn(15, "query", "What language am I learning this year?",                  expected_keyword="mandarin"),
    Turn(16, "query", "What is the latest record in my collection?",             expected_keyword="davis"),
    Turn(17, "query", "What cuisine should I avoid because of allergies?",       expected_keyword="thai"),
    Turn(18, "query", "What keyboard switches do I use?",                        expected_keyword="brown"),
    Turn(19, "query", "What does my sister study?",                              expected_keyword="coral"),
]


# =====================================================================
# 2. Measurement utilities
# =====================================================================

def _sync(dev: torch.device):
    if dev.type == "cuda": torch.cuda.synchronize()

def _timer(dev: torch.device):
    _sync(dev); return time.time()

def _tok_count(model, text: str) -> int:
    if not text: return 0
    ids = model.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return int(ids.shape[1])

def _contains_kw(text: str, kw: Optional[str]) -> bool:
    if not kw: return False
    return kw.lower() in text.lower()

def _mem_bytes_per_entry(model) -> int:
    """Estimated per-MemEntry storage in bytes (matches axis_a_compression spec)."""
    c = model.c
    # Per SPEC Section 4-meta.1 v3.45+: stored_floats_per_mem = 1712
    return 1712 * 4  # float32 equivalent


# =====================================================================
# 3. Mode runners \u2014 each takes (model, facts_so_far, query) -> (text, per_turn_metrics)
# =====================================================================

@dataclass
class TurnMetrics:
    write_ms: float = 0.0
    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    answer_hit: bool = False
    answer_text: str = ""

def _backbone_greedy(model, prompt_text: str, max_new: int) -> Tuple[str, int, int, float]:
    """Backbone-only generation (no AMS prefix). Returns (gen_text, input_tokens, output_tokens, gen_ms)."""
    dev = next(model.parameters()).device
    tk = model.tok(prompt_text, return_tensors="pt")
    ids = tk["input_ids"].to(dev); mask = tk["attention_mask"].to(dev)
    input_tokens = int(ids.shape[1])
    t0 = _timer(dev)
    with torch.no_grad():
        out = model.backbone.model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=max_new, do_sample=False,
            pad_token_id=model.tok.pad_token_id or model.tok.eos_token_id,
            use_cache=True,
        )
    _sync(dev); gen_ms = (time.time() - t0) * 1000
    new_ids = out[0, ids.shape[1]:].tolist()
    gen_text = model.tok.decode(new_ids, skip_special_tokens=True)
    return gen_text, input_tokens, len(new_ids), gen_ms


def run_mode_D(model, facts: List[Turn], query: Turn, mt: int) -> TurnMetrics:
    """D_full_history: everything in the prompt."""
    m = TurnMetrics()
    history = "\n".join(f"User: {f.text}" for f in facts)
    prompt = f"{history}\nUser: {query.text}\nAssistant:"
    gen, in_tok, out_tok, gen_ms = _backbone_greedy(model, prompt, mt)
    m.generate_ms = gen_ms; m.input_tokens = in_tok; m.output_tokens = out_tok
    m.answer_text = gen.strip()
    m.answer_hit = _contains_kw(m.answer_text, query.expected_keyword)
    return m


def _retrieve_flat_cos(model, query_text: str, topk: int) -> List[int]:
    """Flat-scan cosine retrieval over stored semantic_emb. Returns mids sorted by score desc.

    Computes the query embedding the same way MemLLM.write() does:
        o = fwd(ids, mask) -> list of per-layer hidden states
        hs_pooled = layer_pool(hs_list)                       [B, T, d_LLM]
        sem_q = _compute_content_semantic_emb(hs_pooled, ids, mask)   [B, d_LLM]
    so the query and stored embeddings live in the same space.
    """
    dev = next(model.parameters()).device
    tk = model.tok(query_text, return_tensors="pt")
    ids = tk["input_ids"].to(dev); mask = tk["attention_mask"].to(dev)
    with torch.no_grad():
        o = model.fwd(ids, mask)
        hs_pooled = model.layer_pool(o["hs"])
        sem_q = model._compute_content_semantic_emb(hs_pooled, ids, mask)  # [1, d_LLM]
    q = sem_q[0].float()
    store = model.amm.tree.store
    scored = []
    for mid, mem in store.items():
        if mem.semantic_emb is None: continue
        v = mem.semantic_emb.to(dev).float().flatten()
        if v.shape != q.shape:
            if v.numel() == q.numel(): v = v.view_as(q)
            else: continue
        sim = F.cosine_similarity(q.unsqueeze(0), v.unsqueeze(0)).item()
        scored.append((mid, sim))
    scored.sort(key=lambda x: -x[1])
    return [mid for mid, _ in scored[:topk]]


def run_mode_B_flat_cos(model, facts: List[Turn], query: Turn, mt: int, topk: int = 3) -> TurnMetrics:
    """B_flat_cos: flat cosine over semantic_emb, top-k texts prepended."""
    m = TurnMetrics()
    dev = next(model.parameters()).device
    t0 = _timer(dev)
    mids = _retrieve_flat_cos(model, query.text, topk)
    _sync(dev); m.retrieve_ms = (time.time() - t0) * 1000
    texts = [model.amm.tree.store[mid].source_text for mid in mids if mid in model.amm.tree.store]
    history = "\n".join(f"Context: {t}" for t in texts)
    prompt = f"{history}\nUser: {query.text}\nAssistant:" if history else f"User: {query.text}\nAssistant:"
    gen, in_tok, out_tok, gen_ms = _backbone_greedy(model, prompt, mt)
    m.generate_ms = gen_ms; m.input_tokens = in_tok; m.output_tokens = out_tok
    m.answer_text = gen.strip()
    m.answer_hit = _contains_kw(m.answer_text, query.expected_keyword)
    return m


def run_mode_B_ams_text(model, facts: List[Turn], query: Turn, mt: int, topk: int = 3) -> TurnMetrics:
    """B_ams_text: full AMS retrieval pipeline -> text inject."""
    m = TurnMetrics()
    dev = next(model.parameters()).device
    t0 = _timer(dev)
    ctx = model.prepare_decode_context(
        *_tokenize(model, query.text), update_stats=False)
    _sync(dev); m.retrieve_ms = (time.time() - t0) * 1000
    dom = ctx.diag.dominant_per_batch[0] if ctx.diag.dominant_per_batch else None
    non_dom = (ctx.diag.non_dominant_per_batch[0]
               if ctx.diag.non_dominant_per_batch and ctx.diag.non_dominant_per_batch[0]
               else [])
    mids: List[int] = []
    if dom is not None: mids.append(dom)
    mids.extend([x for x in non_dom if x not in mids][:max(0, topk - 1)])
    texts = [model.amm.tree.store[mid].source_text for mid in mids if mid in model.amm.tree.store]
    history = "\n".join(f"Context: {t}" for t in texts)
    prompt = f"{history}\nUser: {query.text}\nAssistant:" if history else f"User: {query.text}\nAssistant:"
    gen, in_tok, out_tok, gen_ms = _backbone_greedy(model, prompt, mt)
    m.generate_ms = gen_ms; m.input_tokens = in_tok; m.output_tokens = out_tok
    m.answer_text = gen.strip()
    m.answer_hit = _contains_kw(m.answer_text, query.expected_keyword)
    return m


def _tokenize(model, text: str):
    dev = next(model.parameters()).device
    tk = model.tok(text, return_tensors="pt")
    return tk["input_ids"].to(dev), tk["attention_mask"].to(dev)


def run_mode_A_ams_prefix(model, facts: List[Turn], query: Turn, mt: int) -> TurnMetrics:
    """A_ams_prefix: AMS prefix injection only, no history text."""
    m = TurnMetrics()
    dev = next(model.parameters()).device
    prompt = f"User: {query.text}\nAssistant:"
    ids, mask = _tokenize(model, prompt)
    input_tokens = int(ids.shape[1])
    t0 = _timer(dev)
    ctx = model.prepare_decode_context(ids, mask, update_stats=False)
    _sync(dev); m.retrieve_ms = (time.time() - t0) * 1000
    t1 = _timer(dev)
    gen_text = model.generate(prompt, mt=mt, greedy=True)
    _sync(dev); m.generate_ms = (time.time() - t1) * 1000
    new_text = gen_text[len(prompt):] if gen_text.startswith(prompt) else gen_text
    m.input_tokens = input_tokens
    m.output_tokens = _tok_count(model, new_text)
    m.answer_text = new_text.strip()
    m.answer_hit = _contains_kw(m.answer_text, query.expected_keyword)
    return m


def run_mode_C_hybrid(model, facts: List[Turn], query: Turn, mt: int) -> TurnMetrics:
    """C_hybrid: AMS prefix + top-1 retrieved source_text as short context."""
    m = TurnMetrics()
    dev = next(model.parameters()).device
    t0 = _timer(dev)
    ids_q, mask_q = _tokenize(model, query.text)
    ctx = model.prepare_decode_context(ids_q, mask_q, update_stats=False)
    _sync(dev); m.retrieve_ms = (time.time() - t0) * 1000
    dom = ctx.diag.dominant_per_batch[0] if ctx.diag.dominant_per_batch else None
    top1_text = ""
    if dom is not None and dom in model.amm.tree.store:
        top1_text = model.amm.tree.store[dom].source_text
    prompt = (f"Context: {top1_text}\nUser: {query.text}\nAssistant:"
              if top1_text else f"User: {query.text}\nAssistant:")
    ids, mask = _tokenize(model, prompt)
    input_tokens = int(ids.shape[1])
    t1 = _timer(dev)
    gen_text = model.generate(prompt, mt=mt, greedy=True)
    _sync(dev); m.generate_ms = (time.time() - t1) * 1000
    new_text = gen_text[len(prompt):] if gen_text.startswith(prompt) else gen_text
    m.input_tokens = input_tokens
    m.output_tokens = _tok_count(model, new_text)
    m.answer_text = new_text.strip()
    m.answer_hit = _contains_kw(m.answer_text, query.expected_keyword)
    return m


MODE_RUNNERS: Dict[str, Callable] = {
    "D_full_history": run_mode_D,
    "B_flat_cos":     run_mode_B_flat_cos,
    "B_ams_text":     run_mode_B_ams_text,
    "A_ams_prefix":   run_mode_A_ams_prefix,
    "C_ams_hybrid":   run_mode_C_hybrid,
}


# =====================================================================
# 4. Driver
# =====================================================================

def _build_model(seed: int = 42) -> Tuple[sb.MemLLM, torch.device]:
    torch.manual_seed(seed)
    c = sb.Cfg()
    m = sb.MemLLM(c)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(dev); m.load(); m.to(dev); m.eval()
    return m, dev

def _seed_memory(model, facts: List[Turn]) -> float:
    """Write facts to AMS memory. Returns total write_ms.

    Uses training_mode=True so the write-gate never silently drops a fact
    (default threshold can block routine-surprise inputs).  This is a
    measurement setup, not a training claim.
    """
    dev = next(model.parameters()).device
    t0 = _timer(dev)
    for f in facts:
        model.write(f.text, training_mode=True)
    try:
        model.amm.maybe_recluster(force=True)
    except Exception as e:
        print(f"  [seed] maybe_recluster skipped: {type(e).__name__}: {e}")
    model._refresh_rare_keyword_indices()
    _sync(dev)
    stored = len(model.amm.tree.store)
    if stored != len(facts):
        print(f"  [seed] WARN: stored={stored} != facts={len(facts)}")
    return (time.time() - t0) * 1000

def run_session_for_mode(model, session: List[Turn], mode: str, mt: int) -> Dict[str, Any]:
    """For a single mode, ingest facts (write to AMS if needed), then run each query turn."""
    runner = MODE_RUNNERS[mode]
    facts = [t for t in session if t.kind == "fact"]
    queries = [t for t in session if t.kind == "query"]

    # D_full_history doesn't need AMS memory; B/A/C do.
    write_ms_total = 0.0
    if mode != "D_full_history":
        # Reset retrieval tree cleanly without tearing down the AMM modules
        model.amm.tree = sb.DirectionTree(model.c, amm_ref=model.amm)
        model.amm.time = 0.0
        model.amm._writes_since_recluster = 0
        if hasattr(model, "_wte_normed") and model._wte_normed is not None:
            model.amm.wte_normed = model._wte_normed
        if hasattr(model, "content_classifier") and model.content_classifier is not None:
            model.amm._content_classifier = model.content_classifier
        write_ms_total = _seed_memory(model, facts)

    turn_records: List[Dict[str, Any]] = []
    for q in queries:
        try:
            tm = runner(model, facts, q, mt)
        except Exception as e:
            print(f"  [turn {q.idx}] EXCEPTION in {mode}: {type(e).__name__}: {e}")
            tm = TurnMetrics(answer_text=f"ERROR {type(e).__name__}: {e}")
        rec = {
            "turn_idx": q.idx, "query": q.text,
            "expected_keyword": q.expected_keyword,
            **asdict(tm),
        }
        turn_records.append(rec)
        hit = "HIT" if tm.answer_hit else "    "
        print(f"  [{mode} t{q.idx:2d}] {hit}  ret={tm.retrieve_ms:6.1f}ms  "
              f"gen={tm.generate_ms:7.1f}ms  in={tm.input_tokens:4d}t  out={tm.output_tokens:3d}t  "
              f"kw='{q.expected_keyword}' ans='{tm.answer_text[:70]}'")
    return {
        "mode": mode,
        "n_facts": len(facts),
        "n_queries": len(queries),
        "write_ms_total": write_ms_total,
        "turns": turn_records,
    }

def aggregate(mode_result: Dict[str, Any]) -> Dict[str, Any]:
    turns = mode_result["turns"]
    n = len(turns)
    def _avg(k): return sum(t[k] for t in turns) / n if n else 0.0
    return {
        "mode": mode_result["mode"],
        "n_queries": n,
        "hit_rate": sum(1 for t in turns if t["answer_hit"]) / max(1, n),
        "avg_retrieve_ms": _avg("retrieve_ms"),
        "avg_generate_ms": _avg("generate_ms"),
        "avg_input_tokens": _avg("input_tokens"),
        "avg_output_tokens": _avg("output_tokens"),
        "write_ms_total": mode_result["write_ms_total"],
    }

def render_markdown(results: Dict[str, Dict[str, Any]], out_path: str, mt: int,
                    model_name: str, device_name: str, trained_path: Optional[str]):
    rows = [aggregate(r) for r in results.values()]
    lines = [
        "# Session-layer viability \u00b7 v3.46-trained",
        "",
        f"- Backbone: `{model_name}`",
        f"- Device: `{device_name}`",
        f"- Trained weights: `{trained_path or '(none, fresh init)'}`",
        f"- Max new tokens per query: `{mt}`",
        f"- Synthetic session: {len(SYNTHETIC_SESSION)} turns "
        f"({sum(1 for t in SYNTHETIC_SESSION if t.kind=='fact')} facts + "
        f"{sum(1 for t in SYNTHETIC_SESSION if t.kind=='query')} queries)",
        "",
        "## Decision table",
        "",
        "| Mode | Hit-rate | avg in-tokens | avg out-tokens | avg retrieve ms | avg generate ms | total write ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['mode']}` | {r['hit_rate']*100:.0f}% "
            f"| {r['avg_input_tokens']:.0f} "
            f"| {r['avg_output_tokens']:.0f} "
            f"| {r['avg_retrieve_ms']:.1f} "
            f"| {r['avg_generate_ms']:.1f} "
            f"| {r['write_ms_total']:.0f} |"
        )
    lines += [
        "",
        "## Per-turn detail",
        "",
    ]
    for mode, res in results.items():
        lines.append(f"### `{mode}`")
        lines.append("")
        lines.append("| turn | expected | hit | in | out | ret ms | gen ms | answer (first 80 chars) |")
        lines.append("|---:|---|:---:|---:|---:|---:|---:|---|")
        for t in res["turns"]:
            hit = "\u2705" if t["answer_hit"] else "\u274c"
            ans = t["answer_text"].replace("|", "\\|").replace("\n", " ")[:80]
            lines.append(
                f"| {t['turn_idx']} | `{t['expected_keyword']}` | {hit} "
                f"| {t['input_tokens']} | {t['output_tokens']} "
                f"| {t['retrieve_ms']:.1f} | {t['generate_ms']:.1f} | {ans} |"
            )
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="reports/session_viability",
                    help="Output directory for report.json + report.md")
    ap.add_argument("--mt", type=int, default=40,
                    help="max_new_tokens per generated answer")
    ap.add_argument("--only-modes", type=str, default="",
                    help="Comma-separated subset of modes to run (default: all)")
    ap.add_argument("--skip-modes", type=str, default="",
                    help="Comma-separated modes to skip")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    trained_path = os.environ.get("AMS_TRAINED_WEIGHTS", "").strip() or None

    print("=" * 70)
    print(f"Session-layer viability spike")
    print(f"  AMS_TRAINED_WEIGHTS = {trained_path or '(not set, fresh init)'}")
    print(f"  max_new_tokens     = {args.mt}")
    print(f"  session turns      = {len(SYNTHETIC_SESSION)}")
    print("=" * 70)

    model, dev = _build_model(seed=args.seed)
    model_name = getattr(model.c, "llm_name", "?")
    dev_name = (torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")
    print(f"  backbone = {model_name}   device = {dev_name}")

    wanted = [m for m in MODE_RUNNERS]
    if args.only_modes.strip():
        wanted = [m.strip() for m in args.only_modes.split(",") if m.strip()]
    if args.skip_modes.strip():
        skip = {m.strip() for m in args.skip_modes.split(",") if m.strip()}
        wanted = [m for m in wanted if m not in skip]
    print(f"  modes              = {wanted}")

    results: Dict[str, Dict[str, Any]] = {}
    for mode in wanted:
        if mode not in MODE_RUNNERS:
            print(f"  [skip] unknown mode: {mode}"); continue
        print(f"\n--- mode: {mode} ---")
        t0 = _timer(dev)
        res = run_session_for_mode(model, SYNTHETIC_SESSION, mode, args.mt)
        _sync(dev); elapsed = time.time() - t0
        res["elapsed_s"] = elapsed
        print(f"  [{mode}] elapsed {elapsed:.1f}s  hit_rate={aggregate(res)['hit_rate']*100:.0f}%")
        results[mode] = res

    out_json = os.path.join(args.out, "report.json")
    out_md   = os.path.join(args.out, "report.md")
    blob = {
        "generated_at_epoch": time.time(),
        "config": {
            "max_new_tokens": args.mt,
            "seed": args.seed,
            "modes": wanted,
            "trained_weights_path": trained_path,
            "backbone": model_name,
            "device": dev_name,
        },
        "session": [asdict(t) for t in SYNTHETIC_SESSION],
        "results": results,
        "aggregates": [aggregate(r) for r in results.values()],
        "storage_bytes_per_memory_estimate": _mem_bytes_per_entry(model),
    }
    with open(out_json, "w") as f:
        json.dump(blob, f, indent=2, default=str)
    render_markdown(results, out_md, args.mt, model_name, dev_name, trained_path)
    print(f"\n[done] report.json -> {out_json}")
    print(f"[done] report.md   -> {out_md}")

    # Short stdout decision table
    print("\n" + "=" * 70)
    print("Decision table (hit-rate / avg-in-tokens / avg-retrieve-ms / avg-gen-ms):")
    print("=" * 70)
    for r in blob["aggregates"]:
        print(f"  {r['mode']:18s}  hit={r['hit_rate']*100:3.0f}%  "
              f"in_tok={r['avg_input_tokens']:5.0f}  "
              f"ret={r['avg_retrieve_ms']:6.1f}ms  "
              f"gen={r['avg_generate_ms']:7.1f}ms")


if __name__ == "__main__":
    main()
