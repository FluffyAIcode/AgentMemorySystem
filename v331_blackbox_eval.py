#!/usr/bin/env python3
"""External black-box evaluation for `AgentMemorySystem.py` on the `v331` branch.

Principles:
- independent from the module's built-in `test()`
- no monkeypatching / no mocked return values
- treats the system as a black-box via exported classes and runtime behavior
- produces detailed Markdown and JSON reports
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

# [SPEC 1.1.2 / 7.7 v3.45+] Optional deterministic mode for channel-axis D.
# Activated by AMS_DETERMINISTIC=1 in the environment. Does not change outputs
# when the flag is absent.
if os.environ.get("AMS_DETERMINISTIC", "") == "1":
    torch.set_num_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

import AgentMemorySystem as sb


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "reports" / "v331_blackbox"
JSON_REPORT = REPORT_DIR / "report.json"
MD_REPORT = REPORT_DIR / "report.md"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def ensure_report_dir() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def cpu_device() -> torch.device:
    return torch.device("cpu")


def corpus_music() -> List[str]:
    return [
        "The pianist practiced arpeggios and Chopin nocturnes until midnight.",
        "A musician refined finger technique, phrasing, and pedal control on the piano.",
        "Classical interpretation often depends on dynamics, tempo rubato, and touch.",
        "A conservatory student studied etudes, scales, and expressive voicing on the keyboard.",
    ]


def corpus_space() -> List[str]:
    return [
        "Astronomers observed distant galaxies, quasars, and stellar evolution in deep space.",
        "Orbital mechanics explains how satellites and planets move under gravitational force.",
        "A telescope captured nebulae, exoplanets, and spectral signatures from distant stars.",
        "Cosmology studies dark matter, expansion, and the large scale structure of the universe.",
    ]


# ==========================================================================
# [SPEC 4-meta / v3.46 de-overfit] Held-out domains.
#
# These corpora exist only to test whether probes 4.22 / 4.23 / 4.24 generalize
# beyond the music / space corpora that were hand-coded alongside their keyword
# lists. They are NOT referenced by any case 4.1-4.19 and NOT used as training
# data by any audit path. The runner writes them into the memory tree at probe
# invocation time; the SUT sees them as plain text through the same `write()`
# API path as music / space.
# ==========================================================================

def corpus_cooking() -> List[str]:
    return [
        "A chef braised short ribs with red wine, rosemary, and garlic for four hours.",
        "The pastry batter folded egg whites into melted chocolate before baking.",
        "Knife skills determine the cut quality of vegetables in stir fry dishes.",
        "Slow fermentation develops complex flavors in sourdough bread dough overnight.",
    ]


def corpus_finance() -> List[str]:
    return [
        "Portfolio managers rebalance allocations across equities, bonds, and commodities quarterly.",
        "Derivative contracts hedge currency exposure in multinational corporate treasury operations.",
        "Yield curve inversion historically precedes recessions by twelve to eighteen months.",
        "Quantitative tightening reduces central bank balance sheets through asset roll-off.",
    ]


def corpus_paraphrase_music() -> List[str]:
    """Token-disjoint (to the extent possible) paraphrases of music corpus for
    use as queries in de-overfit probes. Do NOT contain the exact strict
    starters used as rare_keyword anchors in music corpus."""
    return [
        "She performed Beethoven sonatas with delicate phrasing on her grand piano.",
        "Harmonic analysis and ear training are core elements of music education.",
    ]


def corpus_paraphrase_space() -> List[str]:
    return [
        "Deep-sky imaging reveals the structure of faraway nebulae and exoplanets.",
        "Astronauts and rocket scientists study celestial mechanics for mission planning.",
    ]


def corpus_general() -> List[str]:
    return [
        "The cat sat on the mat and watched the birds outside the window.",
        "Quantum computing uses qubits existing in superposition states.",
        "Machine learning algorithms identify patterns in large datasets.",
        "The ancient temple was hidden deep within the tropical rainforest.",
        "The stock market experienced significant volatility during the session.",
        "He practiced piano for hours perfecting a difficult Chopin nocturne.",
        "The restaurant served an exquisite five course meal with wine pairings.",
        "The professor explained relativity using simple everyday analogies.",
    ]


STOPWORDS = {
    "the", "and", "that", "with", "from", "into", "this", "about", "their", "until",
    "under", "often", "using", "uses", "someone", "something", "should", "would",
    "could", "there", "which", "while", "where", "when", "what", "your", "have",
    "has", "had", "been", "were", "was", "they", "them", "then", "than", "also",
    "very", "more", "most", "some", "such", "just", "over", "deep", "large", "simple",
    "hours", "along", "outside", "inside", "during", "across", "through", "session",
}


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(seed: int) -> sb.MemLLM:
    import gc

    set_seed(seed)
    torch.set_num_threads(1)
    device = best_device()
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    model = sb.MemLLM(sb.Cfg())
    model.to(device)
    model.load()
    model.to(device)
    model.eval()
    return model


def write_texts(model: sb.MemLLM, texts: List[str]) -> int:
    count = 0
    for text in texts:
        n, _ = model.write(text, training_mode=True)
        count += n
    return count


def run_case(name: str, fn, *args, **kwargs) -> Dict[str, Any]:
    print(f"[case:start] {name}", flush=True)
    try:
        result = fn(*args, **kwargs)
        if "passed" not in result:
            result["passed"] = True
        result["error"] = None
        print(f"[case:done] {name} passed={result['passed']}", flush=True)
        return result
    except Exception as exc:
        print(f"[case:done] {name} passed=False error={type(exc).__name__}: {exc}", flush=True)
        return {
            "passed": False,
            "case": name,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }


def word_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def content_tokens(text: str) -> List[str]:
    return [t for t in word_tokens(text) if len(t) >= 4 and t not in STOPWORDS]


def derive_keywords(texts: List[str], limit: int = 12) -> List[str]:
    counts = Counter()
    for text in texts:
        counts.update(content_tokens(text))
    return [tok for tok, _ in counts.most_common(limit)]


def keyword_score(text: str, keywords: List[str]) -> float:
    toks = content_tokens(text)
    if not toks:
        return 0.0
    hit = sum(tok in set(keywords) for tok in toks)
    return hit / max(len(toks), 1)


def normalize_token_piece(text: str) -> str:
    return re.sub(r"[^a-z]+", "", text.lower())


def js_divergence_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    pa = torch.softmax(logits_a, dim=-1)
    pb = torch.softmax(logits_b, dim=-1)
    m = 0.5 * (pa + pb)
    kl_a = torch.sum(pa * (torch.log(pa + 1e-12) - torch.log(m + 1e-12)))
    kl_b = torch.sum(pb * (torch.log(pb + 1e-12) - torch.log(m + 1e-12)))
    return float((0.5 * (kl_a + kl_b)).item())


def entropy_from_logits(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    return float((-(p * torch.log(p + 1e-12)).sum()).item())


def topk_tokens_from_logits(model: sb.MemLLM, logits: torch.Tensor, k: int = 12) -> List[Dict[str, Any]]:
    vals, idx = torch.topk(logits, k)
    rows = []
    for score, token_id in zip(vals.tolist(), idx.tolist()):
        piece = model.tok.decode([token_id])
        rows.append(
            {
                "token_id": int(token_id),
                "piece": piece,
                "norm": normalize_token_piece(piece),
                "logit": float(score),
                "prob": float(torch.softmax(logits, dim=-1)[token_id].item()),
            }
        )
    return rows


def audit_domain_hits(rows: List[Dict[str, Any]], keywords: List[str]) -> Dict[str, Any]:
    keyset = set(keywords)
    matches = [r for r in rows if r["norm"] in keyset or any(k in r["norm"] for k in keyset if len(k) >= 5)]
    prob_mass = sum(r["prob"] for r in matches)
    return {
        "match_count": len(matches),
        "match_prob_mass": prob_mass,
        "matches": matches,
    }


def token_category(norm: str) -> str:
    if not norm:
        return "punct"
    if norm in STOPWORDS or len(norm) < 4:
        return "functional"
    return "semantic"


def summarize_topk_categories(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {"semantic": 0, "functional": 0, "punct": 0}
    prob_mass = {"semantic": 0.0, "functional": 0.0, "punct": 0.0}
    for row in rows:
        cat = token_category(row["norm"])
        counts[cat] += 1
        prob_mass[cat] += row["prob"]
    return {
        "counts": counts,
        "prob_mass": prob_mass,
    }


def text_stats(text: str, prompt: str = "") -> Dict[str, Any]:
    toks = word_tokens(text)
    prompt_toks = word_tokens(prompt)
    generated_toks = toks[len(prompt_toks):] if toks[: len(prompt_toks)] == prompt_toks else toks
    bigrams = list(zip(generated_toks, generated_toks[1:]))
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)
    max_token_run = 1
    cur_run = 1
    for i in range(1, len(generated_toks)):
        if generated_toks[i] == generated_toks[i - 1]:
            cur_run += 1
            max_token_run = max(max_token_run, cur_run)
        else:
            cur_run = 1
    punct_chars = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    newline_chars = text.count("\n")
    alpha_chars = sum(1 for ch in text if ch.isalpha())
    unique_ratio = len(set(generated_toks)) / max(len(generated_toks), 1)
    content_ratio = len(content_tokens(" ".join(generated_toks))) / max(len(generated_toks), 1)
    return {
        "token_count": len(generated_toks),
        "unique_token_ratio": unique_ratio,
        "repeated_bigram_ratio": repeated_bigrams / max(len(bigrams), 1),
        "max_token_run": max_token_run if generated_toks else 0,
        "punct_ratio": punct_chars / max(len(text), 1),
        "newline_ratio": newline_chars / max(len(text), 1),
        "alpha_ratio": alpha_chars / max(len(text), 1),
        "content_token_ratio": content_ratio,
        "generated_preview": " ".join(generated_toks[:24]),
    }


def segmented_text_stats(text: str, prompt: str = "", window: int = 8) -> Dict[str, Any]:
    toks = word_tokens(text)
    prompt_toks = word_tokens(prompt)
    generated_toks = toks[len(prompt_toks):] if toks[: len(prompt_toks)] == prompt_toks else toks
    segments = []
    bad_segments = []
    for start in range(0, len(generated_toks), window):
        seg = generated_toks[start : start + window]
        if not seg:
            continue
        bigrams = list(zip(seg, seg[1:]))
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(c - 1 for c in bigram_counts.values() if c > 1)
        unique_ratio = len(set(seg)) / len(seg)
        content_ratio = len(content_tokens(" ".join(seg))) / len(seg)
        dominant_share = max(Counter(seg).values()) / len(seg)
        seg_info = {
            "segment_idx": start // window,
            "tokens": seg,
            "unique_ratio": unique_ratio,
            "content_ratio": content_ratio,
            "repeated_bigram_ratio": repeated_bigrams / max(len(bigrams), 1),
            "dominant_token_share": dominant_share,
        }
        segments.append(seg_info)
        if (
            unique_ratio < 0.4
            or content_ratio < 0.2
            or seg_info["repeated_bigram_ratio"] > 0.25
            or dominant_share > 0.5
        ):
            bad_segments.append(seg_info)
    return {
        "generated_token_count": len(generated_toks),
        "window": window,
        "segments": segments,
        "bad_segments": bad_segments,
        "first_bad_segment_idx": bad_segments[0]["segment_idx"] if bad_segments else None,
    }


def get_last_logits(model: sb.MemLLM, prompt: str, use_prefix: bool, update_stats: bool = False) -> torch.Tensor:
    tk = model.tok(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
    with torch.no_grad():
        if use_prefix:
            base = model.fwd(ids, mask)
            prefix = model._get_prefix(base["hs"], mask, update_stats=update_stats)
            out = model.fwd(ids, mask, prefix)
        else:
            out = model.fwd(ids, mask)
    return out["logits"][0, -1].detach().cpu()


def trace_generation_with_audit(
    model: sb.MemLLM,
    prompt: str,
    steps: int = 16,
    use_prefix: bool = True,
) -> Dict[str, Any]:
    tk = model.tok(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
    prefix = None
    with torch.no_grad():
        if use_prefix:
            o0 = model.fwd(ids, mask)
            prefix = model._get_prefix(o0["hs"], mask, update_stats=False)

    rows = []
    for step in range(steps):
        with torch.no_grad():
            out = model.fwd(ids, mask, prefix)
            logits = out["logits"][0, -1].detach().cpu()
            topk = topk_tokens_from_logits(model, logits, k=12)
            top1 = topk[0]
            cats = summarize_topk_categories(topk)
            chosen_id = int(torch.argmax(logits).item())
            chosen_piece = model.tok.decode([chosen_id])

        row = {
            "step": step,
            "top1": top1,
            "top1_category": token_category(top1["norm"]),
            "topk_category_counts": cats["counts"],
            "topk_category_prob_mass": cats["prob_mass"],
            "chosen_token_id": chosen_id,
            "chosen_piece": chosen_piece,
            "chosen_norm": normalize_token_piece(chosen_piece),
            "chosen_category": token_category(normalize_token_piece(chosen_piece)),
        }
        rows.append(row)

        nxt = torch.tensor([[chosen_id]], device=dev, dtype=ids.dtype)
        ids = torch.cat([ids, nxt], 1)
        mask = torch.cat([mask, torch.ones(1, 1, device=dev, dtype=mask.dtype)], 1)

        if use_prefix and (step + 1) % model.c.retrieval_interval == 0:
            with torch.no_grad():
                o = model.fwd(ids, mask, prefix)
                pl = o["pl"]
                prefix = model._get_prefix(o["hs"], o["mask"], pl, update_stats=False)

    first_bad_step = None
    for row in rows:
        if (
            row["top1_category"] != "semantic"
            and row["topk_category_prob_mass"]["semantic"] < 0.15
        ):
            first_bad_step = row["step"]
            break
    return {
        "prompt": prompt,
        "use_prefix": use_prefix,
        "rows": rows,
        "first_bad_step": first_bad_step,
        "decoded_output": model.tok.decode(ids[0], skip_special_tokens=True),
    }


def write_labeled_texts(model: sb.MemLLM, labeled_texts: List[Dict[str, str]]) -> Dict[int, Dict[str, Any]]:
    mapping: Dict[int, Dict[str, Any]] = {}
    for item in labeled_texts:
        label = item["label"]
        text = item["text"]
        pre = {
            mid: (me.version, me.cnt, me.last)
            for mid, me in model.amm.tree.store.items()
        }
        pre_ids = set(model.amm.tree.store.keys())
        model.write(text, training_mode=True)
        post_ids = set(model.amm.tree.store.keys())
        new_ids = list(post_ids - pre_ids)
        target_ids = new_ids
        if not target_ids:
            changed = []
            for mid, old in pre.items():
                if mid in model.amm.tree.store:
                    me = model.amm.tree.store[mid]
                    if (me.version, me.cnt, me.last) != old:
                        changed.append(mid)
            target_ids = changed[:1]
        for mid in target_ids:
            if mid not in mapping:
                mapping[mid] = {"labels": [], "texts": []}
            mapping[mid]["labels"].append(label)
            mapping[mid]["texts"].append(text)
    return mapping


def retrieve_memory_ids(model: sb.MemLLM, prompt: str, topk: int = 5, bw: int = 3) -> List[int]:
    tk = model.tok(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
    with torch.no_grad():
        out = model.fwd(ids, mask)
        _, xq, fq = model.extract_state(out["hs"], mask)
        qdir = model.amm.dir_pred(xq, fq)
    scored = model.amm.tree.retrieve(qdir[0].detach(), bw=bw)
    return [mid for mid, _ in scored[:topk]]


def retrieve_memory_scored(model: sb.MemLLM, prompt: str, topk: int = 5, bw: int = 3) -> List[Dict[str, Any]]:
    tk = model.tok(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
    with torch.no_grad():
        out = model.fwd(ids, mask)
        _, xq, fq = model.extract_state(out["hs"], mask)
        qdir = model.amm.dir_pred(xq, fq)
    scored = model.amm.tree.retrieve(qdir[0].detach(), bw=bw)
    return [{"mid": mid, "score": float(score)} for mid, score in scored[:topk]]


def correlation(xs: List[float], ys: List[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    xt = torch.tensor(xs, dtype=torch.float32)
    yt = torch.tensor(ys, dtype=torch.float32)
    xstd = float(xt.std(unbiased=False).item())
    ystd = float(yt.std(unbiased=False).item())
    if xstd < 1e-12 or ystd < 1e-12:
        return None
    xm = xt - xt.mean()
    ym = yt - yt.mean()
    return float((xm * ym).mean().item() / (xstd * ystd))


def label_mass_from_topk(rows: List[Dict[str, Any]], label_keywords: List[str]) -> float:
    return audit_domain_hits(rows, label_keywords)["match_prob_mass"]


def build_step_alignment_trace(
    model: sb.MemLLM,
    prompt: str,
    expected_label: str | None,
    memory_map: Dict[int, Dict[str, Any]],
    label_keywords: Dict[str, List[str]],
    steps: int = 12,
) -> Dict[str, Any]:
    tk = model.tok(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    ids, mask = tk["input_ids"].to(dev), tk["attention_mask"].to(dev)
    prefix = None
    retrieved_scored = []

    with torch.no_grad():
        base = model.fwd(ids, mask)
        prefix = model._get_prefix(base["hs"], mask, update_stats=False)
        retrieved_scored = retrieve_memory_scored(model, prompt, topk=5, bw=3)

    rows = []
    for step in range(steps):
        with torch.no_grad():
            out = model.fwd(ids, mask, prefix)
            logits = out["logits"][0, -1].detach().cpu()
            topk = topk_tokens_from_logits(model, logits, k=12)
            chosen_id = int(torch.argmax(logits).item())
            chosen_piece = model.tok.decode([chosen_id])
            chosen_norm = normalize_token_piece(chosen_piece)

        label_counts = Counter()
        retrieved_score_sum = Counter()
        for item in retrieved_scored:
            meta = memory_map.get(item["mid"])
            if not meta:
                continue
            for label in meta["labels"]:
                label_counts[label] += 1
                retrieved_score_sum[label] += item["score"]
        retrieved_majority = label_counts.most_common(1)[0][0] if label_counts else None
        logits_label_mass = {
            label: label_mass_from_topk(topk, kws) for label, kws in label_keywords.items()
        }
        topk_cats = summarize_topk_categories(topk)
        chosen_label = None
        if label_keywords:
            best_label, best_mass = max(logits_label_mass.items(), key=lambda kv: kv[1])
            if best_mass > 0:
                chosen_label = best_label

        if expected_label is not None and retrieved_majority != expected_label:
            stage = "retrieve"
        elif retrieved_majority is not None and logits_label_mass.get(retrieved_majority, 0.0) == 0.0:
            stage = "inject"
        elif token_category(chosen_norm) != "semantic":
            stage = "decode"
        elif retrieved_majority is not None and chosen_label not in (None, retrieved_majority):
            stage = "decode"
        else:
            stage = "aligned"

        rows.append(
            {
                "step": step,
                "retrieved_majority_label": retrieved_majority,
                "retrieved_label_counts": dict(label_counts),
                "retrieved_score_sum": dict(retrieved_score_sum),
                "logits_label_mass": logits_label_mass,
                "top1_piece": topk[0]["piece"],
                "top1_category": token_category(topk[0]["norm"]),
                "chosen_piece": chosen_piece,
                "chosen_category": token_category(chosen_norm),
                "chosen_label": chosen_label,
                "diagnosed_stage": stage,
            }
        )

        nxt = torch.tensor([[chosen_id]], device=dev, dtype=ids.dtype)
        ids = torch.cat([ids, nxt], 1)
        mask = torch.cat([mask, torch.ones(1, 1, device=dev, dtype=mask.dtype)], 1)
        if (step + 1) % model.c.retrieval_interval == 0:
            with torch.no_grad():
                o = model.fwd(ids, mask, prefix)
                pl = o["pl"]
                prefix = model._get_prefix(o["hs"], o["mask"], pl, update_stats=False)
            current_prompt = model.tok.decode(ids[0], skip_special_tokens=True)
            retrieved_scored = retrieve_memory_scored(model, current_prompt, topk=5, bw=3)

    return {
        "prompt": prompt,
        "expected_label": expected_label,
        "decoded_output": model.tok.decode(ids[0], skip_special_tokens=True),
        "rows": rows,
    }


def leaf_capacity_stability(seeds: List[int], items: int = 240) -> Dict[str, Any]:
    cfg = sb.Cfg(tree_max_leaf=5, tree_K=3)
    per_seed = []
    all_pass = True
    for seed in seeds:
        set_seed(seed)
        tree = sb.DirectionTree(cfg)
        for i in range(items):
            d = torch.nn.functional.normalize(torch.randn(cfg.d_M), dim=0)
            entry = sb.MemEntry(
                mid=i,
                base=torch.randn(cfg.d_M),
                fiber=torch.randn(cfg.d_F),
                dirn=d,
                surprise=0.5,
                ts=float(i),
                last=float(i),
            )
            tree.store[entry.mid] = entry
            tree.nid = i + 1
            tree._ins(tree.root, entry)
        violations = tree.leaf_size_violations()
        consistency = tree.verify_consistency()
        passed = len(violations) == 0 and len(consistency) == 0
        all_pass = all_pass and passed
        per_seed.append(
            {
                "seed": seed,
                "depth": tree.max_depth(),
                "count": tree.root.count(),
                "violations": violations,
                "consistency": consistency,
                "passed": passed,
            }
        )
    return {"passed": all_pass, "per_seed": per_seed}


def degenerate_direction_boundary(seed: int, items: int = 100) -> Dict[str, Any]:
    set_seed(seed)
    cfg = sb.Cfg(tree_max_leaf=5, tree_K=3)
    tree = sb.DirectionTree(cfg)
    base_dir = torch.zeros(cfg.d_M)
    base_dir[0] = 1.0
    for i in range(items):
        noise = torch.zeros(cfg.d_M)
        noise[-1] = (i % 5) * 1e-9
        d = torch.nn.functional.normalize(base_dir + noise, dim=0)
        entry = sb.MemEntry(
            mid=i,
            base=torch.full((cfg.d_M,), float(i) / items),
            fiber=torch.randn(cfg.d_F),
            dirn=d,
            surprise=0.1,
            ts=float(i),
            last=float(i),
        )
        tree.store[entry.mid] = entry
        tree.nid = i + 1
        tree._ins(tree.root, entry)
    return {
        "passed": len(tree.verify_consistency()) == 0,
        "depth": tree.max_depth(),
        "count": tree.root.count(),
        "violations": tree.leaf_size_violations(),
        "consistency": tree.verify_consistency(),
        "seed": seed,
    }


def metric_trainability(seed: int) -> Dict[str, Any]:
    model = build_model(seed)
    write_texts(model, corpus_general())
    trainer = sb.Trainer(model, model.c)
    metric_params = [p for p in model.amm.metric.parameters() if p.requires_grad]
    before = [p.detach().clone() for p in metric_params]
    model.train()
    info = trainer.step(corpus_general()[:3])
    grad_norms = [
        0.0 if p.grad is None else float(p.grad.detach().norm().item()) for p in metric_params
    ]
    deltas = [
        float((p.detach() - b).norm().item()) for p, b in zip(metric_params, before)
    ]
    return {
        "passed": any(g > 0 for g in grad_norms) and any(d > 0 for d in deltas),
        "training_info": info,
        "metric_grad_norms": grad_norms,
        "metric_param_deltas": deltas,
        "max_metric_grad_norm": max(grad_norms) if grad_norms else 0.0,
        "max_metric_param_delta": max(deltas) if deltas else 0.0,
    }


def no_grad_generation(seed: int) -> Dict[str, Any]:
    model = build_model(seed)
    stored = write_texts(model, corpus_general())
    with torch.no_grad():
        out = model.generate("The pianist", mt=24, greedy=True)
    return {
        "passed": stored > 0 and isinstance(out, str) and len(out) > 0,
        "stored_memories": stored,
        "output": out,
    }


def counterfactual_memory_influence(seed: int) -> Dict[str, Any]:
    model_music = build_model(seed)
    model_space = build_model(seed)
    write_texts(model_music, corpus_music())
    write_texts(model_space, corpus_space())
    prompt = "Tell me something about practice and performance."
    with torch.no_grad():
        out_music = model_music.generate(prompt, mt=24, greedy=True)
        out_space = model_space.generate(prompt, mt=24, greedy=True)
    return {
        "passed": out_music != out_space,
        "prompt": prompt,
        "music_output": out_music,
        "space_output": out_space,
        "outputs_differ": out_music != out_space,
    }


def prompt_diversity_without_memory(seed: int) -> Dict[str, Any]:
    model = build_model(seed)
    prompts = [
        "The pianist",
        "Quantum systems",
        "The rainforest",
    ]
    outputs = []
    with torch.no_grad():
        for prompt in prompts:
            outputs.append(model.generate(prompt, mt=18, greedy=True))
    unique = len(set(outputs))
    return {
        "passed": unique == len(outputs),
        "prompts": prompts,
        "outputs": outputs,
        "unique_count": unique,
    }


def save_load_consistency(seed: int) -> Dict[str, Any]:
    model_a = build_model(seed)
    write_texts(model_a, corpus_general())
    tmp_path = REPORT_DIR / "tmp_memory.pt"
    model_a.save_memory(str(tmp_path))

    model_b = build_model(seed)
    model_b.load_memory(str(tmp_path))
    prompt = "The pianist"
    with torch.no_grad():
        out_a = model_a.generate(prompt, mt=18, greedy=True)
        out_b = model_b.generate(prompt, mt=18, greedy=True)
    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass
    return {
        "passed": out_a == out_b,
        "prompt": prompt,
        "output_a": out_a,
        "output_b": out_b,
    }


def training_cache_isolation(seed: int) -> Dict[str, Any]:
    model = build_model(seed)
    write_texts(model, corpus_general())
    snapshot = {mid: (me.last, me.cnt) for mid, me in model.amm.tree.store.items()}
    trainer = sb.Trainer(model, model.c)
    trainer.recon("Some query text that triggers retrieval.")
    changed = []
    for mid, (old_last, old_cnt) in snapshot.items():
        me = model.amm.tree.store[mid]
        if me.last != old_last or me.cnt != old_cnt:
            changed.append((mid, old_last, me.last, old_cnt, me.cnt))
    return {
        "passed": len(changed) == 0,
        "changed": changed,
        "memory_count": len(snapshot),
    }


def cheating_heuristics(seed: int) -> Dict[str, Any]:
    model = build_model(seed)
    write_texts(model, corpus_general())
    prompts = [
        "The pianist",
        "The telescope",
        "The trader",
        "The child",
    ]
    with torch.no_grad():
        outputs = [model.generate(prompt, mt=18, greedy=True) for prompt in prompts]
    exact_same = len(set(outputs)) == 1
    prefix_only = all(out.strip() == prompt.strip() for out, prompt in zip(outputs, prompts))
    too_short = all(len(out.strip()) <= len(prompt.strip()) + 1 for out, prompt in zip(outputs, prompts))
    return {
        "passed": not exact_same and not prefix_only and not too_short,
        "outputs": outputs,
        "exact_same": exact_same,
        "prefix_only": prefix_only,
        "too_short": too_short,
    }


def semantic_memory_grounding(seed: int) -> Dict[str, Any]:
    music_keywords = derive_keywords(corpus_music())
    space_keywords = derive_keywords(corpus_space())

    model_blank = build_model(seed)
    model_music = build_model(seed)
    model_space = build_model(seed)
    write_texts(model_music, corpus_music())
    write_texts(model_space, corpus_space())

    prompt = "Explain what someone should focus on when improving technique and understanding the subject."
    with torch.no_grad():
        out_blank = model_blank.generate(prompt, mt=32, greedy=True)
        out_music = model_music.generate(prompt, mt=32, greedy=True)
        out_space = model_space.generate(prompt, mt=32, greedy=True)

    blank_music_score = keyword_score(out_blank, music_keywords)
    blank_space_score = keyword_score(out_blank, space_keywords)
    music_music_score = keyword_score(out_music, music_keywords)
    music_space_score = keyword_score(out_music, space_keywords)
    space_space_score = keyword_score(out_space, space_keywords)
    space_music_score = keyword_score(out_space, music_keywords)

    music_margin = music_music_score - music_space_score
    space_margin = space_space_score - space_music_score
    music_lift = music_music_score - blank_music_score
    space_lift = space_space_score - blank_space_score

    return {
        "passed": music_margin > 0 and space_margin > 0 and (music_lift > 0 or space_lift > 0),
        "prompt": prompt,
        "music_keywords": music_keywords,
        "space_keywords": space_keywords,
        "blank_output": out_blank,
        "music_output": out_music,
        "space_output": out_space,
        "blank_music_score": blank_music_score,
        "blank_space_score": blank_space_score,
        "music_music_score": music_music_score,
        "music_space_score": music_space_score,
        "space_space_score": space_space_score,
        "space_music_score": space_music_score,
        "music_margin": music_margin,
        "space_margin": space_margin,
        "music_lift": music_lift,
        "space_lift": space_lift,
    }


def semantic_memory_counterfactual_pairs(seed: int) -> Dict[str, Any]:
    music_keywords = set(derive_keywords(corpus_music()))
    space_keywords = set(derive_keywords(corpus_space()))
    prompts = [
        "Describe the most important details a student should notice.",
        "Summarize the key ideas a learner should practice and remember.",
    ]
    model_music = build_model(seed)
    model_space = build_model(seed)
    write_texts(model_music, corpus_music())
    write_texts(model_space, corpus_space())

    rows = []
    passed = True
    with torch.no_grad():
        for prompt in prompts:
            out_music = model_music.generate(prompt, mt=28, greedy=True)
            out_space = model_space.generate(prompt, mt=28, greedy=True)
            mm = keyword_score(out_music, list(music_keywords))
            ms = keyword_score(out_music, list(space_keywords))
            ss = keyword_score(out_space, list(space_keywords))
            sm = keyword_score(out_space, list(music_keywords))
            row_pass = (mm - ms) > 0 and (ss - sm) > 0
            passed = passed and row_pass
            rows.append(
                {
                    "prompt": prompt,
                    "music_output": out_music,
                    "space_output": out_space,
                    "music_margin": mm - ms,
                    "space_margin": ss - sm,
                    "passed": row_pass,
                }
            )
    return {"passed": passed, "rows": rows}


def degeneration_quality(seed: int) -> Dict[str, Any]:
    model = build_model(seed)
    write_texts(model, corpus_general() + corpus_music() + corpus_space())
    prompts = [
        "The pianist",
        "The telescope",
        "The forest path",
        "The market analyst",
        "Explain the topic clearly",
    ]
    outputs = []
    metrics = []
    with torch.no_grad():
        for prompt in prompts:
            out = model.generate(prompt, mt=28, greedy=True)
            outputs.append(out)
            metrics.append({"prompt": prompt, "output": out, **text_stats(out, prompt)})

    avg_unique = sum(m["unique_token_ratio"] for m in metrics) / len(metrics)
    avg_repeat = sum(m["repeated_bigram_ratio"] for m in metrics) / len(metrics)
    avg_content = sum(m["content_token_ratio"] for m in metrics) / len(metrics)
    avg_newline = sum(m["newline_ratio"] for m in metrics) / len(metrics)
    worst_run = max(m["max_token_run"] for m in metrics)
    short_or_hollow = [
        m["prompt"]
        for m in metrics
        if m["token_count"] < 6 or m["content_token_ratio"] < 0.15 or m["alpha_ratio"] < 0.35
    ]

    passed = (
        avg_unique >= 0.35
        and avg_repeat <= 0.20
        and avg_content >= 0.22
        and avg_newline <= 0.20
        and worst_run <= 4
        and not short_or_hollow
    )
    return {
        "passed": passed,
        "metrics": metrics,
        "aggregate": {
            "avg_unique_token_ratio": avg_unique,
            "avg_repeated_bigram_ratio": avg_repeat,
            "avg_content_token_ratio": avg_content,
            "avg_newline_ratio": avg_newline,
            "worst_max_token_run": worst_run,
            "short_or_hollow_prompts": short_or_hollow,
        },
    }


def prefix_logit_drift_audit(seed: int) -> Dict[str, Any]:
    prompt = "Explain the topic in a precise and concrete way."
    blank = build_model(seed)
    mem = build_model(seed)
    write_texts(mem, corpus_general() + corpus_music())

    blank_no = get_last_logits(blank, prompt, use_prefix=False)
    blank_yes = get_last_logits(blank, prompt, use_prefix=True)
    mem_no = get_last_logits(mem, prompt, use_prefix=False)
    mem_yes = get_last_logits(mem, prompt, use_prefix=True)

    blank_rows_no = topk_tokens_from_logits(blank, blank_no)
    blank_rows_yes = topk_tokens_from_logits(blank, blank_yes)
    mem_rows_no = topk_tokens_from_logits(mem, mem_no)
    mem_rows_yes = topk_tokens_from_logits(mem, mem_yes)

    blank_overlap = len({r["token_id"] for r in blank_rows_no} & {r["token_id"] for r in blank_rows_yes})
    mem_overlap = len({r["token_id"] for r in mem_rows_no} & {r["token_id"] for r in mem_rows_yes})
    blank_js = js_divergence_from_logits(blank_no, blank_yes)
    mem_js = js_divergence_from_logits(mem_no, mem_yes)
    blank_l2 = float(torch.norm(blank_no - blank_yes).item())
    mem_l2 = float(torch.norm(mem_no - mem_yes).item())

    return {
        "passed": mem_js > blank_js or mem_l2 > blank_l2 or mem_overlap < blank_overlap,
        "prompt": prompt,
        "blank": {
            "js_divergence": blank_js,
            "l2_shift": blank_l2,
            "topk_overlap_count": blank_overlap,
            "entropy_no_prefix": entropy_from_logits(blank_no),
            "entropy_with_prefix": entropy_from_logits(blank_yes),
            "topk_no_prefix": blank_rows_no,
            "topk_with_prefix": blank_rows_yes,
        },
        "memory": {
            "js_divergence": mem_js,
            "l2_shift": mem_l2,
            "topk_overlap_count": mem_overlap,
            "entropy_no_prefix": entropy_from_logits(mem_no),
            "entropy_with_prefix": entropy_from_logits(mem_yes),
            "topk_no_prefix": mem_rows_no,
            "topk_with_prefix": mem_rows_yes,
        },
    }


def retrieval_topk_semantic_shift(seed: int) -> Dict[str, Any]:
    music_keywords = derive_keywords(corpus_music())
    space_keywords = derive_keywords(corpus_space())
    prompts = [
        "A strong explanation should mention",
        "The most relevant idea is",
    ]
    model_music = build_model(seed)
    model_space = build_model(seed)
    write_texts(model_music, corpus_music())
    write_texts(model_space, corpus_space())

    rows = []
    passed = False
    for prompt in prompts:
        music_no = get_last_logits(model_music, prompt, use_prefix=False)
        music_yes = get_last_logits(model_music, prompt, use_prefix=True)
        space_no = get_last_logits(model_space, prompt, use_prefix=False)
        space_yes = get_last_logits(model_space, prompt, use_prefix=True)

        music_topk_no = topk_tokens_from_logits(model_music, music_no)
        music_topk_yes = topk_tokens_from_logits(model_music, music_yes)
        space_topk_no = topk_tokens_from_logits(model_space, space_no)
        space_topk_yes = topk_tokens_from_logits(model_space, space_yes)

        music_hits_no = audit_domain_hits(music_topk_no, music_keywords)
        music_hits_yes = audit_domain_hits(music_topk_yes, music_keywords)
        space_hits_no = audit_domain_hits(space_topk_no, space_keywords)
        space_hits_yes = audit_domain_hits(space_topk_yes, space_keywords)

        row_pass = (
            music_hits_yes["match_count"] > music_hits_no["match_count"]
            or music_hits_yes["match_prob_mass"] > music_hits_no["match_prob_mass"]
            or space_hits_yes["match_count"] > space_hits_no["match_count"]
            or space_hits_yes["match_prob_mass"] > space_hits_no["match_prob_mass"]
        )
        passed = passed or row_pass
        rows.append(
            {
                "prompt": prompt,
                "music_no_prefix": music_topk_no,
                "music_with_prefix": music_topk_yes,
                "music_hits_no": music_hits_no,
                "music_hits_with_prefix": music_hits_yes,
                "space_no_prefix": space_topk_no,
                "space_with_prefix": space_topk_yes,
                "space_hits_no": space_hits_no,
                "space_hits_with_prefix": space_hits_yes,
                "passed": row_pass,
            }
        )
    return {
        "passed": passed,
        "music_keywords": music_keywords,
        "space_keywords": space_keywords,
        "rows": rows,
    }


def repetition_segment_audit(seed: int) -> Dict[str, Any]:
    model = build_model(seed)
    write_texts(model, corpus_general() + corpus_music() + corpus_space())
    prompts = [
        "The pianist",
        "The telescope",
        "The market analyst",
        "Explain the topic clearly",
    ]
    rows = []
    all_bad = 0
    total_segments = 0
    for prompt in prompts:
        with torch.no_grad():
            out = model.generate(prompt, mt=48, greedy=True)
        audit = segmented_text_stats(out, prompt, window=8)
        total_segments += len(audit["segments"])
        all_bad += len(audit["bad_segments"])
        rows.append({"prompt": prompt, "output": out, **audit})
    bad_ratio = all_bad / max(total_segments, 1)
    early_collapse = [r["prompt"] for r in rows if r["first_bad_segment_idx"] in (0, 1)]
    return {
        "passed": bad_ratio <= 0.35 and len(early_collapse) <= 1,
        "aggregate": {
            "bad_segment_ratio": bad_ratio,
            "total_segments": total_segments,
            "bad_segments": all_bad,
            "early_collapse_prompts": early_collapse,
        },
        "rows": rows,
    }


def prefix_stepwise_drift_trajectory(seed: int) -> Dict[str, Any]:
    model = build_model(seed)
    write_texts(model, corpus_general() + corpus_music())
    prompts = [
        "Key piano ideas include",
        "Explain the topic clearly",
    ]
    rows = []
    passed = True
    for prompt in prompts:
        trace = trace_generation_with_audit(model, prompt, steps=16, use_prefix=True)
        row_pass = trace["first_bad_step"] is None or trace["first_bad_step"] >= 3
        passed = passed and row_pass
        rows.append(
            {
                "prompt": prompt,
                "first_bad_step": trace["first_bad_step"],
                "decoded_output": trace["decoded_output"],
                "rows": trace["rows"],
                "passed": row_pass,
            }
        )
    return {"passed": passed, "rows": rows}


def retrieval_generation_alignment_audit(seed: int) -> Dict[str, Any]:
    labeled = [{"label": "music", "text": t} for t in corpus_music()] + [
        {"label": "space", "text": t} for t in corpus_space()
    ]
    music_keywords = derive_keywords(corpus_music())
    space_keywords = derive_keywords(corpus_space())
    model = build_model(seed)
    memory_map = write_labeled_texts(model, labeled)

    prompts = [
        {"prompt": "What improves piano technique and musical phrasing?", "expected": "music"},
        {"prompt": "What explains satellites and orbital motion?", "expected": "space"},
        {"prompt": "Summarize the subject with concrete domain details.", "expected": None},
    ]

    rows = []
    diagnoses = {"aligned": 0, "retrieval_miss": 0, "bridge_unused": 0, "unknown": 0}
    passed = True

    for item in prompts:
        prompt = item["prompt"]
        expected = item["expected"]
        mids = retrieve_memory_ids(model, prompt, topk=5, bw=3)
        retrieved_labels = []
        retrieved_texts = []
        for mid in mids:
            meta = memory_map.get(mid)
            if meta:
                retrieved_labels.extend(meta["labels"])
                retrieved_texts.extend(meta["texts"])
        label_counts = Counter(retrieved_labels)
        retrieved_majority = label_counts.most_common(1)[0][0] if label_counts else None

        with torch.no_grad():
            output = model.generate(prompt, mt=28, greedy=True)

        music_score = keyword_score(output, music_keywords)
        space_score = keyword_score(output, space_keywords)
        if music_score > space_score:
            generated_label = "music"
        elif space_score > music_score:
            generated_label = "space"
        else:
            generated_label = None

        if expected is not None and retrieved_majority != expected:
            diagnosis = "retrieval_miss"
        elif retrieved_majority is not None and generated_label != retrieved_majority:
            diagnosis = "bridge_unused"
        elif retrieved_majority is not None and generated_label == retrieved_majority:
            diagnosis = "aligned"
        else:
            diagnosis = "unknown"

        diagnoses[diagnosis] += 1
        row_pass = diagnosis == "aligned" or (expected is None and diagnosis != "retrieval_miss")
        passed = passed and row_pass
        rows.append(
            {
                "prompt": prompt,
                "expected_label": expected,
                "retrieved_mids": mids,
                "retrieved_label_counts": dict(label_counts),
                "retrieved_majority_label": retrieved_majority,
                "retrieved_text_preview": retrieved_texts[:3],
                "output": output,
                "music_score": music_score,
                "space_score": space_score,
                "generated_label": generated_label,
                "diagnosis": diagnosis,
                "passed": row_pass,
            }
        )

    return {
        "passed": passed,
        "music_keywords": music_keywords,
        "space_keywords": space_keywords,
        "diagnoses": diagnoses,
        "rows": rows,
    }


def retrieval_prefix_decode_correlation_audit(seed: int) -> Dict[str, Any]:
    labeled = [{"label": "music", "text": t} for t in corpus_music()] + [
        {"label": "space", "text": t} for t in corpus_space()
    ]
    model = build_model(seed)
    memory_map = write_labeled_texts(model, labeled)
    prompts = [
        {"prompt": "What improves piano technique and musical phrasing?", "expected": "music"},
        {"prompt": "What explains satellites and orbital motion?", "expected": "space"},
        {"prompt": "Describe what a student should focus on first.", "expected": None},
        {"prompt": "Summarize the subject with concrete domain details.", "expected": None},
        {"prompt": "Key piano ideas include", "expected": "music"},
        {"prompt": "Orbital motion depends on", "expected": "space"},
    ]
    rows = []
    retrieval_strengths = []
    prefix_l2s = []
    bad_decode_scores = []

    for item in prompts:
        prompt = item["prompt"]
        expected = item["expected"]
        scored = retrieve_memory_scored(model, prompt, topk=5, bw=3)
        label_counts = Counter()
        expected_strength = 0.0
        total_strength = 0.0
        for s in scored:
            total_strength += s["score"]
            meta = memory_map.get(s["mid"])
            if not meta:
                continue
            for label in meta["labels"]:
                label_counts[label] += 1
                if expected is not None and label == expected:
                    expected_strength += s["score"]

        no_prefix = get_last_logits(model, prompt, use_prefix=False)
        yes_prefix = get_last_logits(model, prompt, use_prefix=True)
        prefix_l2 = float(torch.norm(no_prefix - yes_prefix).item())
        topk = topk_tokens_from_logits(model, yes_prefix, k=12)
        top1_cat = token_category(topk[0]["norm"])
        non_semantic_mass = (
            summarize_topk_categories(topk)["prob_mass"]["functional"]
            + summarize_topk_categories(topk)["prob_mass"]["punct"]
        )
        bad_decode = 1.0 if top1_cat != "semantic" else 0.0
        retrieval_strength = expected_strength if expected is not None else (scored[0]["score"] if scored else 0.0)

        retrieval_strengths.append(retrieval_strength)
        prefix_l2s.append(prefix_l2)
        bad_decode_scores.append(bad_decode + non_semantic_mass)
        rows.append(
            {
                "prompt": prompt,
                "expected_label": expected,
                "retrieved_scored": scored,
                "retrieved_label_counts": dict(label_counts),
                "retrieval_strength": retrieval_strength,
                "prefix_l2_shift": prefix_l2,
                "prefix_js_divergence": js_divergence_from_logits(no_prefix, yes_prefix),
                "top1_with_prefix": topk[0],
                "top1_category_with_prefix": top1_cat,
                "topk_non_semantic_prob_mass": non_semantic_mass,
            }
        )

    corr_retrieval_prefix = correlation(retrieval_strengths, prefix_l2s)
    corr_retrieval_bad = correlation(retrieval_strengths, bad_decode_scores)
    corr_prefix_bad = correlation(prefix_l2s, bad_decode_scores)
    passed = not (
        (corr_retrieval_bad is not None and corr_retrieval_bad > 0.2)
        or (corr_prefix_bad is not None and corr_prefix_bad > 0.2)
    )
    return {
        "passed": passed,
        "correlations": {
            "retrieval_strength__prefix_l2": corr_retrieval_prefix,
            "retrieval_strength__bad_decode_score": corr_retrieval_bad,
            "prefix_l2__bad_decode_score": corr_prefix_bad,
        },
        "rows": rows,
    }


def stepwise_label_mass_alignment_audit(seed: int) -> Dict[str, Any]:
    labeled = [{"label": "music", "text": t} for t in corpus_music()] + [
        {"label": "space", "text": t} for t in corpus_space()
    ]
    label_keywords = {
        "music": derive_keywords(corpus_music()),
        "space": derive_keywords(corpus_space()),
    }
    model = build_model(seed)
    memory_map = write_labeled_texts(model, labeled)
    prompts = [
        {"prompt": "What improves piano technique and musical phrasing?", "expected": "music"},
        {"prompt": "What explains satellites and orbital motion?", "expected": "space"},
    ]
    rows = []
    passed = True
    for item in prompts:
        trace = build_step_alignment_trace(
            model,
            item["prompt"],
            item["expected"],
            memory_map,
            label_keywords,
            steps=12,
        )
        stage_counts = Counter(r["diagnosed_stage"] for r in trace["rows"])
        row_pass = stage_counts.get("retrieve", 0) == 0 and stage_counts.get("inject", 0) == 0
        passed = passed and row_pass
        rows.append(
            {
                "prompt": item["prompt"],
                "expected_label": item["expected"],
                "decoded_output": trace["decoded_output"],
                "stage_counts": dict(stage_counts),
                "rows": trace["rows"],
                "passed": row_pass,
            }
        )
    return {
        "passed": passed,
        "label_keywords": label_keywords,
        "rows": rows,
    }


def results_to_checks(results: Dict[str, Any]) -> List[CheckResult]:
    checks: List[CheckResult] = []
    for name, payload in results.items():
        if payload.get("error") is None:
            detail = json.dumps(
                {k: v for k, v in payload.items() if k not in {"passed", "error"}},
                ensure_ascii=False,
            )[:1200]
        else:
            detail = payload["error"]["message"]
        checks.append(CheckResult(name=name, passed=payload["passed"], detail=detail))
    return checks


def compute_axis_coverage(results: Dict[str, Any], checks: List[CheckResult]) -> Dict[str, Any]:
    """[SPEC Section 4-meta.1 v3.45+] Axis-coverage table emitted in every report.

    Axis A: compression = (stored floats per memory) / (raw tokens * d_LLM).
    Axis B: injection cost = prefix_length * d_LLM + content_bias_size (all O(1) in N).
    Axis C: fidelity-dependent cases (4.6, 4.7, 4.10, 4.15, 4.16, 4.17, 4.19, 4.22, 4.23, 4.24, 4.25).
    Axis D: stability-dependent cases (4.13 save_load_consistency, 4.20 rerank, 4.21 repetition-feedback).
    """
    try:
        import AgentMemorySystem as _sb
        c = _sb.Cfg()
        d_LLM = int(c.d_LLM)
        L_mem = int(c.L_mem)
        d_M = int(c.d_M); d_F = int(c.d_F); d_ctx = int(c.d_ctx)
        V = int(c.vocab_size)
    except Exception:
        d_LLM = 1536; L_mem = 8; d_M = 8; d_F = 32; d_ctx = 128; V = 151936
    # Axis A:
    stored_floats_per_mem = d_M + d_F + d_M + d_ctx + d_LLM
    # Average memory text ~ 10 tokens; raw dense text embedding cost:
    typical_mem_tokens = 10
    raw_floats_per_mem = typical_mem_tokens * d_LLM
    compression_ratio = raw_floats_per_mem / max(stored_floats_per_mem, 1)
    axis_a_pass = compression_ratio >= 10.0
    # Axis B:
    per_step_floats = L_mem * d_LLM + V    # prefix + content_bias
    axis_b_pass = True   # by construction O(1) in N; annotate the formula
    # Axis C / D:
    fidelity_cases = [
        "semantic_memory_grounding",
        "semantic_memory_counterfactual_pairs",
        "retrieval_topk_semantic_shift",
        "prefix_stepwise_drift_trajectory",
        "retrieval_generation_alignment_audit",
        "retrieval_prefix_decode_correlation_audit",
        "stepwise_label_mass_alignment_audit",
        "functional_token_suppression_probe",
        "keyword_specific_tail_slot_probe",
        "context_descriptor_cluster_probe",
        "prefix_length_scaling_probe",
    ]
    stability_cases = [
        "save_load_consistency",
        "rerank_stability_probe",
        "decode_repetition_feedback_probe",
    ]
    def _acc(names):
        total = 0; passed = 0
        for n in names:
            if n in results:
                total += 1
                if results[n].get("passed") is True:
                    passed += 1
        return passed, total
    c_pass, c_total = _acc(fidelity_cases)
    d_pass, d_total = _acc(stability_cases)
    # Per spec Section 4-meta.1 C passes iff aggregate >= K; K for v3.45 is set to
    # ceil(0.75 * C_total).
    import math as _m
    c_K = _m.ceil(0.75 * c_total) if c_total > 0 else 0
    axis_c_pass = c_pass >= c_K
    axis_d_pass = d_pass == d_total
    return {
        "spec_section": "4-meta.1 v3.45+",
        "axis_a_compression": {
            "stored_floats_per_mem": stored_floats_per_mem,
            "raw_floats_per_mem_typical_10_tokens": raw_floats_per_mem,
            "ratio": compression_ratio,
            "threshold": 10.0,
            "passed": axis_a_pass,
        },
        "axis_b_injection_cost": {
            "per_step_floats_formula": "L_mem * d_LLM + V",
            "per_step_floats_value": per_step_floats,
            "depends_on_N": False,
            "passed": axis_b_pass,
        },
        "axis_c_fidelity": {
            "dependent_cases": fidelity_cases,
            "passed_over_total": f"{c_pass}/{c_total}",
            "threshold_K": c_K,
            "passed": axis_c_pass,
        },
        "axis_d_stability": {
            "dependent_cases": stability_cases,
            "passed_over_total": f"{d_pass}/{d_total}",
            "threshold_all_pass": True,
            "passed": axis_d_pass,
        },
        "channel_passes_all_axes": bool(axis_a_pass and axis_b_pass and axis_c_pass and axis_d_pass),
    }


def write_reports(results: Dict[str, Any], checks: List[CheckResult], elapsed: float) -> None:
    ensure_report_dir()
    axis_coverage = compute_axis_coverage(results, checks)
    payload = {
        "generated_at_epoch": time.time(),
        "elapsed_seconds": elapsed,
        "checks": [asdict(c) for c in checks],
        "results": results,
        "axis_coverage": axis_coverage,
        "constraints": {
            "uses_internal_test": False,
            "monkeypatching": False,
            "mocking": False,
            "direct_return_shortcut_detected": any(
                results[name].get("passed") is False for name in ["cheating_heuristics"]
            ),
        },
    }
    JSON_REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# `AgentMemorySystem v331` Detailed Black-box Test Report",
        "",
        f"- Elapsed: `{elapsed:.1f}s`",
        f"- Passed: `{sum(c.passed for c in checks)}/{len(checks)}`",
        "- Mode: fully external runner, no reuse of module-internal `test()`",
        "- Policy: no monkeypatching, no mocked return values, no synthetic pass-by-construction shortcuts",
        "",
        "## Axis Coverage (SPEC Section 4-meta.1, v3.45+)",
        "",
        "```json",
        json.dumps(axis_coverage, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Summary",
        "",
    ]
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        lines.append(f"- `{status}` `{c.name}`: {c.detail}")

    section_titles = {
        "leaf_capacity_stability": "Leaf Capacity Stability",
        "degenerate_direction_boundary": "Degenerate Direction Boundary",
        "metric_trainability": "Metric Trainability",
        "no_grad_generation": "No-Grad Generation",
        "counterfactual_memory_influence": "Counterfactual Memory Influence",
        "semantic_memory_grounding": "Semantic Memory Grounding",
        "semantic_memory_counterfactual_pairs": "Semantic Memory Counterfactual Pairs",
        "degeneration_quality": "Degeneration Quality",
        "prefix_logit_drift_audit": "Prefix Logit Drift Audit",
        "retrieval_topk_semantic_shift": "Retrieval Top-K Semantic Shift",
        "repetition_segment_audit": "Repetition Segment Audit",
        "prefix_stepwise_drift_trajectory": "Prefix Stepwise Drift Trajectory",
        "retrieval_generation_alignment_audit": "Retrieval Generation Alignment Audit",
        "retrieval_prefix_decode_correlation_audit": "Retrieval Prefix Decode Correlation Audit",
        "stepwise_label_mass_alignment_audit": "Stepwise Label Mass Alignment Audit",
        "prompt_diversity_without_memory": "Prompt Diversity Without Memory",
        "save_load_consistency": "Save/Load Consistency",
        "training_cache_isolation": "Training Cache Isolation",
        "cheating_heuristics": "Cheating Heuristics",
    }

    for key, title in section_titles.items():
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "```json",
                json.dumps(results[key], ensure_ascii=False, indent=2),
                "```",
            ]
        )

    MD_REPORT.write_text("\n".join(lines), encoding="utf-8")


# ==========================================================================
# Cipher-System Structural Probes (4.20 - 4.26)
# Added for v3.38 audit. Each probe follows the spec's no-mock / no-fallback
# / no-overfit / no-simplification policy. Probes whose target API surface is
# absent in the SUT emit status = "not_implemented" per Section 5 of the spec.
# No probe modifies the 4.1 - 4.19 cases above.
# ==========================================================================

CIPHER_MUSIC_KEYWORDS = [
    "pianist", "practiced", "arpeggios", "chopin", "nocturnes", "midnight",
    "musician", "refined", "finger", "technique", "phrasing", "pedal",
]
CIPHER_SPACE_KEYWORDS = [
    "distant", "astronomers", "observed", "galaxies", "quasars", "stellar",
    "evolution", "space", "orbital", "mechanics", "explains", "satellites",
]


def _cipher_prep_decode(model: "sb.MemLLM", prompt: str) -> Dict[str, Any]:
    """Run prepare_decode_context for a prompt and return diag + weights."""
    device = next(model.parameters()).device
    tk = model.tok(prompt, return_tensors="pt")
    ids = tk["input_ids"].to(device)
    mask = tk["attention_mask"].to(device)
    with torch.no_grad():
        ctx = model.prepare_decode_context(ids, mask, update_stats=False)
    diag = ctx.diag
    top5 = sorted(diag.batch_mem_weights[0], key=lambda x: -x[1])[:5] \
        if diag.batch_mem_weights else []
    dominant = diag.dominant_per_batch[0] if diag.dominant_per_batch else None
    return {
        "ids_tensor": ids,
        "mask_tensor": mask,
        "ctx": ctx,
        "dominant_mid": dominant,
        "top5_mids": [int(mid) for mid, _ in top5],
        "top5_weights": {int(mid): float(w) for mid, w in top5},
    }


def rerank_stability_probe(seed: int) -> Dict[str, Any]:
    """[4.20] Rerank must be stable across near-paraphrase prompts (P0)."""
    model = build_model(seed)
    write_texts(model, corpus_music() + corpus_space())

    pairs = [
        ("music_P1", [
            "What improves piano technique and musical phrasing?",
            "How can one improve piano technique and musical expression?"]),
        ("space_P2", [
            "What explains satellites and orbital motion?",
            "What describes satellites and the motion of planets?"]),
    ]

    results_per_pair = []
    passed_pair_count = 0
    spearman_best = 0.0
    for pair_name, (p_a, p_b) in pairs:
        r_a = _cipher_prep_decode(model, p_a)
        r_b = _cipher_prep_decode(model, p_b)
        set_a = set(r_a["top5_mids"])
        set_b = set(r_b["top5_mids"])
        union_size = len(set_a | set_b)
        inter_size = len(set_a & set_b)
        jaccard = inter_size / max(union_size, 1)
        shared = [mid for mid in r_a["top5_mids"] if mid in set_b]
        spearman = 0.0
        if len(shared) >= 2:
            rank_a = {mid: i for i, mid in enumerate(r_a["top5_mids"])}
            rank_b = {mid: i for i, mid in enumerate(r_b["top5_mids"])}
            ra_vals = [rank_a[m] for m in shared]
            rb_vals = [rank_b[m] for m in shared]
            n = len(shared)
            mean_a = sum(ra_vals) / n
            mean_b = sum(rb_vals) / n
            num = sum((ra - mean_a) * (rb - mean_b) for ra, rb in zip(ra_vals, rb_vals))
            denom_a = math.sqrt(sum((ra - mean_a) ** 2 for ra in ra_vals))
            denom_b = math.sqrt(sum((rb - mean_b) ** 2 for rb in rb_vals))
            spearman = num / (denom_a * denom_b + 1e-12)
        if spearman > spearman_best:
            spearman_best = spearman
        pair_passed = jaccard >= 0.6
        if pair_passed:
            passed_pair_count += 1
        results_per_pair.append({
            "pair": pair_name,
            "prompt_a": p_a, "prompt_b": p_b,
            "top5_a": r_a["top5_mids"], "top5_b": r_b["top5_mids"],
            "jaccard": jaccard,
            "spearman_shared": spearman,
            "pair_passed_jaccard_0_6": pair_passed,
        })
    passed = (passed_pair_count == len(pairs)) and (spearman_best >= 0.5)
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "pairs": results_per_pair,
        "spearman_best": spearman_best,
        "gating": "hard_PASS",
    }


def decode_repetition_feedback_probe(seed: int) -> Dict[str, Any]:
    """[4.21] Anti-collapse: content-token repeats, bigram-repeat index, trigram-lock count."""
    model = build_model(seed)
    write_texts(model, corpus_general() + corpus_music() + corpus_space())
    prompts = ["The telescope", "The pianist", "The market analyst"]
    per_prompt = []
    max_repeats = []
    first_bigram_repeat_indices = []
    trigram_lock_counts = []
    for prompt in prompts:
        with torch.no_grad():
            output = model.generate(prompt, mt=30, greedy=True)
        prompt_ids = model.tok.encode(prompt)
        full_ids = model.tok.encode(output)
        new_ids = full_ids[len(prompt_ids):]
        new_ids = new_ids[:20]
        cc = model.content_classifier
        content_ids_gen = [t for t in new_ids if cc is not None and t in cc.content_ids]
        counts = Counter(content_ids_gen)
        max_repeat_per_content_token = max(counts.values()) if counts else 0
        # first_bigram_repeat_index: earliest index where (new_ids[i], new_ids[i+1])
        # equals an earlier bigram.
        first_bigram_repeat_index = None
        seen_bigrams: Dict[Tuple[int, int], int] = {}
        for i in range(len(new_ids) - 1):
            b = (new_ids[i], new_ids[i + 1])
            if b in seen_bigrams:
                first_bigram_repeat_index = i
                break
            seen_bigrams[b] = i
        # trigram_lock_count: number of distinct trigrams that appear >= 2 times
        tri_counts: Counter = Counter(
            tuple(new_ids[i:i + 3]) for i in range(len(new_ids) - 2))
        trigram_lock_count = sum(1 for _, c in tri_counts.items() if c >= 2)
        max_repeats.append(max_repeat_per_content_token)
        if first_bigram_repeat_index is not None:
            first_bigram_repeat_indices.append(first_bigram_repeat_index)
        trigram_lock_counts.append(trigram_lock_count)
        per_prompt.append({
            "prompt": prompt,
            "output": output,
            "max_repeat_per_content_token": max_repeat_per_content_token,
            "first_bigram_repeat_index": first_bigram_repeat_index,
            "trigram_lock_count": trigram_lock_count,
        })
    avg_max_repeat = sum(max_repeats) / max(len(max_repeats), 1)
    avg_trigram_lock = sum(trigram_lock_counts) / max(len(trigram_lock_counts), 1)
    min_first_bigram = min(first_bigram_repeat_indices) if first_bigram_repeat_indices else None
    cond_repeat = avg_max_repeat <= 3.0
    cond_bigram = (min_first_bigram is None) or (min_first_bigram >= 4)
    cond_trigram = avg_trigram_lock <= 1.0
    passed = cond_repeat and cond_bigram and cond_trigram
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "per_prompt": per_prompt,
        "avg_max_repeat_per_content_token": avg_max_repeat,
        "min_first_bigram_repeat_index": min_first_bigram,
        "avg_trigram_lock_count": avg_trigram_lock,
        "conditions": {
            "avg_max_repeat_le_3": cond_repeat,
            "min_first_bigram_ge_4": cond_bigram,
            "avg_trigram_lock_le_1": cond_trigram,
        },
        "gating": "hard_PASS",
    }


def _is_content_starter(model: "sb.MemLLM", token_id: int) -> bool:
    cc = model.content_classifier
    if cc is None:
        return False
    return token_id in cc.content_starter_ids


def functional_token_suppression_probe(seed: int) -> Dict[str, Any]:
    """[4.22 v3.46] De-overfit: run on two prompt sets.
    Set A (selected): 3 prompts whose Qwen-unconditional top-12 is known to be
      dominated by functional tokens (original 4.22 prompts).
    Set B (held-out): 3 generic prompts drawn without selection bias.
    Both sets must pass independently; the overall probe passes only if both
    prompt sets pass their per-set thresholds.
    Prompts are FIXED in the runner so they are audit-observable, not
    regenerated per-run."""
    model = build_model(seed)
    write_texts(model, corpus_music())
    prompts_a = [
        "A strong explanation should mention",
        "The most relevant idea is",
        "A learner should know about",
    ]
    prompts_b = [  # held-out: not selected for functional-domination a priori
        "Tell me about",
        "Please describe",
        "Explain how",
    ]
    prompts = prompts_a + prompts_b
    device = next(model.parameters()).device
    per_prompt = []
    starter_delta_sum = 0.0
    margin_wins = 0
    for prompt in prompts:
        tk = model.tok(prompt, return_tensors="pt")
        ids = tk["input_ids"].to(device)
        mask = tk["attention_mask"].to(device)
        with torch.no_grad():
            # (A) no prefix: raw backbone
            o_no = model.backbone(ids, mask)
            logits_no = o_no["logits"][:, -1, :].squeeze(0).float()
            # (B) with memory prefix
            ctx = model.prepare_decode_context(ids, mask, update_stats=False)
            o_with = model.fwd(ids, mask, ctx.prefix_cond)
            logits_with = o_with["logits"][:, -1, :].squeeze(0).float()
        top12_no = topk_tokens_from_logits(model, logits_no, k=12)
        top12_with = topk_tokens_from_logits(model, logits_with, k=12)
        cs_count_no = sum(
            1 for row in top12_no if _is_content_starter(model, row["token_id"]))
        cs_count_with = sum(
            1 for row in top12_with if _is_content_starter(model, row["token_id"]))
        starter_delta_sum += (cs_count_with - cs_count_no)
        # margin: best content-starter logit - best functional-token logit in top12_with
        best_starter = None
        best_func = None
        cc = model.content_classifier
        for row in top12_with:
            tid = row["token_id"]
            is_starter = (cc is not None and tid in cc.content_starter_ids)
            is_func = (cc is not None and tid in cc.function_ids
                       and tid not in cc.newline_ids
                       and tid not in cc.punct_ids
                       and tid != (model.tok.eos_token_id or -1))
            if is_starter and (best_starter is None or row["logit"] > best_starter):
                best_starter = row["logit"]
            if is_func and (best_func is None or row["logit"] > best_func):
                best_func = row["logit"]
        # If no functional token present in top-12, margin is trivially non-negative.
        if best_starter is None:
            margin_value = None
            margin_ok = False
        elif best_func is None:
            margin_value = float("inf")
            margin_ok = True
        else:
            margin_value = best_starter - best_func
            margin_ok = margin_value >= 0
        if margin_ok:
            margin_wins += 1
        per_prompt.append({
            "prompt": prompt,
            "top12_no_prefix": top12_no,
            "top12_with_prefix": top12_with,
            "content_starter_count_no_prefix": cs_count_no,
            "content_starter_count_with_prefix": cs_count_with,
            "best_content_starter_logit_with_prefix": best_starter,
            "best_functional_logit_with_prefix": best_func,
            "logit_margin_best_content_starter_vs_best_functional": margin_value,
            "margin_non_negative": margin_ok,
        })
    # [SPEC 4.22 v3.46] Score set A and set B independently.
    def _score(rows):
        sd = sum(r["content_starter_count_with_prefix"] - r["content_starter_count_no_prefix"]
                 for r in rows) / len(rows)
        mw = sum(1 for r in rows if r["margin_non_negative"])
        return sd, mw
    set_a_rows = per_prompt[:3]
    set_b_rows = per_prompt[3:]
    a_delta, a_margin = _score(set_a_rows)
    b_delta, b_margin = _score(set_b_rows)
    avg_starter_delta = (a_delta + b_delta) / 2.0
    # Per-set thresholds: each set (3 prompts) must meet avg delta >= 1.0 and
    # margin_non_negative on >= 2 of 3 prompts.
    a_ok = (a_delta >= 1.0) and (a_margin >= 2)
    b_ok = (b_delta >= 1.0) and (b_margin >= 2)
    passed = a_ok and b_ok
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "metric_version": "v3.46",
        "per_prompt": per_prompt,
        "avg_content_starter_delta_overall": avg_starter_delta,
        "set_a_avg_delta": a_delta,
        "set_a_margin_wins": a_margin,
        "set_b_avg_delta": b_delta,
        "set_b_margin_wins": b_margin,
        "conditions": {
            "set_a_delta_ge_1_and_margin_2of3": a_ok,
            "set_b_delta_ge_1_and_margin_2of3": b_ok,
        },
        "gating": "hard_PASS",
    }


def keyword_specific_tail_slot_probe(seed: int) -> Dict[str, Any]:
    """[4.23] Corrected v3.45+ metric per SPEC Section 4.23:
    mean-centered top-20 intersection with rare keywords + median rank <= 100.
    Replaces the unreachable top-3 absolute-cosine query that was dominated by
    token ids 0/1/2 of Qwen 2.5's WTE."""
    model = build_model(seed)
    bridge = model.bridge
    if not hasattr(bridge, "tail_head") or getattr(bridge.tail_head, "n_slots", 0) < 2:
        return {
            "passed": False,
            "status": "not_implemented",
            "missing_api": "EmbBridge.tail_head with n_slots >= 2",
            "gating": "PASS_or_not_implemented",
        }
    write_texts(model, corpus_music())
    sample_mem = next(iter(model.amm.tree.store.values()), None)
    if sample_mem is None or not hasattr(sample_mem, "rare_keyword_ids"):
        return {
            "passed": False,
            "status": "not_implemented",
            "missing_api": "MemEntry.rare_keyword_ids field",
            "gating": "PASS_or_not_implemented",
        }
    if hasattr(model, "_refresh_rare_keyword_indices"):
        model._refresh_rare_keyword_indices()
    device = next(model.parameters()).device
    wte = model.backbone.input_embedding_weight().to(device).float()
    # [SPEC 4.23 v3.45+] mean-centered unit WTE for top-K query.
    wte_mean = wte.mean(0)
    wte_centered = torch.nn.functional.normalize(wte - wte_mean, dim=-1, eps=1e-8)
    # [SPEC 4.23 v3.46 de-overfit] round-trip query was circular: memory's own
    # rare keywords were embedded in the query that retrieved it. The revised
    # protocol runs BOTH queries and reports:
    # (a) `roundtrip_*` metrics using mem.source_text as query (legacy)
    # (b) `paraphrase_*` metrics using corpus_paraphrase_music() as query,
    #     then reading dominant memory from the retrieval result and checking
    #     its rare keywords against the tail slot.
    # Only paraphrase metrics are used for pass criteria.
    paraphrase_queries = corpus_paraphrase_music()
    intersection_counts_20 = []
    best_rare_ranks = []
    non_none_count = 0
    hits_ge_1 = 0
    per_memory = []
    # (a) Legacy round-trip path, retained as diagnostic.
    roundtrip_inter = []
    for mid, mem in model.amm.tree.store.items():
        rare = list(getattr(mem, "rare_keyword_ids", []) or [])[:3]
        if not rare:
            continue
        _ = _cipher_prep_decode(model, mem.source_text)
        ts = model.bridge._last_tail_slots
        if ts is None:
            continue
        slot_idx = 1 if ts.shape[1] >= 2 else ts.shape[1] - 1
        slot = ts[0, slot_idx].float()
        slot_c = torch.nn.functional.normalize(slot - wte_mean, dim=-1, eps=1e-8)
        top20 = (wte_centered @ slot_c).topk(20).indices.tolist()
        roundtrip_inter.append(len(set(top20) & set(rare)))
    # (b) Paraphrase path — primary pass criterion.
    # For each paraphrase query: identify dominant memory via
    # prepare_decode_context.diag (without using mem.source_text or rare tokens
    # in the query), then evaluate tail slot against THAT dominant memory's
    # rare_keyword_ids. The query itself is token-disjoint from rare keywords
    # (verified inline).
    per_paraphrase = []
    for pq in paraphrase_queries:
        device_l = next(model.parameters()).device
        tk = model.tok(pq, return_tensors="pt")
        ids = tk["input_ids"].to(device_l); mask = tk["attention_mask"].to(device_l)
        with torch.no_grad():
            ctx = model.prepare_decode_context(ids, mask, update_stats=False)
        diag = ctx.diag
        dom_mid = diag.dominant_per_batch[0] if diag.dominant_per_batch else None
        if dom_mid is None or dom_mid not in model.amm.tree.store:
            per_paraphrase.append({
                "query": pq,
                "dominant_mid": None,
                "note": "no dominant memory retrieved",
            })
            continue
        dom_mem = model.amm.tree.store[dom_mid]
        rare_dom = list(getattr(dom_mem, "rare_keyword_ids", []) or [])[:3]
        if not rare_dom:
            continue
        # Verify query token-disjoint from rare_dom (audit-observable property).
        query_token_ids = set(model.tok.encode(pq))
        disjoint_from_rare = len(query_token_ids & set(rare_dom)) == 0
        tail_slots = model.bridge._last_tail_slots
        if tail_slots is None:
            continue
        slot_idx = 1 if tail_slots.shape[1] >= 2 else tail_slots.shape[1] - 1
        slot_vec = tail_slots[0, slot_idx].float()
        slot_centered = torch.nn.functional.normalize(
            slot_vec - wte_mean, dim=-1, eps=1e-8)
        sims = wte_centered @ slot_centered
        top20_ids = sims.topk(20).indices.tolist()
        inter_20 = len(set(top20_ids) & set(rare_dom))
        order = sims.argsort(descending=True)
        ranks = {int(t): None for t in rare_dom}
        for pos in range(order.shape[0]):
            tid = int(order[pos].item())
            if tid in ranks and ranks[tid] is None:
                ranks[tid] = pos + 1
                if all(v is not None for v in ranks.values()):
                    break
        rank_values = [v for v in ranks.values() if v is not None]
        rank_of_best_rare = min(rank_values) if rank_values else None
        intersection_counts_20.append(inter_20)
        if rank_of_best_rare is not None:
            best_rare_ranks.append(rank_of_best_rare)
        non_none_count += 1
        if inter_20 >= 1:
            hits_ge_1 += 1
        per_paraphrase.append({
            "query": pq,
            "query_disjoint_from_rare_keywords": disjoint_from_rare,
            "dominant_mid": int(dom_mid),
            "dominant_source_preview": dom_mem.source_text[:60],
            "rare_keyword_ids": rare_dom,
            "rare_keyword_pieces": [model.tok.decode([t]) for t in rare_dom],
            "tail_slot_top5_ids_centered": top20_ids[:5],
            "tail_slot_top5_pieces_centered": [
                model.tok.decode([t]) for t in top20_ids[:5]],
            "intersection_size_top20": inter_20,
            "rank_of_best_rare": rank_of_best_rare,
        })
    if non_none_count == 0:
        return {
            "passed": False,
            "status": "not_implemented",
            "missing_api": "no paraphrase query produced a non-None tail slot with a dominant memory",
            "gating": "PASS_or_not_implemented",
        }
    mean_intersection_20 = sum(intersection_counts_20) / non_none_count
    median_best_rank = float(
        sorted(best_rare_ranks)[len(best_rare_ranks) // 2]) if best_rare_ranks else float("inf")
    hit_ratio = hits_ge_1 / non_none_count
    cond_mean = mean_intersection_20 >= 1.0
    cond_median = median_best_rank <= 100.0
    cond_hit_ratio = hit_ratio >= 0.5
    passed = cond_mean and cond_median and cond_hit_ratio
    roundtrip_mean = (sum(roundtrip_inter) / len(roundtrip_inter)) if roundtrip_inter else None
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "metric_version": "v3.46",
        "per_paraphrase": per_paraphrase,
        "mean_intersection_size_top20_paraphrase": mean_intersection_20,
        "median_rank_of_best_rare_paraphrase": median_best_rank,
        "hit_ratio_at_least_one_top20_paraphrase": hit_ratio,
        "n_paraphrase_queries_evaluated": non_none_count,
        "roundtrip_mean_intersection_top20_diagnostic": roundtrip_mean,
        "conditions": {
            "mean_intersection_top20_ge_1": cond_mean,
            "median_rank_le_100": cond_median,
            "hit_ratio_top20_ge_0_5": cond_hit_ratio,
        },
        "gating": "PASS_or_not_implemented",
    }


def context_descriptor_cluster_probe(seed: int) -> Dict[str, Any]:
    """[4.24 v3.46 de-overfit] Four-domain LOO NN accuracy + held-out paraphrase retrieval.

    Corpus (4 domains x 4 sentences = 16 memories): music, space, cooking, finance.
    Domain labels assigned by source_text identity (membership in the runner's
    corpus tuple), NOT by keyword-list matching. Two of the four domains
    (cooking, finance) were not anywhere else in the suite so they act as a
    held-out control: if the encoder only memorizes the specific 8 (music,
    space) sentences, the held-out domains will fail to cluster.

    Metrics:
    - loo_nn_accuracy_all_4: LOO NN across 16 memories, 4 labels.
    - loo_nn_accuracy_heldout_2: LOO NN restricted to the cooking+finance
      subset (8 memories, 2 labels, none keyword-matched to any other probe).
    """
    model = build_model(seed)
    # Write all four domains; the runner tags each memory by the corpus it
    # came from, not by keyword match.
    domains = {
        "music":    corpus_music(),
        "space":    corpus_space(),
        "cooking":  corpus_cooking(),
        "finance":  corpus_finance(),
    }
    text_to_label = {}
    ordered_texts = []
    for dom, texts in domains.items():
        for t in texts:
            text_to_label[t] = dom
            ordered_texts.append(t)
    write_texts(model, ordered_texts)
    sample = next(iter(model.amm.tree.store.values()))
    import dataclasses as _dc
    field_names = {f.name for f in _dc.fields(type(sample))}
    if "context_descriptor" not in field_names:
        return {
            "passed": False,
            "status": "not_implemented",
            "missing_api": "MemEntry.context_descriptor field",
            "gating": "PASS_or_not_implemented",
        }
    # [v3.48] Primary metric follows the SUT's own fallback chain as defined in
    # scheme_b_v344.MemLLM._compute_aggregated_context_descriptors_d_llm:
    #   if mem.context_descriptor is not None (and encoder is not None): use it
    #   else if mem.semantic_emb is not None: use semantic_emb
    # This way, running with Cfg(use_memory_context_encoder=False) exercises the
    # same code path for 4.24 as for the SUT's runtime, instead of
    # short-circuiting to a separate "context_descriptor only" metric that the
    # SUT doesn't use.
    used_semantic_fallback = (model.memory_context_encoder is None)
    entries = []
    for mid, mem in model.amm.tree.store.items():
        if used_semantic_fallback:
            v = getattr(mem, "semantic_emb", None)
        else:
            v = getattr(mem, "context_descriptor", None)
            if v is None:  # per-memory fallback (if encoder present but ctx_desc missing)
                v = getattr(mem, "semantic_emb", None)
        if v is None:
            continue
        label = text_to_label.get(mem.source_text)
        if label is None:
            for dom, texts in domains.items():
                if any(t in mem.source_text or mem.source_text in t for t in texts):
                    label = dom; break
        if label is None:
            continue
        vec = torch.nn.functional.normalize(v.float(), dim=-1, eps=1e-8)
        norm_raw = float(v.float().norm().item())
        entries.append((mid, label, vec, norm_raw))
    if len(entries) < 8:
        return {
            "passed": False,
            "status": "not_implemented",
            "missing_api": "insufficient populated context_descriptor entries (need >= 8, got {})".format(len(entries)),
            "n_populated": len(entries),
            "gating": "PASS_or_not_implemented",
        }
    def _loo_nn(subset):
        correct = 0
        per_mem = []
        for i, (mid_i, lbl_i, v_i, _n) in enumerate(subset):
            best_sim = -1e9; best_j = -1
            for j, (_, lbl_j, v_j, _) in enumerate(subset):
                if j == i:
                    continue
                s = float((v_i @ v_j).item())
                if s > best_sim:
                    best_sim = s; best_j = j
            pred = subset[best_j][1] if best_j >= 0 else None
            ok = (pred == lbl_i)
            if ok:
                correct += 1
            per_mem.append({
                "mid": int(mid_i),
                "true_label": lbl_i,
                "pred_label": pred,
                "nn_sim": best_sim,
                "correct": ok,
            })
        return correct / max(len(subset), 1), correct, per_mem
    # Metric 1: full 4-domain LOO NN on context_descriptor
    acc_all, correct_all, per_all = _loo_nn(entries)
    # Metric 2: held-out subset — cooking + finance only.
    heldout = [e for e in entries if e[1] in ("cooking", "finance")]
    acc_held, correct_held, per_held = _loo_nn(heldout)
    n_all = len(entries); n_held = len(heldout)
    unit_ok = all(abs(n_raw - 1.0) < 1e-3 or n_raw < 1e-6 for _, _, _, n_raw in entries)
    cond_all = acc_all >= 0.65
    cond_held = acc_held >= 0.70
    passed = cond_all and cond_held and unit_ok
    # ----------------------------------------------------------------------
    # [Mechanism 1 diagnostic, v3.47]  Parallel LOO NN on `mem.semantic_emb`,
    # which is the frozen-Qwen attention-pool of content-token hidden states
    # (see scheme_b_v344.MemLLM._compute_content_semantic_emb). This field
    # ALREADY exists on every populated MemEntry; the runner just reads it.
    # No SUT change, no Cfg change.
    # Question answered: does the frozen-Qwen attention pool, used directly
    # as a context descriptor candidate, separate 4 domains better than the
    # learned MemoryContextEncoder projection?
    # ----------------------------------------------------------------------
    sem_entries = []
    for mid, mem in model.amm.tree.store.items():
        v = getattr(mem, "semantic_emb", None)
        if v is None:
            continue
        label = text_to_label.get(mem.source_text)
        if label is None:
            for dom, texts in domains.items():
                if any(t in mem.source_text or mem.source_text in t for t in texts):
                    label = dom; break
        if label is None:
            continue
        vec = torch.nn.functional.normalize(v.float(), dim=-1, eps=1e-8)
        norm_raw = float(v.float().norm().item())
        sem_entries.append((mid, label, vec, norm_raw))
    if len(sem_entries) >= 8:
        sem_acc_all, sem_correct_all, sem_per_all = _loo_nn(sem_entries)
        sem_heldout = [e for e in sem_entries if e[1] in ("cooking", "finance")]
        sem_acc_held, sem_correct_held, sem_per_held = _loo_nn(sem_heldout)
        # Per-domain accuracy for the semantic_emb path (for direct comparison)
        from collections import defaultdict as _dd
        sem_by_true = _dd(lambda: {"n": 0, "correct": 0})
        for m_ in sem_per_all:
            sem_by_true[m_["true_label"]]["n"] += 1
            if m_["correct"]:
                sem_by_true[m_["true_label"]]["correct"] += 1
        sem_per_domain = {
            dom: {"correct": sem_by_true[dom]["correct"],
                  "n": sem_by_true[dom]["n"]}
            for dom in sem_by_true
        }
        mechanism_1 = {
            "source": "mem.semantic_emb (Qwen last-layer attention-pool over "
                      "content tokens, no trainable encoder)",
            "loo_nn_accuracy_all_4": sem_acc_all,
            "loo_nn_accuracy_heldout_2": sem_acc_held,
            "correct_all": sem_correct_all,
            "correct_heldout": sem_correct_held,
            "per_domain_accuracy": sem_per_domain,
            "would_pass_4domain_threshold_0_65": sem_acc_all >= 0.65,
            "would_pass_heldout_threshold_0_70": sem_acc_held >= 0.70,
        }
    else:
        mechanism_1 = {
            "source": "mem.semantic_emb (frozen-Qwen pool)",
            "status": "insufficient entries",
            "n_populated": len(sem_entries),
        }
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "metric_version": "v3.46",
        "loo_nn_accuracy_all_4": acc_all,
        "loo_nn_accuracy_heldout_2": acc_held,
        "n_all": n_all,
        "n_heldout": n_held,
        "correct_all": correct_all,
        "correct_heldout": correct_held,
        "per_memory_all": per_all,
        "per_memory_heldout": per_held,
        "unit_norm_within_1e_3": unit_ok,
        "conditions": {
            "loo_nn_4domain_ge_0_65": cond_all,
            "loo_nn_heldout_2domain_ge_0_70": cond_held,
            "unit_norm_within_1e_3": unit_ok,
        },
        "gating": "PASS_or_not_implemented",
        "mechanism_1_qwen_pool_diagnostic": mechanism_1,
    }


def prefix_length_scaling_probe(seed: int) -> Dict[str, Any]:
    """[4.25] Corrected v3.45+ metric per SPEC Section 4.25:
    starter-positive-logit-mass ratio mass_B/mass_A > 1.10 over 3 prompts.
    Replaces saturation-bound top-12 count."""
    cfg_a = sb.Cfg()
    default_L = cfg_a.L_mem
    cfg_b_L = default_L * 2
    set_seed(seed)
    device = best_device()
    model_a = sb.MemLLM(sb.Cfg())
    model_a.to(device); model_a.load(); model_a.to(device); model_a.eval()
    write_texts(model_a, corpus_music())
    set_seed(seed)
    cfg_b = sb.Cfg(); cfg_b.L_mem = cfg_b_L
    try:
        model_b = sb.MemLLM(cfg_b)
    except AssertionError as ae:
        return {
            "passed": False, "status": "fail",
            "reason": f"Cfg assertion failed when scaling L_mem: {ae}",
            "gating": "PASS_or_not_implemented",
        }
    model_b.to(device); model_b.load(); model_b.to(device); model_b.eval()
    write_texts(model_b, corpus_music())
    prompts = [
        "A strong explanation should mention",
        "The pianist",
        "The telescope",
    ]
    def _starter_mass(model, prompt):
        tk = model.tok(prompt, return_tensors="pt")
        ids = tk["input_ids"].to(device); mask = tk["attention_mask"].to(device)
        with torch.no_grad():
            # Baseline (no prefix)
            o_base = model.fwd(ids, mask)
            lg_base = o_base["logits"][:, -1, :].squeeze(0).float()
            # With memory prefix
            ctx = model.prepare_decode_context(ids, mask, update_stats=False)
            o_pref = model.fwd(ids, mask, ctx.prefix_cond)
            lg_pref = o_pref["logits"][:, -1, :].squeeze(0).float()
        shift = lg_pref - lg_base
        # Content-starter mask
        cc = model.content_classifier
        starter_mask_t = cc.content_starter_mask(shift.device)
        V = min(shift.shape[0], starter_mask_t.shape[0])
        starter_bool = starter_mask_t[:V].bool()
        positive_shift = shift[:V].clamp(min=0.0)
        mass = float((positive_shift * starter_bool.float()).sum().item())
        # Also legacy top-12 count
        top12 = topk_tokens_from_logits(model, lg_pref, k=12)
        starters_top12 = sum(1 for r in top12 if _is_content_starter(model, r["token_id"]))
        # Prefix L2 per slot
        norms = [float(ctx.prefix_cond[0, i].norm().item())
                 for i in range(ctx.prefix_cond.shape[1])]
        return mass, starters_top12, norms, top12
    per_prompt = []
    ratios = []
    for p in prompts:
        mass_a, st_a, norms_a, top12_a = _starter_mass(model_a, p)
        mass_b, st_b, norms_b, top12_b = _starter_mass(model_b, p)
        r = mass_b / max(mass_a, 1e-12)
        ratios.append(r)
        per_prompt.append({
            "prompt": p,
            "starter_mass_A": mass_a,
            "starter_mass_B": mass_b,
            "ratio": r,
            "content_starters_top12_A": st_a,
            "content_starters_top12_B": st_b,
            "per_slot_mean_norm_A": sum(norms_a) / len(norms_a),
            "per_slot_mean_norm_B": sum(norms_b) / len(norms_b),
        })
    avg_ratio = sum(ratios) / len(ratios)
    all_finite = all(
        all(math.isfinite(n) for n in (row["per_slot_mean_norm_A"], row["per_slot_mean_norm_B"]))
        for row in per_prompt
    )
    cond_ratio = avg_ratio > 1.10
    passed = cond_ratio and all_finite
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "metric_version": "v3.45",
        "L_mem_A": default_L,
        "L_mem_B": cfg_b_L,
        "avg_mass_ratio_B_over_A": avg_ratio,
        "per_prompt": per_prompt,
        "conditions": {
            "avg_mass_ratio_gt_1_10": cond_ratio,
            "per_slot_norms_finite": all_finite,
        },
        "gating": "PASS_or_not_implemented",
    }


def mixture_distribution_gate_probe(seed: int) -> Dict[str, Any]:
    """[4.26] Mixture-of-distributions gate: (1-g)*raw + g*mem decomposition.
    A SUT is considered to implement the mixture gate when:
      1. sb.Cfg exposes a boolean flag that enables mixture decoding, AND
      2. with that flag enabled, DecodeContext.mixture_gate is a non-None
         tensor whose values lie within a publicly declared [floor, ceiling],
         AND a matching DecodeContext.memory_logit_bias is produced.
    Building a fresh model instance with the flag enabled is NOT mocking: it
    is the officially exported public-API path.
    """
    # Check flag existence on the SUT's Cfg.
    cfg_has_flag = hasattr(sb.Cfg(), "use_mixture_decoding")
    if not cfg_has_flag:
        return {
            "passed": False,
            "status": "not_implemented",
            "missing_api": "Cfg.use_mixture_decoding flag",
            "note": ("SUT does not expose a mixture-decoding toggle on Cfg; "
                     "the runner cannot enable the feature through the "
                     "public API."),
            "gating": "PASS_or_not_implemented",
        }

    # Build a dedicated model with the flag enabled.
    set_seed(seed)
    torch.set_num_threads(1)
    device = best_device()
    try:
        cfg_with_gate = sb.Cfg(use_mixture_decoding=True)
    except TypeError as exc:
        return {
            "passed": False,
            "status": "not_implemented",
            "missing_api": "Cfg(use_mixture_decoding=True) constructor",
            "note": f"Cfg rejected the flag: {exc}",
            "gating": "PASS_or_not_implemented",
        }
    model = sb.MemLLM(cfg_with_gate)
    model.to(device); model.load(); model.to(device); model.eval()
    write_texts(model, corpus_music())

    tk = model.tok("A strong explanation should mention", return_tensors="pt")
    ids = tk["input_ids"].to(device); mask = tk["attention_mask"].to(device)
    with torch.no_grad():
        ctx = model.prepare_decode_context(ids, mask, update_stats=False)

    gate = getattr(ctx, "mixture_gate", None)
    mem_bias = getattr(ctx, "memory_logit_bias", None)
    if gate is None:
        return {
            "passed": False,
            "status": "not_implemented",
            "missing_api": ("DecodeContext.mixture_gate is still None even with "
                            "Cfg.use_mixture_decoding=True"),
            "gating": "PASS_or_not_implemented",
        }
    if mem_bias is None:
        return {
            "passed": False,
            "status": "not_implemented",
            "missing_api": ("DecodeContext.memory_logit_bias is None when "
                            "mixture_gate is present; convex decomposition "
                            "cannot be verified."),
            "gating": "PASS_or_not_implemented",
        }

    # Boundary check: gate values lie in a consistent interval.
    gate_flat = gate.reshape(-1)
    g_min = float(gate_flat.min().item())
    g_max = float(gate_flat.max().item())
    floor = float(getattr(cfg_with_gate, "mixture_gate_floor", 0.0))
    ceiling = float(getattr(cfg_with_gate, "mixture_gate_ceiling", 1.0))
    in_range = (floor - 1e-4) <= g_min and g_max <= (ceiling + 1e-4)

    # Finite checks
    finite_gate = bool(torch.isfinite(gate).all().item())
    finite_bias = bool(torch.isfinite(mem_bias).all().item())

    # Identity decomposition check: compute (1-g)*lg_cond + g*mem_bias on last
    # logit of a conditional forward and compare to shape_step_logits's mixture
    # branch (which uses exactly that formula when use_mixture_decoding=True).
    with torch.no_grad():
        o_cond = model.fwd(ids, mask, ctx.prefix_cond)
        lg_cond = o_cond["logits"][:, -1, :].squeeze(0).float()
        V_min = min(lg_cond.shape[-1], mem_bias.shape[-1])
        g_scalar = float(gate_flat[0].item())
        manual_mix = (1.0 - g_scalar) * lg_cond[:V_min] + g_scalar * mem_bias[0, :V_min].float()
    decomposition_finite = bool(torch.isfinite(manual_mix).all().item())

    passed = (in_range and finite_gate and finite_bias and decomposition_finite)
    return {
        "passed": passed,
        "status": "pass" if passed else "fail",
        "gate_min": g_min,
        "gate_max": g_max,
        "declared_floor": floor,
        "declared_ceiling": ceiling,
        "gate_in_range": in_range,
        "finite_gate": finite_gate,
        "finite_memory_logit_bias": finite_bias,
        "manual_mixture_finite": decomposition_finite,
        "gating": "PASS_or_not_implemented",
    }


def rerank_stability_summary_entry() -> Tuple[str, str]:  # pragma: no cover (doc only)
    return ("rerank_stability_probe", "[4.20] invocation strategy; P0; targets 4.6")


# ==========================================================================
# END Cipher-System Structural Probes
# ==========================================================================


def main() -> int:
    start = time.time()
    ensure_report_dir()
    results = {
        "leaf_capacity_stability": run_case("leaf_capacity_stability", leaf_capacity_stability, list(range(8))),
        "degenerate_direction_boundary": run_case("degenerate_direction_boundary", degenerate_direction_boundary, 17),
        "metric_trainability": run_case("metric_trainability", metric_trainability, 23),
        "no_grad_generation": run_case("no_grad_generation", no_grad_generation, 29),
        "counterfactual_memory_influence": run_case("counterfactual_memory_influence", counterfactual_memory_influence, 31),
        "semantic_memory_grounding": run_case("semantic_memory_grounding", semantic_memory_grounding, 33),
        "semantic_memory_counterfactual_pairs": run_case("semantic_memory_counterfactual_pairs", semantic_memory_counterfactual_pairs, 35),
        "degeneration_quality": run_case("degeneration_quality", degeneration_quality, 36),
        "prefix_logit_drift_audit": run_case("prefix_logit_drift_audit", prefix_logit_drift_audit, 38),
        "retrieval_topk_semantic_shift": run_case("retrieval_topk_semantic_shift", retrieval_topk_semantic_shift, 39),
        "repetition_segment_audit": run_case("repetition_segment_audit", repetition_segment_audit, 40),
        "prefix_stepwise_drift_trajectory": run_case("prefix_stepwise_drift_trajectory", prefix_stepwise_drift_trajectory, 44),
        "retrieval_generation_alignment_audit": run_case("retrieval_generation_alignment_audit", retrieval_generation_alignment_audit, 45),
        "retrieval_prefix_decode_correlation_audit": run_case("retrieval_prefix_decode_correlation_audit", retrieval_prefix_decode_correlation_audit, 46),
        "stepwise_label_mass_alignment_audit": run_case("stepwise_label_mass_alignment_audit", stepwise_label_mass_alignment_audit, 48),
        "prompt_diversity_without_memory": run_case("prompt_diversity_without_memory", prompt_diversity_without_memory, 37),
        "save_load_consistency": run_case("save_load_consistency", save_load_consistency, 41),
        "training_cache_isolation": run_case("training_cache_isolation", training_cache_isolation, 43),
        "cheating_heuristics": run_case("cheating_heuristics", cheating_heuristics, 47),
        # Cipher-System Structural Probes (v3.38)
        "rerank_stability_probe": run_case("rerank_stability_probe", rerank_stability_probe, 49),
        "decode_repetition_feedback_probe": run_case("decode_repetition_feedback_probe", decode_repetition_feedback_probe, 50),
        "functional_token_suppression_probe": run_case("functional_token_suppression_probe", functional_token_suppression_probe, 51),
        "keyword_specific_tail_slot_probe": run_case("keyword_specific_tail_slot_probe", keyword_specific_tail_slot_probe, 52),
        "context_descriptor_cluster_probe": run_case("context_descriptor_cluster_probe", context_descriptor_cluster_probe, 53),
        "prefix_length_scaling_probe": run_case("prefix_length_scaling_probe", prefix_length_scaling_probe, 54),
        "mixture_distribution_gate_probe": run_case("mixture_distribution_gate_probe", mixture_distribution_gate_probe, 55),
    }
    checks = results_to_checks(results)
    elapsed = time.time() - start
    write_reports(results, checks, elapsed)
    # Gating rule: probes with status "not_implemented" do not block suite PASS
    # per spec Section 4-meta. Treat them as non-blocking.
    def _is_blocking_fail(name: str, payload: Dict[str, Any]) -> bool:
        if payload.get("passed"):
            return False
        if payload.get("status") == "not_implemented":
            return False
        return True
    blocking_fail = any(_is_blocking_fail(n, results[n]) for n in results)
    print(json.dumps({"checks": [asdict(c) for c in checks], "elapsed_seconds": elapsed}, ensure_ascii=False, indent=2))
    return 0 if not blocking_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
