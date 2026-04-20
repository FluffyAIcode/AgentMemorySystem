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
import re
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch

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


def write_reports(results: Dict[str, Any], checks: List[CheckResult], elapsed: float) -> None:
    ensure_report_dir()
    payload = {
        "generated_at_epoch": time.time(),
        "elapsed_seconds": elapsed,
        "checks": [asdict(c) for c in checks],
        "results": results,
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
    }
    checks = results_to_checks(results)
    elapsed = time.time() - start
    write_reports(results, checks, elapsed)
    print(json.dumps({"checks": [asdict(c) for c in checks], "elapsed_seconds": elapsed}, ensure_ascii=False, indent=2))
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
