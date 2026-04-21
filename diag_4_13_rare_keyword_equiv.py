#!/usr/bin/env python3
"""[4.13 diagnostic, v3.45] Verify that `_refresh_rare_keyword_indices()` at
end of `write()` closes the fresh-vs-load asymmetry in `MemEntry.rare_keyword_ids`.

This is NOT a probe.  It is a structural assertion that runs before the
audit to confirm the hypothesis laid out in the v3.45 plan (change #3).
Prints a PASS/FAIL line but does not gate anything.
"""
import os, sys, tempfile
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scheme_b_v344 as sb

CORPUS = [
    "The cat sat on the mat and watched the birds outside the window.",
    "Quantum computing uses qubits existing in superposition states.",
    "Machine learning algorithms identify patterns in large datasets.",
    "The ancient temple was hidden deep within the tropical rainforest.",
    "The stock market experienced significant volatility during the session.",
    "He practiced piano for hours perfecting a difficult Chopin nocturne.",
    "The restaurant served an exquisite five course meal with wine pairings.",
    "The professor explained relativity using simple everyday analogies.",
]

def build(seed):
    torch.manual_seed(seed)
    c = sb.Cfg()
    m = sb.MemLLM(c)
    dev = torch.device("cpu")
    m.to(dev); m.load(); m.to(dev); m.eval()
    return m

def main():
    print("[diag 4.13] build model_a, write corpus, save.")
    m_a = build(seed=41)
    for t in CORPUS:
        m_a.write(t, training_mode=True)
    tmp_path = tempfile.mktemp(suffix=".pt")
    m_a.save_memory(tmp_path)

    print("[diag 4.13] build model_b from same seed, load saved memory.")
    m_b = build(seed=41)
    m_b.load_memory(tmp_path)
    os.remove(tmp_path)

    # Per-mid bit-level comparisons.
    mids = sorted(m_a.amm.tree.store.keys())
    assert sorted(m_b.amm.tree.store.keys()) == mids, \
        "mid set differs after save/load"

    issues = []
    for mid in mids:
        ma = m_a.amm.tree.store[mid]
        mb = m_b.amm.tree.store[mid]
        if ma.rare_keyword_ids != mb.rare_keyword_ids:
            issues.append((
                "rare_keyword_ids", mid,
                list(ma.rare_keyword_ids),
                list(mb.rare_keyword_ids),
            ))
        for fname in ("base", "fiber", "dirn", "semantic_emb",
                      "context_descriptor"):
            va = getattr(ma, fname)
            vb = getattr(mb, fname)
            if va is None and vb is None:
                continue
            if va is None or vb is None:
                issues.append((fname, mid, "None-mismatch", va is None, vb is None))
                continue
            if va.shape != vb.shape:
                issues.append((fname, mid, "shape", tuple(va.shape), tuple(vb.shape)))
                continue
            if not torch.equal(va.cpu(), vb.cpu()):
                max_abs = (va.cpu().float() - vb.cpu().float()).abs().max().item()
                issues.append((fname, mid, "bit-diff", max_abs))
        for fname in ("content_token_ids", "expanded_content_ids",
                      "strict_starter_ids"):
            va = getattr(ma, fname)
            vb = getattr(mb, fname)
            if list(va) != list(vb):
                issues.append((fname, mid, "list-diff",
                               list(va), list(vb)))

    if issues:
        print(f"[diag 4.13] FOUND {len(issues)} inconsistencies between "
              f"fresh-and-saved vs loaded:")
        for it in issues[:12]:
            print("   ", it)
        if len(issues) > 12:
            print(f"    ... and {len(issues)-12} more")
        return 1
    print("[diag 4.13] CLEAN: all per-mem fields bit-identical "
          "between model_a (fresh + write + save) and model_b (load). "
          "If 4.13 still fails after this, the remaining bit-flip source "
          "is downstream of MemEntry fields.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
