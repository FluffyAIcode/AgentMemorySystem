#!/usr/bin/env python3
"""[4.23 diagnostic, v3.45 cond-buffer] Verify that bridge._last_cond_tail_slots
holds the residual-bearing tail slot after prepare_decode_context, and that the
mean-centered top-20 of slot_1 intersects the dominant memory's rare_keyword_ids.

This is NOT the probe — it's a sanity check that the cond-buffer plumbing
actually carries the cond-path tail.
"""
import os, sys
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scheme_b_v344 as sb

MUSIC = [
    "The pianist practiced arpeggios and Chopin nocturnes until midnight.",
    "A musician refined finger technique, phrasing, and pedal control on the piano.",
    "Classical interpretation often depends on dynamics, tempo rubato, and touch.",
    "A conservatory student studied etudes, scales, and expressive voicing on the keyboard.",
]

PARAPHRASES = [
    "She performed Beethoven sonatas with delicate phrasing on her grand piano.",
    "Harmonic analysis and ear training are core elements of music education.",
]

def main():
    torch.manual_seed(52)
    c = sb.Cfg()
    m = sb.MemLLM(c)
    dev = torch.device("cpu")
    m.to(dev); m.load(); m.to(dev); m.eval()
    for t in MUSIC:
        m.write(t, training_mode=True)
    m.amm.maybe_recluster(force=True)
    m._refresh_rare_keyword_indices()

    wte = m.backbone.input_embedding_weight().to(dev).float()
    wte_mean = wte.mean(0)
    wte_centered = F.normalize(wte - wte_mean, dim=-1, eps=1e-8)

    for q in PARAPHRASES:
        tk = m.tok(q, return_tensors="pt")
        ids = tk["input_ids"].to(dev); mask = tk["attention_mask"].to(dev)
        with torch.no_grad():
            ctx = m.prepare_decode_context(ids, mask, update_stats=False)

        dom_mid = ctx.diag.dominant_per_batch[0] if ctx.diag.dominant_per_batch else None
        dom_mem = m.amm.tree.store.get(dom_mid)
        rare = dom_mem.rare_keyword_ids if dom_mem else []

        # Both buffers side by side.
        ts_any  = m.bridge._last_tail_slots
        ts_cond = m.bridge._last_cond_tail_slots
        res_cond = m.bridge._last_cond_residual
        diag_cond = m.bridge._last_cond_inject_diag
        diag_any  = m.bridge._last_inject_diag

        print(f'\nquery: {q}')
        print(f'  dom_mid={dom_mid}  dom_source={dom_mem.source_text!r}')
        print(f'  rare_keyword_ids={rare}  pieces={[m.tok.decode([t]) for t in rare[:5]]}')
        print(f'  _last_inject_diag.is_cond_path={diag_any.get("is_cond_path")}  (expected False after uncond inject)')
        print(f'  _last_cond_inject_diag.is_cond_path={diag_cond.get("is_cond_path")}  (expected True)')

        for label, ts in [("_last_tail_slots (shared)", ts_any),
                          ("_last_cond_tail_slots (new)", ts_cond)]:
            if ts is None:
                print(f'  {label}: None'); continue
            s = ts[0, 1].float()  # slot_1
            print(f'  {label}: slot_1 L2={s.norm().item():.4f}')
            sc = F.normalize(s - wte_mean, dim=-1, eps=1e-8)
            top5 = (wte_centered @ sc).topk(5).indices.tolist()
            sims = wte_centered @ sc
            top20 = sims.topk(20).indices.tolist()
            top5_pieces = [m.tok.decode([t]) for t in top5]
            inter = set(top20) & set(rare)
            print(f'    top5 pieces: {top5_pieces}')
            print(f'    top20 ∩ rare_dom = {inter}  (size={len(inter)})')
            if rare:
                order = sims.argsort(descending=True)
                for rid in rare[:3]:
                    pos = (order == rid).nonzero(as_tuple=True)[0]
                    if pos.numel():
                        piece = m.tok.decode([rid]).strip()
                        print(f"    rank of '{piece}' (id={rid}) = {int(pos.item())+1}")
        print(f'  _last_cond_residual L2={res_cond.norm().item():.4f}' if res_cond is not None else '  _last_cond_residual: None')

if __name__ == "__main__":
    main()
