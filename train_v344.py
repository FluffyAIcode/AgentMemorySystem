#!/usr/bin/env python3
"""Training driver for v3.44-Trained.

Runs N Trainer.step iterations over a rotating corpus of the same 6 memories
that the audit uses, plus a few generic sentences for context separation.
Saves the non-backbone state_dict to `ckpt/v344_trained.pt`.

Usage:
    python3 train_v344.py --steps 30 --out ckpt/v344_trained.pt
"""
import argparse, os, time, json, math, sys
import torch
# make sure we import the v344 SUT, not whatever the redirect points to
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scheme_b_v344 as sb

MUSIC = [
    "He practiced piano for hours perfecting a difficult Chopin nocturne.",
    "She studied music theory and harmonic progression at the conservatory.",
    "The orchestra performed Beethoven symphony with remarkable precision.",
]
SPACE = [
    "The telescope revealed distant galaxies beyond the Milky Way.",
    "Astronauts trained for the Mars mission in simulated zero gravity.",
    "The nebula emitted radiation across the electromagnetic spectrum.",
]
GENERIC = [
    "The pianist practiced arpeggios and Chopin nocturnes until midnight.",
    "A musician refined finger technique, phrasing, and pedal control.",
    "Classical interpretation often depends on dynamics, tempo rubato, and touch.",
    "A conservatory student studied etudes, scales, and expressive keyboard skills.",
    "Distant astronomers observed galaxies quasars and stellar evolution.",
    "Space orbital mechanics explains satellites and planetary motion.",
]
ALL = MUSIC + SPACE + GENERIC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--out", type=str, default="ckpt/v344_trained.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=str, default="ckpt/train_log.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))  # keep default

    c = sb.Cfg()
    print(f"[build] d_LLM={c.d_LLM}  L_mem={c.L_mem}  dampen={c.fwd_path_bias_dampen}")
    m = sb.MemLLM(c)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device); m.load(); m.to(device)
    print(f"[build] device={device}  tok_pad={m.tok.pad_token}")
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"[build] params total={total:,} trainable={trainable:,}")

    # warm the memory store
    for t in ALL: m.write(t, training_mode=True)
    m._refresh_rare_keyword_indices()
    print(f"[build] memories stored: {len(m.amm.tree.store)}")

    trainer = sb.Trainer(m, c)

    # pick minibatches rotating through ALL
    log_f = open(args.log, "w")
    t0 = time.time()
    for step in range(args.steps):
        start = (step * args.batch) % len(ALL)
        end = start + args.batch
        batch = (ALL + ALL)[start:end]  # wrap
        ts = time.time()
        try:
            out = trainer.step(batch)
        except Exception as e:
            print(f"[step {step}] ERROR {type(e).__name__}: {e}")
            break
        dt = time.time() - ts
        # reduce grad_norms to top-5 for log
        gn = out.get('grad_norms', {})
        top5_gn = dict(sorted(gn.items(), key=lambda kv: -kv[1])[:5])
        rec = {
            'step': step, 'dt': dt,
            'total': out['total'], 'recon': out['recon'],
            'semantic_alignment': out['semantic_alignment'],
            'encoder_throughput': out['encoder_throughput'],
            'tail_semantic_anchor': out['tail_semantic_anchor'],
            'functional_suppression': out['functional_suppression'],
            'context_separation': out['context_separation'],
            'vocab_anchor': out['vocab_anchor'],
            'top5_grad_norms': top5_gn,
        }
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        print(f"[step {step:>3} | {dt:5.1f}s] tot={out['total']:.3f} "
              f"recon={out['recon']:.3f} sa={out['semantic_alignment']:.3f} "
              f"et={out['encoder_throughput']:.3f} tsa={out['tail_semantic_anchor']:.3f} "
              f"va={out['vocab_anchor']:.3f} cs={out['context_separation']:.3f}")
    elapsed = time.time() - t0
    print(f"\n[done] total train time: {elapsed:.1f}s  avg/step={elapsed/max(1,step+1):.1f}s")

    # save only non-backbone trainable weights + affected buffers
    state = {}
    for n, p in m.named_parameters():
        if p.requires_grad and not n.startswith('backbone'):
            state[n] = p.detach().cpu().clone()
    for n, b in m.named_buffers():
        if not n.startswith('backbone'):
            state[n] = b.detach().cpu().clone()
    torch.save({
        'state_dict': state,
        'steps': args.steps,
        'elapsed': elapsed,
        'cfg_version': 'v3.44',
    }, args.out)
    print(f"[done] checkpoint saved: {args.out}  ({len(state)} tensors)")
    log_f.close()


if __name__ == "__main__":
    main()
