#!/usr/bin/env python3
"""Training driver for v3.46-trained.

Starts from v346-revertE-topk-nonexclusive-7e97 SUT (attention-pool ctx encoder,
cluster-crowding retrieval, refresh-on-write, additive tail residual,
top1-exclusive OFF, cond-buffer mirror).  Runs N Trainer.step iterations
over a rotating corpus; saves non-backbone state_dict to ckpt/v346_trained.pt.

Per SPRINT_CLOSEOUT_v3.46.md §5.3 / §5.4.
"""
import argparse, os, time, json, math, sys
import torch
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
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--out", type=str, default="ckpt/v346_trained.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=str, default="ckpt/v346_train_log.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    log_dir = os.path.dirname(args.log) or "."
    os.makedirs(log_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    c = sb.Cfg()
    # Sanity: confirm v3.46 Cfg (same assert as §8 step 3, catches env corruption)
    assert c.use_top1_exclusive_content_bias is False, \
        "Cfg.use_top1_exclusive_content_bias must be False on v3.46"
    assert c.tail_slot_residual_dominant is False, \
        "Cfg.tail_slot_residual_dominant must be False on v3.46 (revert [B])"

    m = sb.MemLLM(c)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        if os.environ.get("AMS_ALLOW_CPU_TRAIN", "0") != "1":
            raise AssertionError(
                "train_v346 expects CUDA; CPU fallback is ~10x slower and not the intent. "
                "Set AMS_ALLOW_CPU_TRAIN=1 to override explicitly.")
        print("[build] WARNING: running on CPU (AMS_ALLOW_CPU_TRAIN=1)")
    m.to(device); m.load(); m.to(device)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m.parameters())
    print(f"[build] device={device}  params total={total:,}  trainable={trainable:,}")

    for t in ALL:
        m.write(t, training_mode=True)
    try:
        m.amm.maybe_recluster(force=True)
    except Exception as e:
        print(f"[build] amm.maybe_recluster skipped: {type(e).__name__}: {e}")
    m._refresh_rare_keyword_indices()
    m.eval()
    print(f"[build] initial memory count = {len(m.amm.tree.store)}")

    # Pre-training mechanism snapshot (per §5.6): tail_head[1] + vocab_proj last weights
    def _probe_weights(model):
        out = {}
        try:
            w = model.bridge.tail_head.slot_heads[1][0].weight
            out["tail_head_slot1_abs_mean"] = float(w.detach().abs().mean())
        except Exception as e:
            out["tail_head_slot1_abs_mean"] = f"ERR {type(e).__name__}"
        try:
            w = model.vocab_proj.proj[-1].weight
            out["vocab_proj_last_abs_mean"] = float(w.detach().abs().mean())
        except Exception as e:
            out["vocab_proj_last_abs_mean"] = f"ERR {type(e).__name__}"
        return out
    pre_probe = _probe_weights(m)
    print(f"[probe pre-train] {pre_probe}")

    trainer = sb.Trainer(m, c)
    print(f"[train] Trainer built  batch={args.batch}  steps={args.steps}")

    t_start = time.time()
    with open(args.log, "w") as flog:
        for step in range(args.steps):
            start = (step * args.batch) % len(ALL)
            batch = [ALL[(start + i) % len(ALL)] for i in range(args.batch)]
            t0 = time.time()
            try:
                stats = trainer.step(batch)
            except Exception as e:
                print(f"[step {step}] EXCEPTION: {type(e).__name__}: {e}")
                raise
            dt = time.time() - t0
            tot = stats.get("total")
            print(
                f"step {step:3d}  total={tot:.4f}  "
                f"recon={stats.get('recon', 0):.3f}  "
                f"sa={stats.get('semantic_alignment', 0):.3f}  "
                f"tsa={stats.get('tail_semantic_anchor', 0):.3f}  "
                f"va={stats.get('vocab_anchor', 0):.3f}  "
                f"fs={stats.get('functional_suppression', 0):.3f}  "
                f"cs={stats.get('context_separation', 0):.3f}  "
                f"dt={dt:.1f}s"
            )
            rec = {"step": step, "dt_s": dt,
                   **{k: v for k, v in stats.items()
                      if k not in ("grad_norms", "loss_weights")}}
            flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
            flog.flush()
    elapsed = time.time() - t_start
    post_probe = _probe_weights(m)
    print(f"[probe post-train] {post_probe}")
    print(f"[train] elapsed {elapsed:.1f}s  avg/step={elapsed/max(1,args.steps):.2f}s")

    sd = {n: p.detach().cpu() for n, p in m.named_parameters() if "backbone" not in n}
    for n, b in m.named_buffers():
        if "backbone" not in n:
            sd[n] = b.detach().cpu()
    torch.save({
        "state_dict": sd,
        "cfg_snapshot": {k: getattr(c, k) for k in (
            "L_mem", "d_ctx", "d_M", "d_F", "cfg_scale",
            "use_top1_exclusive_content_bias",
            "tail_slot_residual_dominant",
            "use_inter_domain_margin",
            "context_encoder_use_attention_pool",
        )},
        "provenance": "AgentMemory/v346-revertE-topk-nonexclusive-7e97",
        "steps": args.steps,
        "elapsed_s": elapsed,
        "pre_probe": pre_probe,
        "post_probe": post_probe,
    }, args.out)
    print(f"[save] wrote {args.out}  tensors={len(sd)}")


if __name__ == "__main__":
    main()
