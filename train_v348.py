#!/usr/bin/env python3
"""Training driver for v3.48 (attention-sharing mechanisms 1+2+3+4).

All mechanisms operate outside the SUT code:

  [M1] Cfg(use_memory_context_encoder=False)
       Disables the random-orthogonal projection and routes context slots
       through mem.semantic_emb (frozen-Qwen attention pool). Zero trainable
       encoder -> zero training convergence cost on that subchannel.

  [M2] Qwen layer-0 K/V warm-start into QFormer layer-0 cross-attention.
       Qwen 2.5 1.5B uses GQA with 256-dim K/V heads. QFormer MHA expects
       1536-dim K/V. We tile Qwen's (256, 1536) k_proj / v_proj six times
       along dim 0 to reach (1536, 1536), giving the bridge a semantically
       informed prior. Q is directly copyable (both 1536x1536).

  [M3] Distillation loss: after each Trainer.step, run a short attention-pool
       alignment: force bridge.proj(f).mean(1) to match Qwen's content-token
       hidden_mean of the SAME text. Uses a second optimizer restricted to
       bridge.proj parameters so it doesn't fight with Trainer's global step.

  [M4] bridge.proj.q learnable-query pool-init: at startup, set each of the
       L_mem learned queries to Qwen's content-token hidden_mean of a random
       corpus sentence, plus small noise. Replaces randn*0.02.

Also: loss reweighting per the earlier "Fix 1/2" plan, so training gradient
flows primarily through the prefix attention path.

Deterministic execution: enables single-threaded torch + deterministic
algorithms before SUT import.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse, time, json, math, sys
import torch
torch.set_num_threads(1)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scheme_b_v344 as sb


MUSIC = [
    "He practiced piano for hours perfecting a difficult Chopin nocturne.",
    "She studied music theory and harmonic progression at the conservatory.",
    "The orchestra performed Beethoven symphony with remarkable precision.",
    "The pianist practiced arpeggios and Chopin nocturnes until midnight.",
    "A musician refined finger technique, phrasing, and pedal control on the piano.",
    "Classical interpretation often depends on dynamics, tempo rubato, and touch.",
]
SPACE = [
    "The telescope revealed distant galaxies beyond the Milky Way.",
    "Astronauts trained for the Mars mission in simulated zero gravity.",
    "The nebula emitted radiation across the electromagnetic spectrum.",
    "Astronomers observed distant galaxies, quasars, and stellar evolution in deep space.",
    "Orbital mechanics explains how satellites and planets move under gravitational force.",
    "Cosmology studies dark matter, expansion, and the large scale structure of the universe.",
]
COOKING = [
    "A chef braised short ribs with red wine, rosemary, and garlic for four hours.",
    "The pastry batter folded egg whites into melted chocolate before baking.",
]
FINANCE = [
    "Portfolio managers rebalance allocations across equities, bonds, and commodities quarterly.",
    "Derivative contracts hedge currency exposure in multinational corporate treasury operations.",
]
ALL = MUSIC + SPACE + COOKING + FINANCE


def build_cfg():
    """[M1] + loss reweighting."""
    return sb.Cfg(
        # [M1] disable learned context encoder -> fallback to mem.semantic_emb
        use_memory_context_encoder=False,
        # loss reweighting: push prefix subchannel, dampen decode-side hints
        loss_weights={
            "recon":                 1.0,
            "semantic_alignment":    1.0,    # was 3.0 (vocab_proj, Layer 2)
            "encoder_throughput":    3.0,    # was 1.5 (Layer 1 LM-CE)  <- UP
            "contrast":              0.02,
            "holonomy":              0.005,
            "write_policy":          0.1,
            "semantic_probe":        0.3,
            "dir_diversity":         0.1,
            "reranker_ranking":      0.2,
            "vocab_anchor":          0.4,    # was 0.2 <- UP
            "tail_semantic_anchor":  0.1,    # was 0.5 (Layer 2) <- DOWN
            "functional_suppression":0.1,    # was 0.4 (Layer 2) <- DOWN
            "context_separation":    0.0,    # [M1] irrelevant without encoder
        },
    )


def mechanism_2_warm_start(m):
    """Copy Qwen layer-0 q/k/v projections into QFormer layer-0 cross-attention.

    Shapes:
      Qwen (GQA):  q_proj (1536, 1536)  k_proj (256, 1536)  v_proj (256, 1536)
      QFormer MHA: in_proj_weight (4608, 1536) = [Q; K; V] each (1536, 1536)

    K/V strategy: tile Qwen's (256, 1536) six times along dim 0 to reach
    (1536, 1536). This seeds the 4 cross-attention heads (each 384-dim) with
    Qwen-learned K/V subspace geometry rather than randn*sqrt(2/fanin).
    """
    if len(m.bridge.proj.layers) == 0:
        return {"applied": False, "reason": "bridge.proj.layers empty"}
    qwen_l0 = m.backbone.model.model.layers[0].self_attn
    qf_ca0 = m.bridge.proj.layers[0].ca
    d = m.c.d_LLM
    with torch.no_grad():
        qw = qwen_l0.q_proj.weight.detach().float()    # (1536, 1536)
        kw = qwen_l0.k_proj.weight.detach().float()    # (256, 1536)  in GQA
        vw = qwen_l0.v_proj.weight.detach().float()    # (256, 1536)
        if qw.shape != (d, d):
            return {"applied": False, "reason": f"q_proj shape {tuple(qw.shape)} != ({d},{d})"}
        tile_k = d // kw.shape[0]
        if kw.shape[1] != d or kw.shape[0] * tile_k != d:
            return {"applied": False,
                    "reason": f"k_proj shape {tuple(kw.shape)} not divisible by d={d}"}
        k_tiled = kw.repeat(tile_k, 1)      # (tile_k * 256, 1536)
        v_tiled = vw.repeat(tile_k, 1)
        # Assemble [Q; K; V]
        new_in_proj = torch.cat([qw, k_tiled, v_tiled], dim=0)  # (4608, 1536)
        assert new_in_proj.shape == qf_ca0.in_proj_weight.shape, (
            f"{new_in_proj.shape} vs {qf_ca0.in_proj_weight.shape}")
        qf_ca0.in_proj_weight.data.copy_(new_in_proj.to(qf_ca0.in_proj_weight.dtype))
        if qf_ca0.in_proj_bias is not None:
            qb = (qwen_l0.q_proj.bias.detach().float()
                  if qwen_l0.q_proj.bias is not None else torch.zeros(d))
            kb = (qwen_l0.k_proj.bias.detach().float().repeat(tile_k)
                  if qwen_l0.k_proj.bias is not None else torch.zeros(d))
            vb = (qwen_l0.v_proj.bias.detach().float().repeat(tile_k)
                  if qwen_l0.v_proj.bias is not None else torch.zeros(d))
            qf_ca0.in_proj_bias.data.copy_(
                torch.cat([qb, kb, vb]).to(qf_ca0.in_proj_bias.dtype))
    return {
        "applied": True,
        "qwen_q_shape": tuple(qw.shape),
        "qwen_k_shape": tuple(kw.shape),
        "tiled_to_d_llm": d,
        "tile_factor_k": tile_k,
    }


def mechanism_4_pool_init_queries(m, corpus_texts, seed=99):
    """[M4] Initialize bridge.proj.q (shape [L_mem, d_LLM]) by pooling Qwen
    content-token hidden_mean over randomly selected corpus texts.

    Replaces the default nn.Parameter(randn * 0.02). Each of the L_mem slots
    gets a distinct pool direction + small noise, so the learned queries start
    in Qwen's representation manifold rather than on the origin sphere.
    """
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    idxs = torch.randperm(len(corpus_texts), generator=g).tolist()[:m.c.L_mem]
    pooled_vecs = []
    device = next(m.parameters()).device
    with torch.no_grad():
        for i in idxs:
            t = corpus_texts[i]
            tk = m.tok(t, return_tensors="pt")
            ids = tk["input_ids"].to(device); mask = tk["attention_mask"].to(device)
            o = m.backbone(ids, mask)
            hs_pooled = m.layer_pool(o["hs"])   # [1, T, d]
            pooled = m._compute_content_semantic_emb(hs_pooled, ids, mask)  # [1, d]
            # Scale to the same distribution as randn*0.02 to avoid over-dominating
            # early training. Later loss will reshape as needed.
            p = pooled[0].float()
            p = p * (0.02 * (p.numel() ** 0.5) / (p.norm().clamp(min=1e-6)))
            pooled_vecs.append(p)
        stacked = torch.stack(pooled_vecs, dim=0)      # [L_mem, d]
        # add small noise to break ties
        noise = torch.randn(stacked.shape, generator=g, device="cpu") * 0.005
        stacked = stacked + noise.to(stacked.device)
        m.bridge.proj.q.data.copy_(stacked.to(m.bridge.proj.q.dtype))
    return {
        "applied": True,
        "init_texts": [corpus_texts[i][:60] for i in idxs],
        "l_mem": m.c.L_mem,
        "pooled_mean_l2": float(stacked.norm(dim=-1).mean().item()),
    }


def mechanism_3_distill_step(m, texts, distill_opt):
    """[M3] One distillation step: push bridge.proj output toward Qwen's
    content-token hidden_mean of the SAME text. Computes loss + backward +
    step on the distill optimizer; does NOT touch the main Trainer.

    bridge.proj is forward-run on the memories' fiber vectors; its per-slot
    output is mean-pooled and MSE'd against Qwen's content-token hidden_mean.
    """
    device = next(m.parameters()).device
    tk = m.tok(texts, return_tensors="pt", padding=True, truncation=True)
    ids = tk["input_ids"].to(device); mask = tk["attention_mask"].to(device)
    with torch.no_grad():
        o = m.backbone(ids, mask)
        hs_pooled = m.layer_pool(o["hs"])
        target = m._compute_content_semantic_emb(hs_pooled, ids, mask)  # [B, d]
        surp = m.amm.surprise_proxy(o["logits"][:, :-1], ids[:, 1:])
    # fiber vector (trainable path)
    pooled = m.layer_pool(o["hs"]).mean(1)
    base = m.amm.ctx(pooled)
    fiber = m.amm.fib(pooled, base, surp)                     # [B, d_F]
    # bridge.proj expects [B, C, d_F]
    fiber_in = fiber.unsqueeze(1)
    mem_mask = torch.ones(fiber_in.shape[0], 1, device=device)
    prefix = m.bridge.proj(fiber_in, mem_mask=mem_mask)       # [B, L_mem, d]
    prefix_pooled = prefix.mean(1)                            # [B, d]
    # Align (cos sim maximized == -cos minimized)
    tn = F.normalize(target.float(), dim=-1, eps=1e-8)
    pn = F.normalize(prefix_pooled.float(), dim=-1, eps=1e-8)
    cos_loss = -(tn * pn).sum(dim=-1).mean()                  # [-1, 1] -> minimize -cos
    # Also L2 match to pull magnitude toward target.
    mse_loss = F.mse_loss(prefix_pooled.float(), target.float()) * 1e-4
    loss = cos_loss + mse_loss
    distill_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for n, p in m.named_parameters()
         if p.requires_grad and n.startswith("bridge.proj")], 1.0)
    distill_opt.step()
    return float(loss.item()), float(cos_loss.item()), float(mse_loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--distill_every", type=int, default=1,
                    help="distill steps per Trainer step (mechanism 3)")
    ap.add_argument("--distill_lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default="ckpt/v348_stacked.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log", type=str, default="ckpt/v348_train_log.jsonl")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.manual_seed(args.seed)

    c = build_cfg()
    print(f"[build] Cfg: use_memory_context_encoder={c.use_memory_context_encoder}")
    print(f"[build] loss_weights[encoder_throughput]={c.loss_weights['encoder_throughput']}")
    print(f"[build] loss_weights[semantic_alignment]={c.loss_weights['semantic_alignment']}")
    m = sb.MemLLM(c)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(dev); m.load(); m.to(dev)
    print(f"[build] device={dev}  params_trainable="
          f"{sum(p.numel() for p in m.parameters() if p.requires_grad):,}")

    # [M2] Qwen layer-0 Q/K/V warm-start into QFormer cross-attention.
    info_m2 = mechanism_2_warm_start(m)
    print(f"[M2 warm-start] {info_m2}")

    # [M4] Pool-init learnable queries.
    info_m4 = mechanism_4_pool_init_queries(m, ALL, seed=args.seed)
    print(f"[M4 pool-init] init_texts[0]={info_m4.get('init_texts',[None])[0]!r}  "
          f"pooled_mean_l2={info_m4.get('pooled_mean_l2'):.3f}")

    # Memory warm-up
    for t in ALL:
        m.write(t, training_mode=True)
    m._refresh_rare_keyword_indices()
    print(f"[build] memories stored: {len(m.amm.tree.store)}")

    trainer = sb.Trainer(m, c)

    # [M3] Second optimizer, only over bridge.proj params.
    distill_params = [p for n, p in m.named_parameters()
                      if p.requires_grad and n.startswith("bridge.proj")]
    distill_opt = torch.optim.AdamW(distill_params, lr=args.distill_lr, weight_decay=0.01)
    print(f"[M3 distill opt] params={sum(p.numel() for p in distill_params):,}  "
          f"lr={args.distill_lr}")

    log_f = open(args.log, "w")
    t0 = time.time()
    for step in range(args.steps):
        start = (step * args.batch) % len(ALL)
        batch = (ALL + ALL)[start:start + args.batch]
        ts = time.time()
        try:
            out = trainer.step(batch)
        except Exception as e:
            print(f"[step {step}] TRAINER ERROR {type(e).__name__}: {e}")
            break
        # [M3] distill alignment step(s)
        distill_info = None
        for _k in range(args.distill_every):
            try:
                dl, dcos, dmse = mechanism_3_distill_step(m, batch, distill_opt)
                distill_info = {"loss": dl, "cos": dcos, "mse": dmse}
            except Exception as e:
                print(f"[step {step}] DISTILL ERROR {type(e).__name__}: {e}")
                break
        dt = time.time() - ts
        rec = {
            "step": step, "dt": dt,
            "trainer_total": out["total"], "recon": out["recon"],
            "encoder_throughput": out["encoder_throughput"],
            "semantic_alignment": out["semantic_alignment"],
            "tail_semantic_anchor": out["tail_semantic_anchor"],
            "functional_suppression": out["functional_suppression"],
            "vocab_anchor": out["vocab_anchor"],
            "distill": distill_info,
        }
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if step % 10 == 0 or step == args.steps - 1:
            d_str = (f"  d_loss={distill_info['loss']:+.3f} "
                     f"cos={-distill_info['cos']:+.3f}"
                     if distill_info else "")
            print(f"[step {step:>3} | {dt:5.1f}s] "
                  f"tot={out['total']:.3f} recon={out['recon']:.3f} "
                  f"et={out['encoder_throughput']:.3f} "
                  f"va={out['vocab_anchor']:.3f}{d_str}")
    elapsed = time.time() - t0
    total_steps = step + 1
    print(f"\n[done] total train time: {elapsed:.1f}s  "
          f"avg/step={elapsed/max(1,total_steps):.1f}s  ({total_steps}/{args.steps} steps)")

    state = {}
    for n, p in m.named_parameters():
        if p.requires_grad and not n.startswith("backbone"):
            state[n] = p.detach().cpu().clone()
    for n, b in m.named_buffers():
        if not n.startswith("backbone"):
            state[n] = b.detach().cpu().clone()
    torch.save({
        "state_dict": state,
        "steps": total_steps,
        "elapsed": elapsed,
        "cfg_version": "v3.48",
        "mechanisms": {"M1": True, "M2": info_m2, "M3": True, "M4": info_m4},
    }, args.out)
    print(f"[done] checkpoint saved: {args.out}  ({len(state)} tensors)")
    log_f.close()


if __name__ == "__main__":
    main()
