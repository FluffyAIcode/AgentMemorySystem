# Sprint Close-Out · v3.46 · fresh-init ceiling reached, training path blocked on GPU

**Handoff from**: CPU-only cloud agent on VM without GPU
**Handoff to**: Cloud agent with GPU-enabled instance type
**Current branch**: `AgentMemory/v346-revertE-topk-nonexclusive-7e97`
**Current audit score**: **21/26** (elapsed 1456 s on CPU, fresh init, `AMS_DETERMINISTIC=1`)
**Runner contract**: `v331_blackbox_eval.py` at v3.49 rev (4.24 substitution ban active)

> This document is the full context for a new agent to pick up. Read this first, then read `V331_BLACKBOX_TEST_SPEC.md`, then the latest two SUT versions (`scheme_b_v344.py`, `scheme_b_v343.py` for comparison). Do not re-audit older versions — their numbers are in `reports/`.

---

## 1. Where we are

### 1.1 SUT architecture (current HEAD of `v346` branch)

`AgentMemorySystem.py` re-exports `scheme_b_v344.py`. Active channel mechanisms:

| Mechanism | Cfg flag | State on v3.46 | Audit carrier |
|---|---|---|---|
| [A] attention-pool `MemoryContextEncoder` | `context_encoder_use_attention_pool: True`, `context_encoder_residual_weight: 0.3` | Active, structural replacement of the v3.42 single-Linear encoder | 4.24 (`0.9375 / 1.000`) |
| [B] residual-dominant tail slot | `tail_slot_residual_dominant: False` | **Reverted** in v3.45 — combine path had `β·LN(head) L2=11.76` vs `α·residual L2=1.07`, head dominated direction | 4.23 + 4.25 |
| [C] inter-domain margin + retrieval crowding | `use_inter_domain_margin: True`, `retrieval_crowding_lambda: 0.15`, `inter_domain_kmeans_k: 2` | Active; write-time KMeans clusters on `semantic_emb` + retrieve-time λ penalty on cross-cluster entries | 4.16 (`retrieval_miss: 0`) |
| [D] refresh rare_keyword_ids on write | `MemLLM.write()` calls `self._refresh_rare_keyword_indices()` | Active | 4.13 (bit-identical greedy outputs) |
| [E] top1-exclusive content_bias | `use_top1_exclusive_content_bias: False` | **Reverted** in v3.46 — was metric-directed overfit, [C] already carried 4.16 | — |
| [F] circuit breaker | `use_circuit_breaker: True`, clamps `mixture_gate` ceiling | Present but dead-path: `use_mixture_decoding: False` by default | none |
| cond-buffer mirror on `EmbBridge` | `_last_cond_{tail_slots, residual, context_slot, ...}` | Active; `inject(is_cond_path=False)` from `_build_contrastive_uncond_prefix` so uncond inject does not clobber diagnostic buffers | 4.23 measurability |

Untouched through the sprint: attention-pool `QFormer` (`bridge_heads=4, bridge_layers=2`), `L_mem=8`, `d_LLM=1536` (Qwen 2.5-1.5B-Instruct), `d_ctx=128`, `tau=0.07`, CFG decoding (`use_cfg_decoding=True, cfg_scale=3.5`), all repetition/suppression guards.

### 1.2 v3.46 audit table (fresh-init, PR #27, reports/v346_revertE_blackbox/)

**PASS (21)**: 4.1 leaf_capacity, 4.2 degenerate_direction_boundary, 4.3 metric_trainability, 4.4 no_grad_generation, 4.5 counterfactual_memory_influence, 4.6 semantic_memory_grounding, 4.10 prefix_logit_drift_audit, 4.12 repetition_segment_audit, 4.13 save_load_consistency, 4.14 training_cache_isolation, 4.15 prefix_stepwise_drift_trajectory, 4.16 retrieval_generation_alignment_audit, 4.17 retrieval_prefix_decode_correlation_audit, 4.18 cheating_heuristics, 4.20 rerank_stability_probe, 4.22 functional_token_suppression_probe, 4.23 keyword_specific_tail_slot_probe, 4.24 context_descriptor_cluster_probe, 4.25 prefix_length_scaling_probe, 4.26 mixture_distribution_gate_probe, 4.9 prompt_diversity_without_memory.

**FAIL (5) — all identified as pre-training axis-C/D gaps**:

| Case | Metric | Observed | Threshold | Failure class |
|---|---|---|---|---|
| 4.7 semantic_memory_counterfactual_pairs | `music_margin > 0 AND space_margin > 0` | 0, 0 | > 0 | axis C |
| 4.8 degeneration_quality | `avg_unique_token_ratio >= 0.35` | 0.343 | ≥ 0.35 (diff 0.007) | axis C |
| 4.11 retrieval_topk_semantic_shift | any keyword in top-12 | 0 hits | ≥ 1 | axis C |
| 4.19 stepwise_label_mass_alignment_audit | no `retrieve`/`inject` stage | mis-aligned | 0 | axis C (cascade of 4.11) |
| 4.21 decode_repetition_feedback_probe | `avg_max_repeat ≤ 3.0` | 4.67 | ≤ 3 (diff 1.67) | axis D |

### 1.3 Axis coverage (v3.49 runner reporting, Section 4-meta.1)

| Axis | Metric | Status |
|---|---|---|
| A compression | `stored_floats/raw_floats = 1712 / 15360 = 8.97`, threshold 10.0 | **FAIL** |
| B injection cost | `L_mem * d_LLM + V = 164224` per step, `depends_on_N = False` | **PASS** |
| C fidelity | 8/11 case-level PASS, threshold K=9 | **FAIL** (gap = 1 pre-training case away from threshold) |
| D stability | 2/3 case-level PASS (4.21 only FAIL), threshold all-pass | **FAIL** (1 case pre-training) |

Axis A is structurally capped by per-memory `semantic_emb (d_LLM=1536 floats)` dominating the per-memory byte count. Known lever (not in current path): `kakeya_codec.py` — dormant legacy code from v3.12, not imported by the SUT or runner. Activating it is a separate architecture question, **not** a pre-training issue.

---

## 2. What changed during this sprint (audit-level, most recent first)

| Version | Branch | Audit | Delta | Core change |
|---|---|---|---|---|
| v3.46 | `AgentMemory/v346-revertE-topk-nonexclusive-7e97` | 21/26 | 0 | Revert [E] (one-line Cfg) |
| v3.45-cond-buffer | `AgentMemory/v345-bridge-cond-buffer-7e97` | 21/26 | +1 | Add `_last_cond_*` mirror on `EmbBridge`; runner reads cond-preferred buffer for 4.23 |
| v3.45-revertB-refreshD | `AgentMemory/v345-revertB-refreshD-7e97` | 20/26 | +2 | Revert [B] LN-dominated tail; add `_refresh_rare_keyword_indices()` in `write()` |
| v3.44-rewrite | `AgentMemory/v344-rewrite-abcdef-audit-7e97` | 18/26 | −1 | Six-mechanism rewrite [A]/[B]/[C]/[D]/[E]/[F] — [A]/[C]/[D] validated, [B]/[E]/[F] regressed or dead |
| v3.49 (runner) | `AgentMemory/v349-runner-fallback-revert-7e97` | — | — | Runner: remove 4.24 substitution rule; SPEC §4.24 substitution ban + §7.9 retraction |
| v3.48 stacked | earlier | 19/26 (trained 120 steps) | — | Baseline trained SUT + stacked mechanisms M1–M4 (partially retracted) |

All PRs on GitHub, open/draft: #24, #25, #26, #27.

## 3. Errors the sprint corrected or exposed

### 3.1 Prediction errors I caught only after seeing audit data

1. **v3.44-rewrite**: I predicted `[A]+[B]+[C]+[D] = +4 cases`. Actual: **+2 wins (4.24, 4.16), 3 regressions (4.8, 4.21, 4.25), net −1**. Five predictions wrong, five error categories pinned:
   - Unit mismatch: predicted cosine alignment for a rank metric (4.23).
   - Unit mismatch: predicted parameter-state determinism for token-string equality (4.13).
   - Scope mismatch: predicted retrieval quality fixes lexical-class top-k (4.11/4.19).
   - Magnitude blindness: no L2 calc of `β·LN(head)` vs `α·residual`.
   - Regression blindness: no race calc of `content_bias × scale` vs `repeat_penalty × k`.

2. **v3.46**: I predicted reverting [E] restores 4.7/4.8/4.21 because "v3.48 baseline had them PASS under `top1_exclusive=False`". Actual: **same 21/26**. Error: compared a 120-step trained baseline to a fresh-init revert. Same class as (1).

3. **4.23 aliasing**: Across v3.38 → v3.48 the probe was reading the **uncond-path** tail slots (the second `inject` call in `prepare_decode_context` clobbered `bridge._last_tail_slots`). `median_rank=1089` on v3.48 was noise from the uncond prefix, not a measurement of anything. Closed in v3.45-cond-buffer via an SUT-side mirror buffer.

### 3.2 Mechanism errors the sprint fixed

- **[B] combine_with_residual had inverted magnitude**: `β·LN(head_out) L2=11.76` vs `α·residual L2=1.07`. On fresh init with zero-init `slot_heads[1]`, `LN(0)` reduces to LayerNorm γ direction (uniform `(1,...,1)/√d`), far from any rare-keyword WTE direction. 4.23 `median_rank=1402` (v3.44-rewrite) was caused by this, not by anything the probe tests.

- **[D] rare_keyword_ids fresh/load asymmetry**: `MemLLM.write()` left `MemEntry.rare_keyword_ids = []`; `MemLLM.load_memory()` called `_refresh_rare_keyword_indices()` post-load, so `model_a (fresh+write)` and `model_b (load)` had different rare_keyword_ids → `_compute_rare_keyword_wte_residual` returned `None` on model_a, non-zero on model_b → `prefix_cond` diverged → 4.13 FAILed under byte-level greedy-string equality. Closed by adding the refresh call to `write()`.

- **[E] top1-exclusive was overfit residue**: [C] alone carried 4.16. [E] was concentration on top of correct retrieval that broke the `content_bias` × `content_repeat_penalty` race (bias +22 logit on ~8 tokens, penalty 2.5×k only wins at k≥10, `cyclic_content_max_count=5` hard-masks at k=5). Reverted in v3.46.

- **cond-buffer aliasing**: `prepare_decode_context` calls `bridge.inject` twice. Uncond call with `rare_keyword_wte_residual=None` overwrote `_last_tail_slots` before the audit probe could read the cond-path value. Closed by adding `is_cond_path` kwarg to `inject` and `_last_cond_*` mirror fields to `EmbBridge`. Runner now reads `_last_cond_tail_slots` preferentially.

### 3.3 Anti-patterns to NOT repeat

1. **Threshold-chasing parameter tuning**: picking Cfg values because they satisfy a specific audit threshold (`≤ 3.0`), with no independent structural reason. Examples: v3.44-rewrite [E] (`w=0.7, floor=0.5`), the v3.45 plan's #2 (tune `content_bias_scale=4.0, repeat_penalty=3.0` to let penalty win at k=3). These earn SPEC §1.1.3 borderline overfit, even if not outright `prompt-keyed routing`.

2. **Decode-time metric patching** (e.g. plan #5 `rare_keyword_floor_boost`): force top-k to contain specific tokens at decode time to bypass the channel's inability to transmit them through embedding space. 4.11/4.19 asked "does the channel carry lexical content end-to-end?" — metric-patching gives a yes by editing the endpoint. Strong overfit. Ruled out.

3. **Mechanism hooked to dead Cfg path** (e.g. [F] circuit breaker on `use_mixture_decoding=False`): always verify the mechanism's gating consumer is live before claiming it does anything.

---

## 4. What is not solvable on CPU / without training

Five remaining FAILs (4.7 / 4.8 / 4.11 / 4.19 / 4.21) share one root cause: **fresh init has two zero-initialized paths that dilute `content_bias` concentration across the vocabulary during decode**:

```python
# scheme_b_v344.py, ContentSemanticTailHead.__init__
if tied_extra and zero_init_tied and i == 1:
    nn.init.zeros_(head[0].weight); nn.init.zeros_(head[0].bias)

# scheme_b_v344.py, MemoryVocabProjector.__init__
nn.init.zeros_(self.proj[-1].weight); nn.init.zeros_(self.proj[-1].bias)
```

Both produce `0` on fresh init. So:

- `tail_head(fiber)` → `0` on slot_1 → slot carries only `α·residual` (a per-memory constant direction, identical across prompts from the same corpus)
- `vocab_proj(fiber_summary, wte)` → `0` → `lg += vocab_bias × semantic_boost_scale` contributes nothing to decode-time logits

The only **live** memory-side signal at decode is the aggregated `content_bias` over top-k retrieved memories' content tokens. On the music + space + general corpus this is a ~12-token set. `content_repeat_penalty = 2.5 × k` catches up at `k ≥ 6–7`, but `cyclic_content_max_count = 5` hard-masks by then, and 20-step generation locks into the repetition. Outputs look like:

```
The pianist pian pian midnight Pell pian Ell night pian noct midnight practiced midnight midnight pianian
The pianist pian piano pian pian hours pian Tao pian perfect hours hours perfectperfectAppPerfectSoftware
```

The `tail_head + vocab_proj` paths exist **precisely** so that training can populate them with distributed vocabulary contributions that dilute this concentration. Without training they're 0. This is pre-training gap, not a Cfg bug. Plan #5's `rare_keyword_floor_boost` was a handcrafted substitute for what training should produce — hence the overfit verdict.

For 4.11 / 4.19 specifically: the prefix signal attenuates through 28 Qwen layers to ~0.14 logit on any rare-keyword direction (estimated `1/√28 ≈ 0.19` per-layer average), far below the ~10 logit modal-transition baseline. Training the QFormer and bridge moves activation magnitudes into ranges where the final-layer logit geometry shows domain-keyword lift.

---

## 5. Next step for the GPU-enabled agent

### 5.1 Immediate objective

Train the v3.46 SUT for 60–100 steps on GPU, save checkpoint, re-audit, compare to v3.46 fresh-init baseline (21/26) and v3.48 stacked (19/26).

### 5.2 Infrastructure needed

- `torch.cuda.is_available() == True`
- Qwen/Qwen2.5-1.5B-Instruct can load in bf16 (needs ~3 GB VRAM for model + 2–4 GB for Trainer's activations; a single 16 GB GPU is comfortable)
- `transformers`, `torch` already in the repo's env — verify with `python3 -c "import torch, transformers"`

### 5.3 Implementation (copy `train_v344.py` as starting point)

Create `train_v346.py` in the repo root:

```python
#!/usr/bin/env python3
"""Training driver for v3.46-trained.

Starts from v346-revertE-topk-nonexclusive-7e97 SUT (attention-pool ctx encoder,
cluster-crowding retrieval, refresh-on-write, additive tail residual,
top1-exclusive OFF, cond-buffer mirror).  Runs N Trainer.step iterations
over a rotating corpus; saves non-backbone state_dict to ckpt/v346_trained.pt.
"""
import argparse, os, time, json, math, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scheme_b_v344 as sb

MUSIC = [  # same as train_v344.py
    "He practiced piano for hours perfecting a difficult Chopin nocturne.",
    "She studied music theory and harmonic progression at the conservatory.",
    "The orchestra performed Beethoven symphony with remarkable precision."]
SPACE = [
    "The telescope revealed distant galaxies beyond the Milky Way.",
    "Astronauts trained for the Mars mission in simulated zero gravity.",
    "The nebula emitted radiation across the electromagnetic spectrum."]
GENERIC = [
    "The pianist practiced arpeggios and Chopin nocturnes until midnight.",
    "A musician refined finger technique, phrasing, and pedal control.",
    "Classical interpretation often depends on dynamics, tempo rubato, and touch.",
    "A conservatory student studied etudes, scales, and expressive keyboard skills.",
    "Distant astronomers observed galaxies quasars and stellar evolution.",
    "Space orbital mechanics explains satellites and planetary motion."]
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
    torch.manual_seed(args.seed)

    c = sb.Cfg()
    m = sb.MemLLM(c)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "train_v346 expects CUDA; CPU fallback is 10x slower and not the intent"
    m.to(device); m.load(); m.to(device)
    print(f"[build] device={device}  params trainable={sum(p.numel() for p in m.parameters() if p.requires_grad):,}")

    # write initial corpus so retrieval has something to return during training
    for t in ALL:
        m.write(t, training_mode=True)
    m.amm.maybe_recluster(force=True)
    m._refresh_rare_keyword_indices()
    m.eval()
    print(f"[build] initial memory count = {len(m.amm.tree.store)}")

    trainer = sb.Trainer(m, c)
    print(f"[train] optimizer AdamW lr=1e-4 wd=0.01  batch={args.batch}  steps={args.steps}")

    with open(args.log, "w") as flog:
        for step in range(args.steps):
            # rotate batch
            start = (step * args.batch) % len(ALL)
            batch = [ALL[(start + i) % len(ALL)] for i in range(args.batch)]
            t0 = time.time()
            try:
                stats = trainer.step(batch)
            except Exception as e:
                print(f"[step {step}] EXCEPTION: {e}")
                raise
            dt = time.time() - t0
            # small loss log
            tot = stats.get("total")
            print(f"step {step:3d}  total={tot:.4f}  "
                  f"recon={stats.get('recon', 0):.3f}  "
                  f"sa={stats.get('semantic_alignment', 0):.3f}  "
                  f"tsa={stats.get('tail_semantic_anchor', 0):.3f}  "
                  f"va={stats.get('vocab_anchor', 0):.3f}  "
                  f"dt={dt:.1f}s")
            flog.write(json.dumps({"step": step, "dt_s": dt,
                                    **{k: v for k, v in stats.items()
                                       if k not in ('grad_norms', 'loss_weights')}},
                                   ensure_ascii=False) + "\n")
            flog.flush()

    # save non-backbone state_dict + memory store
    sd = {n: p.detach().cpu() for n, p in m.named_parameters() if 'backbone' not in n}
    torch.save({
        "state_dict": sd,
        "cfg_snapshot": {k: getattr(c, k) for k in ("L_mem","d_ctx","d_M","d_F","cfg_scale",
                                                     "use_top1_exclusive_content_bias",
                                                     "tail_slot_residual_dominant",
                                                     "use_inter_domain_margin",
                                                     "context_encoder_use_attention_pool")},
        "provenance": "AgentMemory/v346-revertE-topk-nonexclusive-7e97",
    }, args.out)
    print(f"[save] wrote {args.out}  state_dict keys={len(sd)}")

if __name__ == "__main__":
    main()
```

Also add a `_load_trainable_weights` checkpoint loader to the SUT (there's a pattern in `scheme_b_v344.py` — check `MemLLM.load` for the existing `AMS_TRAINED_WEIGHTS` hook; if absent on current branch, add it mirroring the v3.44-Trained hook).

### 5.4 Training rules to enforce

- **Do NOT modify `Cfg`** during or after training. The Cfg on v3.46 is what the audit will evaluate; changing it between training and audit invalidates the comparison.
- **Do NOT modify the Trainer loss family** (`recon`, `semantic_alignment`, `tail_semantic_anchor`, `vocab_anchor`, `functional_suppression`, `context_separation`, `slot_residual_alignment` **weight is 0**, `inter_domain_margin`). Adding new losses to target specific probes would re-enter the overfit zone.
- `slot_residual_alignment` weight is intentionally `0.0` on v3.46 (Cfg revert in v3.45). Leave it at 0 — it targeted the [B]-reverted path.
- Use `batch=3`, `steps=60` as starting point; scale to `steps=100` if 60 leaves `semantic_alignment` / `tail_semantic_anchor` still decreasing.
- Checkpoint goes to `ckpt/v346_trained.pt`; git-ignore per existing `.gitignore` pattern.

### 5.5 Audit protocol after training

```bash
# from AgentMemory/v346-revertE-topk-nonexclusive-7e97 (or a child branch)
export AMS_DETERMINISTIC=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export AMS_TRAINED_WEIGHTS=ckpt/v346_trained.pt   # SUT's MemLLM.load() will pick this up
rm -rf reports/v331_blackbox
python3 v331_blackbox_eval.py 2>&1 | tee reports/v346_trained_blackbox/stdout.log
cp reports/v331_blackbox/report.json  reports/v346_trained_blackbox/report.json
cp reports/v331_blackbox/report.md    reports/v346_trained_blackbox/report.md
```

### 5.6 Honest predictions — **do not re-make my mistake**

The lesson from v3.44-rewrite and v3.46 audits: **do not predict a Δ pass count before seeing the data**. Instead, predict **mechanism-level observable changes** (per SPEC §7.7 norm):

- `tail_head.slot_heads[1].0.weight.abs().mean()` should move from 0 to some non-zero value after training (call it `ε_tail`). Sanity range: `1e-4 ≤ ε_tail ≤ 1e-2`.
- `vocab_proj.proj[-1].weight.abs().mean()` should move from 0 to `ε_vocab`. Same range.
- These are necessary conditions for 4.7/4.8/4.11/4.19/4.21 to have any hope. They do not guarantee the audit flips. Measure first, commit the audit number as data, then make axis-C / axis-D claims.

### 5.7 If training does not move 4.11 / 4.19

Even post-training, 4.11/4.19's fundamental obstruction (prefix signal through 28 Qwen layers) may still hold. In that case:

- Option A: **accept** the `8/11 axis-C` outcome and report under SPEC §7.7 as "pre-amplification gap — channel carries context descriptor (4.24) and retrieval routing (4.16) but does not achieve next-token top-k amplification under the current bridge depth/width."
- Option B: amend SPEC §4.11/§4.19 to explicitly allow a decode-time `vocab_bias` channel as a legitimate mechanism (similar to how content_bias is already legitimate per §1.1.2 item 2). This is a SPEC change, not a SUT change, and requires user approval.

Do not implement `rare_keyword_floor_boost` or any equivalent mechanism — the overfit analysis in sprint is on record (PR #27 body + `reports/v346_revertE_blackbox/`).

---

## 6. Files and artifacts

```
/workspace/
├── AgentMemorySystem.py                   # re-exports scheme_b_v344
├── scheme_b_v344.py                       # current SUT source (v3.46 Cfg)
├── v331_blackbox_eval.py                  # v3.49 runner
├── V331_BLACKBOX_TEST_SPEC.md             # current spec
├── train_v344.py                          # template for training driver
├── train_v348.py                          # stacked-mechanisms driver (reference only)
├── ckpt/                                  # git-ignored *.pt
│   ├── v344_trained.pt                    # (incompatible with current SUT, different param shapes)
│   └── v348_stacked.pt                    # (incompatible)
├── reports/
│   ├── v344_rewrite_blackbox/             # 18/26 fresh
│   ├── v345_revertB_refreshD_blackbox/    # 20/26 fresh
│   ├── v345_condbuffer_blackbox/          # 21/26 fresh
│   └── v346_revertE_blackbox/             # 21/26 fresh  ← current ceiling
├── diag_4_13_rare_keyword_equiv.py        # write/load bit-equality assertion
├── diag_4_23_cond_buffer.py               # _last_cond_tail_slots verification
├── diag_4_23_slot_direction.py            # _last_residual / tail_head buffer trace
└── SPRINT_CLOSEOUT_v3.46.md               # this file
```

Existing checkpoints `ckpt/v344_trained.pt` and `ckpt/v348_stacked.pt` were trained against **different parameter shapes** (v3.42 had a single-Linear ctx encoder; v3.48 added `M3` distillation loss and `M2` K/V warm-start). Loading them into the v3.46 SUT will fail a `state_dict` shape assertion — **train fresh on v3.46**.

---

## 7. Open PRs

| PR | Branch | Status | Contents |
|---|---|---|---|
| #23 | v349-runner-fallback-revert | draft | Runner + SPEC: 4.24 substitution ban |
| #24 | v344-rewrite-abcdef-audit | draft | v3.44 six-mechanism rewrite + 18/26 audit |
| #25 | v345-revertB-refreshD | draft | Revert [B], refresh timing, 20/26 audit |
| #26 | v345-bridge-cond-buffer | draft | cond-buffer aliasing fix, 21/26 audit |
| #27 | v346-revertE-topk-nonexclusive | draft | **Current head.** Revert [E], 21/26 audit |

New agent should create a child branch off #27's branch (or merge #27 first per user call) before starting `train_v346.py`. Suggested branch name: `AgentMemory/v346-trained-{suffix}-7e97`.

---

## 8. Sanity prompts for the next agent before any training run

Run these in order. Each should succeed without intervention:

1. `git status` → working tree clean, branch = `AgentMemory/v346-...`
2. `python3 -c "import torch; assert torch.cuda.is_available(), 'GPU must be available'"` — this is the check that blocked this sprint
3. `python3 -c "import scheme_b_v344 as sb; c = sb.Cfg(); assert c.use_top1_exclusive_content_bias is False; assert c.tail_slot_residual_dominant is False"` — confirm v3.46 Cfg
4. `python3 diag_4_23_cond_buffer.py` — should report `rank of ' control' = 1` on both paraphrases (validates cond-buffer plumbing end-to-end on this branch, no training needed)
5. Create `train_v346.py` per Section 5.3, test it **without** saving (set `--steps 2`) to verify Trainer runs clean on GPU
6. Full training run: `python3 train_v346.py --steps 60 --out ckpt/v346_trained.pt`
7. Audit: Section 5.5 protocol
8. Commit, push, create PR against main, update this document with outcome

---

## 9. What this document does NOT say

- **It does not give a Δ pass-count prediction for the trained audit.** That is deliberate. Prior predictions have been wrong 4 consecutive times. The correct move is to run training, report the audit number as data, then reason about it.
- **It does not claim the channel is "working" or "not working".** Per SPEC §7.7 that phrasing is banned. Reports claim specific axes, at specific numerical thresholds, with specific case coverage.
- **It does not instruct on Cfg tuning post-training.** If the trained audit is at some number N ≤ 21, the v3.46 Cfg is what the sprint leaves locked; any Cfg change after the fact re-enters threshold chasing unless it is a revert with structural justification (the pattern that delivered 4.13, 4.23, 4.25).
