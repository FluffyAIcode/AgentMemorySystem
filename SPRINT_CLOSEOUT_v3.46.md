# Sprint Close-Out · v3.46 · trained audit complete — 60-step training lowers score by 3

**Handoff from**: CPU-only cloud agent on VM without GPU
**Handoff to (closed)**: GPU-enabled cloud agent via SSH to vast.ai (NVIDIA H200, cu128, torch 2.11.0)
**Current branch**: `AgentMemory/v346-trained-gpu-7e97` (child of `AgentMemory/v346-revertE-topk-nonexclusive-7e97`, PR #28)
**Trained audit score**: **18/26** (elapsed 1250 s on H200, `AMS_TRAINED_WEIGHTS=ckpt/v346_trained.pt`, `AMS_DETERMINISTIC=1`)
**Fresh-init baseline (for delta)**: 21/26 (unchanged, re-listed in §1.2)
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

### 1.4 v3.46-trained audit table (60 training steps on H200, PR #28, reports/v346_trained_blackbox/)

Training run: `python3 train_v346.py --steps 60` — 335 s wall on H200 (≈5.6 s/step), single-GPU, bf16 backbone, 113.8 M trainable non-backbone params, 11 memories stored pre-training. `Cfg` unchanged vs §1.1. Checkpoint: `ckpt/v346_trained.pt`, 455 MB, 202 non-backbone tensors, provenance `AgentMemory/v346-revertE-topk-nonexclusive-7e97`.

**§5.6 mechanism observables (as data, per SPEC §7.7 norm)**:

| Observable | Pre-train | Post-train | §5.6 target range | In range? |
|---|---|---|---|---|
| `bridge.tail_head.slot_heads[1][0].weight.abs().mean()` | `0.0` | `7.30e-4` | `[1e-4, 1e-2]` | yes |
| `vocab_proj.proj[-1].weight.abs().mean()` | `0.0` | `5.49e-4` | `[1e-4, 1e-2]` | yes |

Both necessary conditions named in §5.6 are met. §5.6 explicitly stated this does not guarantee the audit flips — audit data below is the test.

**PASS (18)**: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.10, 4.12, 4.13, 4.14, 4.15, 4.16, 4.18, 4.22, 4.23, 4.24, 4.26, 4.9.

**FAIL (8)**:

| Case | Metric | Observed (trained) | Observed (fresh) | Threshold | Delta |
|---|---|---|---|---|---|
| 4.7 semantic_memory_counterfactual_pairs | `music_margin > 0 AND space_margin > 0` | 0, 0 | 0, 0 | > 0 | unchanged (still axis-C fail) |
| 4.8 degeneration_quality | `avg_unique_token_ratio ≥ 0.35` | 0.296 on "The pianist" (worse than fresh 0.343) | 0.343 | ≥ 0.35 | **regressed** (trained more repetitive) |
| 4.11 retrieval_topk_semantic_shift | any keyword in top-12 | 0 hits | 0 hits | ≥ 1 | unchanged |
| 4.17 retrieval_prefix_decode_correlation_audit | `retrieval_strength__prefix_l2` finite + sign-correct | `null` (prefix_l2_shift=3.22e+11 → variance blew up) | passed | finite | **regression**: trained prefix has extreme L2 shift, correlation undefined |
| 4.19 stepwise_label_mass_alignment_audit | staged alignment ≥ 0 | mis-aligned (decode picks " Options", `stage_counts.decode=2 < inject=6`) | mis-aligned | aligned | unchanged (cascade of 4.11) |
| 4.20 rerank_stability_probe | both pairs jaccard ≥ 0.6 | `space_P2` jaccard=0.429 (spearman 0.961) | passed | both ≥ 0.6 | **regression**: training perturbed retrieval clustering on one prompt pair |
| 4.21 decode_repetition_feedback_probe | `avg_max_repeat ≤ 3.0` | 5.0 (worse than fresh 4.67) | 4.67 | ≤ 3 | regressed |
| 4.25 prefix_length_scaling_probe | `avg_mass_ratio_B_over_A > 1.10` | 0.824 (< 1.0, doubling L_mem *reduces* starter mass) | passed | > 1.10 | **regression**: trained slot weights do not scale positively with L_mem |

**Net: +0 gains, −3 regressions (4.17/4.20/4.25), score 21 → 18.**

Axis coverage (v3.49 runner):

| Axis | v3.46 fresh | v3.46 trained |
|---|---|---|
| A compression | FAIL (8.97 / 10.0) | FAIL (8.97 / 10.0) — structural, unchanged |
| B injection cost | PASS | PASS |
| C fidelity | FAIL (8/11) | FAIL (6/11) |
| D stability | FAIL (2/3) | FAIL (1/3) |

### 1.5 Why 60-step training did not help — structural read

The §5.6 observables moved into range, confirming the zero-init dead paths `tail_head.slot_heads[1]` and `vocab_proj.proj[-1]` did start receiving gradient. But none of the five pre-training FAILs flipped (4.7/4.8/4.11/4.19/4.21), and three previously-passing cases flipped FAIL:

- **4.17**: `prefix_l2_shift = 3.22e+11`. The trained prefix magnitude is ~6 orders of magnitude larger than the baseline hidden-state norm. Something in the training loss (most likely `semantic_alignment` at weight 3.0 against an unconstrained prefix magnitude) drove the injected prefix to saturate — this is consistent with `sa = 9.9 → 9.0` barely moving across 60 steps while producing a prefix with huge norm. The audit's correlation computation drops to `null` when inputs are non-finite or near-constant.
- **4.20**: `space_P2` pair jaccard dropped from ≥0.6 (fresh) to 0.429 (trained). Both prompts still rank `mid=5` first, but the tail of top-5 diverges between paraphrases — the trained retrieval clusters are sharper but more brittle to paraphrase.
- **4.25**: doubling `L_mem` 8→16 decreased starter-positive mass ratio to 0.82 (< 1.10). The trained slots behave anti-correlated with `L_mem`: more slots = more dilution of the starter-direction signal. This is the inverse of what the probe requires.
- **4.21**: `avg_max_repeat_per_content_token` went from 4.67 → 5.0. Training reinforced the corpus-local repetition pattern, making the 4.21 FAIL slightly worse.
- **4.8**: "The pianist" unique-token ratio fell from 0.343 → 0.296. Same class as 4.21.

The shared pattern: `sa` (3.0× weight, reconstruction-anchored to the Qwen embedding space) trained the ctx encoder to push prefix magnitude up without a counterbalancing norm constraint, and the tail/vocab paths gained small weights that reinforce the retrieved memories' own repetitive phrasing rather than distributed vocabulary. 60 steps on a 12-text corpus is too small and too narrow for the Qwen latent-space geometry to develop a dilution signal; it's exactly long enough to overfit the corpus's own repetition. This is the §5.7 **option A** territory (pre-amplification gap under current bridge depth/width + loss family), now confirmed with data rather than predicted.

Two things this sprint **does not** recommend based on this data:

1. Trivially training longer (100–300 steps) on the same 12-text corpus. With no norm regularizer on the prefix and `sa` weight at 3.0, longer training will push `prefix_l2_shift` further up and regress 4.17 more.
2. Adding a prefix-norm regularizer or a decode-time `vocab_bias` amplifier. Both would be threshold-chasing under §3.3 anti-pattern (1) / §5.7 option B without a SPEC amendment.

---

## 2. What changed during this sprint (audit-level, most recent first)

| Version | Branch | Audit | Delta | Core change |
|---|---|---|---|---|
| v3.46-trained | `AgentMemory/v346-trained-gpu-7e97` | 18/26 | **−3** | 60-step train on H200 (train_v346.py §5.3); AMS_TRAINED_WEIGHTS loader added |
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
| #27 | v346-revertE-topk-nonexclusive | draft | Revert [E], 21/26 fresh-init audit (base for #28) |
| #28 | v346-trained-gpu-7e97 | draft | **Current head.** train_v346.py + AMS_TRAINED_WEIGHTS loader + **18/26 trained audit** |

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

---

## 10. Product-viability spike · AMS as session layer (v3.46-session-viability branch, PR #29)

### 10.1 Framing

Blackbox audit measures **single-step prefix-channel fidelity**. It does **not** measure multi-turn cost × quality as a session layer between an application and an LLM. Those are independent questions. Before committing to the P0–P4 blackbox-audit climb, `session_viability.py` quantifies whether current code — no Cfg changes, no loss changes — already makes AMS useful in that product role.

### 10.2 Protocol

- 20-turn synthetic session: 10 facts + 10 targeted-recall queries.
- Hit = `expected_keyword` substring present in the generated answer (case-insensitive).
- Backbone: Qwen2.5-1.5B-Instruct, bf16, HF `model.generate(do_sample=False)` for non-AMS modes; `MemLLM.generate(greedy=True)` for AMS modes.
- Writes use `training_mode=True` so the write-gate never silently drops a fact (measurement setup, not a training claim).
- `max_new_tokens = 30`.

Five modes:

| Mode | Retrieval | Injection | Token growth | AMS path exercised |
|---|---|---|---|---|
| `D_full_history` | none | all 10 facts in prompt | O(N) | bypass (ceiling) |
| `B_flat_cos` | flat cosine over `semantic_emb` | top-3 `source_text` prepended | O(k) | storage only |
| `B_ams_text` | full AMS (`prepare_decode_context` + gate + rerank) | top-3 `source_text` prepended | O(k) | retrieval only |
| `A_ams_prefix` | full AMS | prefix embeddings only (`MemLLM.generate`) | O(1) | full blackbox mechanism |
| `C_ams_hybrid` | full AMS | prefix + top-1 `source_text` | O(1) | full mechanism + minimal text context |

### 10.3 Fresh-init results (CPU, Qwen2.5-1.5B-Instruct, mt=30)

Two store sizes compared, same 10 queries:

**N=10 facts (identity-only, `reports/session_viability_fresh/`)**:

| Mode | Hit-rate | avg in-tokens | avg retrieve ms | avg generate ms |
|---|---:|---:|---:|---:|
| `D_full_history` | **100%** | 159 | 0.0 | 4138 |
| `B_flat_cos` | **80%** | 55 | 144 | 4187 |
| `B_ams_text` | 70% | 56 | 526 | 4030 |
| `A_ams_prefix` | 60% | **11** | 453 | 19722 |
| `C_ams_hybrid` | 70% | 26 | 471 | 21147 |

**N=20 facts (10 identity + 10 distractors, `reports/session_viability_fresh_20facts/`)**:

| Mode | Hit-rate | avg in-tokens | avg retrieve ms | avg generate ms |
|---|---:|---:|---:|---:|
| `D_full_history` | **100%** | **301** | 0.0 | 4590 |
| `B_flat_cos` | 70% (**−10**) | 55 | 119 | 3954 |
| `B_ams_text` | **90% (+20)** | 54 | 544 | 4025 |
| `A_ams_prefix` | 60% | 11 | 473 | 18502 |
| `C_ams_hybrid` | 70% | 26 | 455 | 20320 |

### 10.4 Read

Four robust signals from the cross-N comparison:

1. **`D_full_history`'s input-token cost scales O(N)**: 159 → **301** tokens when we doubled the store from 10 to 20 facts. Non-`D` modes stayed flat (55 / 54 / 11 / 26). Confirmed with data, not handwaving.
2. **The AMS retrieval pipeline pays for itself at N≥20 and inverts the ranking**. At N=10, `B_flat_cos` (80%) > `B_ams_text` (70%); at N=20, `B_ams_text` (**90%**) > `B_flat_cos` (70%). Distractors are exactly what the strict-overlap content gate + rerank exist to filter; at small N flat cosine is fine, at larger N the tree+gate is worth its 3.5× higher retrieve latency. This is a **clean mechanistic advantage that does not come from the blackbox prefix channel at all** — it comes from the retrieval side of AMS, which the blackbox audit does not even measure.
3. **Prefix-only (`A_ams_prefix`) at 60% is non-trivial but flat across N**: same hit rate at N=10 and N=20, tokens constant at 11. The prefix routes topic reliably but can't deliver fluent lexical answers — exactly what blackbox axis-C said. Example: `kw='chopin' ans='love piano User pianopro: love love classical SSP'` — keyword would have been 'classical' if we'd chosen it as the test keyword; 'chopin' specifically requires fluent generation.
4. **`C_ams_hybrid` = prefix + top-1 text delivers 70% at 26 tokens/turn**. At N=20, `B_ams_text` at 90% / 54 tokens is strictly better than `C_hybrid` at 70% / 26 tokens on hit-rate per token, but `C_hybrid` beats `B_flat_cos` (70% / 55) on tokens per hit. So there is a Pareto point, not a dominance.

Generation-time cost on AMS modes (A/C) is ~5× higher than text-only modes because `MemLLM.generate` uses CFG decoding (double forward per step) plus the full logit-shaping pipeline. This is expected, not a regression — but it **is** the ceiling on A/C's commercial viability.

### 10.5 Decision

Per §10.1 decision framework:

- **Does the full AMS retrieval stack beat flat cosine at realistic store size?** Yes, at N=20: 90% vs 70%. The AMS retrieval pipeline — `DirectionTree` + `prepare_decode_context`'s strict-overlap gate + rerank — is the **immediately-shippable commercial value of this codebase**, independent of the blackbox prefix channel.
- **Does `C_ams_hybrid` (prefix + top-1 text) dominate `B_ams_text` (top-k text only) at same tokens?** No: at N=20, `B_ams_text` 90%/54t beats `C_hybrid` 70%/26t. The prefix channel's contribution to `C_hybrid` is **not yet pulling its weight** at these scales — the gain from prefix at `C_hybrid` can't close the 20-point gap created by reducing from top-k=3 to top-1 text context.
- **Does only `D_full_history` pass at parity quality?** No: `B_ams_text` at N=20 hits 90% at **18% of D's input-token cost**.

**Branch outcome:** AMS v3.46 code **already is** a competitive session layer for N≥20 stores, on the strength of its retrieval stack alone (`B_ams_text` winning at 90%). The prefix channel (A_ams_prefix, C_ams_hybrid) is a distinct R&D track that is not yet commercially justified by hit-rate or cost — and **this is what the 18/26 blackbox audit was correctly measuring**. The P0–P4 improvements should be evaluated against this bar: do they lift `A_ams_prefix` and `C_ams_hybrid` past `B_ams_text` at same or lower cost? If not, they are not worth shipping.

**Revised recommendation for product**: ship `B_ams_text` mode now as the "cheaper RAG backend" product. Move blackbox audit (P0–P4) into a separate research track with an explicit success bar: **trained A or C mode must beat `B_ams_text` at N=20 on both hit-rate and total cost**. Today neither does.

### 10.6 Pending comparisons (follow-up commits on this branch)

- **Trained checkpoint on vast.ai**: re-run both N=10 and N=20 with `AMS_TRAINED_WEIGHTS=ckpt/v346_trained.pt` to measure whether 60-step training lifts `A_ams_prefix` and `C_ams_hybrid` hit-rates past `B_ams_text`. If yes, P0–P4 become justified. If no, they are nice-to-have. Currently blocked on vast.ai SSH `Connection reset by peer` outage (~1h as of last check); will land when the remote recovers.
- **N=50 synthetic session**: extrapolate D's O(N) cost curve and confirm retrieval stack stays flat.
- **LongMemEval subset**: swap synthetic corpus for a 50-entry LongMemEval slice; cross-check against the v3.12 baseline (0.057 KW-F1, 11.5% HasAnswer).

### 10.7 Trained-ckpt results (v3.46-trained, 60 steps, NVIDIA H200, mt=30)

Training driver: `train_v346.py --steps 60 --out ckpt/v346_trained.pt`
Elapsed 193.5 s on H200. `post_probe = {tail_head_slot1_abs_mean: 7.25e-4, vocab_proj_last_abs_mean: 5.09e-4}` (handoff expected 7.30e-4 / 5.49e-4 — matches within <10 % noise).
Loader verification on both trained runs: `loaded=202 skipped=0 shape_errs=0  provenance=AgentMemory/v346-revertE-topk-nonexclusive-7e97`.

Δ columns compare trained vs **fresh-GPU** (not the CPU fresh rows in §10.3), so generate-time deltas are meaningful. CPU vs GPU latencies are on different hardware and are not commensurable.

#### N=10 (identity-only, `reports/session_viability_trained/` vs `reports/session_viability_fresh_gpu/`)

| Mode | Hit-rate | avg in-tokens | avg retrieve ms | avg generate ms | Δ fresh-GPU hit | Δ fresh-GPU gen-ms |
|---|---:|---:|---:|---:|---:|---:|
| `D_full_history` | 100% | 159 | 0.0 | 537 | 0 | +21 |
| `B_flat_cos` | 80% | 55 | 31 | 526 | 0 | +20 |
| `B_ams_text` | 80% | 55 | 415 | 376 | 0 | −9 |
| `A_ams_prefix` | 50% | 11 | 452 | 13033 | 0 | −1865 |
| `C_ams_hybrid` | 70% | 27 | 466 | 14520 | 0 | −843 |

#### N=20 (+10 distractors, `reports/session_viability_trained_20facts/` vs `reports/session_viability_fresh_gpu_20facts/`)

| Mode | Hit-rate | avg in-tokens | avg retrieve ms | avg generate ms | Δ fresh-GPU hit | Δ fresh-GPU gen-ms |
|---|---:|---:|---:|---:|---:|---:|
| `D_full_history` | 100% | 301 | 0.0 | 539 | 0 | +28 |
| `B_flat_cos` | 70% | 55 | 32 | 498 | 0 | +13 |
| `B_ams_text` | 80% | 55 | 417 | 373 | 0 | +3 |
| `A_ams_prefix` | 50% | 11 | 435 | 12900 | 0 | −2196 |
| `C_ams_hybrid` | 70% | 27 | 447 | 13853 | 0 | −1458 |

**Cross-cut vs §10.3 CPU fresh numbers**: GPU fresh-init `B_ams_text` at N=20 lands at 80 %, not the 90 % observed on CPU in §10.3. Greedy decode is deterministic given identical tokens, but the inputs differ by one turn of retrieval state (GPU bf16 vs CPU fp32 numerics in `layer_pool + _compute_content_semantic_emb` move the top-k ordering on one borderline query). Call the underlying B_ams_text capability 80–90 % at N=20 — one point (`davis`) is on the edge of the retrieval gate and flips between devices. Not a bug in `session_viability.py`.

### 10.8 Decision update

Answering §10.1 / handoff §5.1 questions **from the data**:

1. **Did training lift `A_ams_prefix` hit-rate past `B_ams_text` at N=20?** **No.** Observed: A = 50 % vs B_ams_text = 80 %. Gap is 30 points; training did not move A at all (same 50 % as fresh-GPU and the fresh-CPU 60 % of §10.3 is within synthetic-session noise on 10 queries).
2. **Did training lift `C_ams_hybrid` hit-rate past `B_ams_text` at N=20?** **No.** Observed: C = 70 % vs B_ams_text = 80 %. Same 70 % hit-rate as fresh-GPU; training did not help here either.
3. **Is the A/C generate-time cost still ~5× the text modes?** **Yes — but smaller.** Observed at N=20 trained: text modes gen ≈ 373–539 ms, A = 12 900 ms (≈ 24×), C = 13 853 ms (≈ 26×). So the multiplier is actually larger on H200 than the ~5× the fresh-init CPU rows showed — CFG double-forward + logit-shaping is a bigger share of wall time on fast hardware. Training shaved 1–2 s off A/C vs fresh-GPU, but not enough to change the order of magnitude.

**Rule per handoff §5.2:** all three answers ≥ 1 and 2 are "no", 3 is "yes" → §10.5 recommendation stands.

**Final decision (locked):**

1. **Ship `B_ams_text` as the v3.46 "cheaper RAG backend" product-line.** At N=20 on H200 it delivers 80 % hit-rate with 55 input tokens and 373 ms generate — roughly 18 % of `D_full_history`'s 301-token / 539-ms cost at parity quality on 8 of 10 queries.
2. **Move P0–P4 blackbox-audit work to a research track**, not a ship track. The success bar is unchanged: trained `A_ams_prefix` or `C_ams_hybrid` must beat `B_ams_text` at N=20 on both hit-rate **and** total token-ms cost. With 60-step training, neither approaches either bar.
3. **Training-efficacy finding (new, for a separate follow-up):** 60 steps on the §5.3 6-sentence rotating corpus is enough to nudge the `vocab_proj` weights by ~5e-4 abs mean, but **not enough to measurably move any hit-rate** in this spike. If P0–P4 research resumes, the first experiment should be "does 10× more training + a real corpus move A_ams_prefix / C_ams_hybrid at all?" — not "does the current `Cfg` need another knob turned?".

### 10.9 Framing correction (supersedes §10.5 / §10.8 "ship B_ams_text" wording)

§10.5 and §10.8 above were written with the wrong product frame. AMS **is not a RAG system** and **is not a knowledge graph**; reframing it as a "cheaper RAG backend" collapses the project to something it was never trying to be and retires its single differentiator. This subsection corrects the framing; the raw numbers in §10.3 / §10.7 stand unchanged.

#### 10.9.1 What AMS actually is

AMS's core mechanism is a **prefix / hidden-state injection channel**: memories are encoded to continuous vectors (per-slot prefix embeddings, optionally with logit-shaping), and those vectors are delivered into the backbone's forward pass **without going through text re-prompting**. That is exactly what `A_ams_prefix` and `C_ams_hybrid` exercise. The rest of the stack (`DirectionTree`, content gate, rerank, `semantic_emb` storage) exists **to feed** that channel, not to stand in for it.

Two things AMS is **not**, restated so it stays stated:
- **Not RAG.** RAG = retrieve text chunks, prepend them to the prompt, let the frozen LLM read them. `B_flat_cos` and `B_ams_text` are RAG-shaped modes. Shipping either is shipping a RAG product with AMS-flavored retrieval — it does **not** ship AMS's mechanism.
- **Not a knowledge graph.** No explicit entities, relations, or symbolic query surface. The `DirectionTree` is a routing structure over continuous embeddings, not an entity-relation graph.

#### 10.9.2 Re-reading §10.7 under the correct frame

Same numbers, different product identity:

- `B_ams_text` at 80–90 % N=20 is **a retrieval-side diagnostic, not a product candidate.** What it tells us: *given the right `source_text` in the prompt, the frozen Qwen2.5-1.5B can already answer 8–9 / 10 of these queries.* That is an upper bound on anything downstream, and it confirms `DirectionTree` + gate + rerank **do find the right memory**. It is not a competitor to A/C because it is not running AMS's mechanism at all — it is running a textual-prompt baseline.
- `A_ams_prefix` at 50 % N=20, trained or fresh, is **the actual measurement of AMS's mechanism as of v3.46.** 60 steps on the §5.3 corpus does not move it.
- `C_ams_hybrid` at 70 % N=20 sits between the two. It is still running the prefix channel; the +1 text memory is a crutch the product will remove once A is strong enough.
- `B_ams_text` − `A_ams_prefix` = 30 points at N=20 is therefore **the prefix-channel deficit to close**, not a gap to sidestep by shipping `B_ams_text`.

#### 10.9.3 Corrected decision

Replacing the §10.5 / §10.8 "ship B_ams_text, move P0–P4 to research track" wording with:

1. **P0–P4 blackbox-audit work remains the ship track** for AMS, because it is the only track that exercises AMS's actual mechanism. The §10.5 wording that moved it to "research track, not a ship track" was a framing error and is retracted here.
2. **`B_ams_text` is a retained diagnostic**, not a product line. Its role in the benchmark is: (a) upper-bound the information the retrieval side is successfully locating; (b) isolate whether a regression is on the retrieval side or the prefix side. It should **not** be presented externally as an AMS product.
3. **Success bar restated correctly (not inverted):** AMS is *done* when `A_ams_prefix` or `C_ams_hybrid` matches `D_full_history` at a large fraction of `D`'s token cost — independent of whether any text-injection mode (RAG-shaped) can also hit the same target. Beating `B_ams_text` is a *necessary intermediate milestone*, not the finish line, because `B_ams_text` is a different product category.
4. **Training-efficacy finding stands (§10.8.3):** 60 steps / §5.3 corpus is too small to conclude anything about the prefix channel's ceiling. The immediate next experiment is scale (10× steps + a real corpus), not a `Cfg` knob.

#### 10.9.4 What the §10.3 "inversion at N=20" meant

The CPU §10.3 row where `B_ams_text` (90 %) beat `B_flat_cos` (70 %) at N=20 was presented as "AMS retrieval wins at realistic N". Re-read under the correct frame: it is evidence that **AMS's routing structure is doing useful work on the retrieval side even before the prefix channel trains up**. That is architecturally encouraging, because it means when P0–P4 closes the prefix gap, the retrieval side is not the bottleneck. It is **not** a reason to ship RAG-shaped `B_ams_text` as the headline artifact.
