# v3.47-Mechanism1-Diagnostic Black-Box Audit Feedback

Compliant with `V331_BLACKBOX_TEST_SPEC.md` Sections 7, 7.7.

## 1. Run parameters

- SUT version: `scheme_b_v344.py` (unchanged)
- Runner version: `v331_blackbox_eval.py` with an additional diagnostic block inside 4.24 that reads `mem.semantic_emb` and computes an independent LOO NN on the frozen-Qwen attention-pool path. No pass criteria changed; the existing 4.24 metric on `context_descriptor` is retained as the primary.
- Weights: `ckpt/v344_trained.pt` (unchanged from v3.44-Trained)
- Env: `AMS_TRAINED_WEIGHTS=ckpt/v344_trained.pt`, `AMS_DETERMINISTIC=1`
- Device: CPU (single-threaded)
- Seed policy: per-case seeds as defined in SPEC Section 4
- Elapsed: 1498.0 s
- Exit code: 0

## 2. Count summary

- total: 26
- pass: 19
- fail: 7
- not_implemented: 0
- error: 0
- blocking_fail: 5 (4.7, 4.11, 4.13, 4.16, 4.19)

Identical count to v3.46-Deoverfit. No primary metric changed.

## 3. Mechanism 1 diagnostic (Section 4.24, v3.47+)

The runner computed LOO NN accuracy on two encodings of the same 16 memories drawn from 4 domains (music, space, cooking, finance):

| encoding | `loo_nn_accuracy_all_4` | `loo_nn_accuracy_heldout_2` | would pass thresholds? | per-domain (correct/n) |
|---|---|---|---|---|
| `context_descriptor` (learned `MemoryContextEncoder` + 60-step Trainer) | 0.625 (10/16) | 0.875 (7/8) | **no** — 4-domain metric below 0.65 | music 1/4, space 2/4, cooking 4/4, finance 3/4 |
| `semantic_emb` (frozen Qwen last-layer attention-pool over content-token positions; zero trainable parameters) | **0.812 (13/16)** | **0.875 (7/8)** | **yes** — both thresholds met | music 3/4, space 3/4, cooking 4/4, finance 3/4 |

Delta:
- 4-domain: +0.188 absolute accuracy (+30.0% relative)
- held-out: identical (both paths achieve 7/8)
- music specifically: +0.50 (1/4 → 3/4)
- space specifically: +0.25 (2/4 → 3/4)

## 4. Mechanism interpretation

`mem.semantic_emb` is computed by `scheme_b_v344.MemLLM._compute_content_semantic_emb`:

```
pooled = self.layer_pool(hs)                    # [B, T, d_LLM]
content_hs[b] = hidden_states[b, content_positions_b]
semantic_emb[b] = content_hs.mean(0)            # [d_LLM]
```

i.e., a content-token-masked mean of Qwen's last-layer hidden state. This IS attention-pooled (by Qwen's own forward pass) over the input tokens; no trainable AMS parameter touches it.

`mem.context_descriptor` is computed by `MemoryContextEncoder`:

```
ctx_desc = normalize(W_wte @ wte_centroid + 0.8 * W_hid @ hidden_mean)
```

`W_wte`, `W_hid` are orthogonal-initialized `Linear(d_LLM, d_ctx=128)` matrices.

Under the v3.44-Trained checkpoint, `semantic_emb` outperforms the learned `context_descriptor` on the 4-domain clustering task by 30% relative.

## 5. Operational consequence

`scheme_b_v344.MemLLM._compute_aggregated_context_descriptors_d_llm` already contains a fallback:

```
if mem.context_descriptor is not None and self.memory_context_encoder is not None:
    d_llm_vec = self.memory_context_encoder.decode(mem.context_descriptor.to(dev).float())
elif mem.semantic_emb is not None:
    d_llm_vec = mem.semantic_emb.to(dev).float()
```

Setting `Cfg(use_memory_context_encoder=False)` at model construction time:
- disables the learned encoder
- `mem.context_descriptor = None` on every `store_mem`
- the fallback path activates
- context slots are populated from `mem.semantic_emb`

This change is a single Cfg field override (not a SUT code change). No checkpoint retraining required.

## 6. Falsifiable prediction for the follow-up audit

If `Cfg(use_memory_context_encoder=False)` is set and the 26-case audit is rerun on the same `ckpt/v344_trained.pt` with `AMS_DETERMINISTIC=1`:
- 4.24 `loo_nn_accuracy_all_4` is predicted to transition from 0.625 (FAIL) to ≥ 0.80 (PASS)
- 4.24 `loo_nn_accuracy_heldout_2` is predicted to stay ≥ 0.87 (PASS)
- 4.24 overall transitions FAIL → PASS
- No other case's pass/fail state should change; `context_descriptor` is not consumed by any other case
- Total pass count transitions 19/26 → 20/26

Falsification condition: if 4.24 remains FAIL or any other case transitions PASS → FAIL, the prediction is refuted and the mechanism-1 account of the observed gap is incomplete.

## 7. Unchanged failing cases

Identical to v3.46-Deoverfit:
- 4.7, 4.11, 4.13, 4.16, 4.19, 4.23, 4.24

4.23 remains FAIL: mechanism 1 addresses the `context_descriptor` subchannel only; 4.23 measures the tail-slot subchannel, which is a different parameter group.

## 8. Artifact links

- `reports/v347_mechanism1_blackbox/report.json`
- `reports/v347_mechanism1_blackbox/report.md`
- `reports/v347_mechanism1_blackbox/runner.log`
- `reports/v347_mechanism1_blackbox/audit_feedback.md` (this file)

## 9. Next step

The measurement supports proceeding to the actual landing of mechanism 1 (Cfg override + rerun) before evaluating mechanisms 2, 3, 4 of the attention-sharing plan. Mechanism 1's delta is observable on the current checkpoint; no additional training is required.
