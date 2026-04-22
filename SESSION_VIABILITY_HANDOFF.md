# Handoff · PR #29 · session-viability trained-ckpt comparison

**To**: next agent with GPU access (vast.ai H200 preferred, any CUDA-capable device acceptable)
**From**: CPU-only cloud agent, session-viability spike author
**PR**: [#29](https://github.com/FluffyAIcode/AgentMemorySystem/pull/29) — `AgentMemory/v346-session-viability-7e97`
**Parent PR**: [#28](https://github.com/FluffyAIcode/AgentMemorySystem/pull/28) — `AgentMemory/v346-trained-gpu-7e97`

---

## 1. What's done

PR #29 already contains:
- `session_viability.py` — 5-mode benchmark (`D_full_history`, `B_flat_cos`, `B_ams_text`, `A_ams_prefix`, `C_ams_hybrid`), two store sizes (`--n-facts 10` / `--n-facts 20`), outputs `report.{json,md}` + stdout log.
- Fresh-init CPU results under `reports/session_viability_fresh/` (N=10) and `reports/session_viability_fresh_20facts/` (N=20).
- `SPRINT_CLOSEOUT_v3.46.md` §10 with the decision framework and the fresh-init decision.

**What's missing**: the trained-checkpoint comparison. The previous agent could not reach vast.ai (SSH `Connection reset by peer` for ~1h at handoff time) and the trained checkpoint `ckpt/v346_trained.pt` (455 MB, git-ignored) exists only on that remote.

## 2. Your task

Produce four trained-vs-fresh comparison reports (or as many as your GPU budget allows) and update the PR with them:

| Output directory | Command |
|---|---|
| `reports/session_viability_trained/` | `AMS_TRAINED_WEIGHTS=ckpt/v346_trained.pt python3 session_viability.py --mt 30 --n-facts 10 --out reports/session_viability_trained` |
| `reports/session_viability_trained_20facts/` | `AMS_TRAINED_WEIGHTS=ckpt/v346_trained.pt python3 session_viability.py --mt 30 --n-facts 20 --out reports/session_viability_trained_20facts` |

(Fresh-init reports in the same directories as currently committed are CPU baselines; if you want an apples-to-apples GPU baseline for the same backbone, also run the spike **without** `AMS_TRAINED_WEIGHTS` into `reports/session_viability_fresh_gpu/` and `reports/session_viability_fresh_gpu_20facts/`. Optional but recommended — CPU vs GPU latency numbers are not comparable.)

Expected wall time on an H200: ~3 min per `--n-facts 10` run, ~5 min per `--n-facts 20` run. Total ≤ 30 min.

## 3. Getting the trained checkpoint

**Option A — vast.ai still has it**: SSH to `vast` (credentials in `vast_ssh_key` secret + `vast_ssh_host` / `VAST_SSH_PORT` / `VAST_SSH_USER` env) and copy from `/workspace/AgentMemorySystem/ckpt/v346_trained.pt` (455 MB).

```bash
scp vast:/workspace/AgentMemorySystem/ckpt/v346_trained.pt ckpt/v346_trained.pt
```

Verify with:

```bash
python3 -c "import torch; blob = torch.load('ckpt/v346_trained.pt', map_location='cpu', weights_only=False); \
  print('keys', list(blob.keys())); \
  print('provenance', blob.get('provenance')); \
  print('steps', blob.get('steps')); \
  print('elapsed_s', blob.get('elapsed_s')); \
  print('post_probe', blob.get('post_probe'))"
```

Expected:
- `provenance = 'AgentMemory/v346-revertE-topk-nonexclusive-7e97'`
- `steps = 60`, `elapsed_s ≈ 335`
- `post_probe ≈ {'tail_head_slot1_abs_mean': 7.30e-4, 'vocab_proj_last_abs_mean': 5.49e-4}`

**Option B — vast.ai is gone, retrain from scratch**: the spec for the training run is in `SPRINT_CLOSEOUT_v3.46.md` §5.3. The training driver is already committed in this branch (inherited from #28):

```bash
# Preconditions: CUDA available, HF cache for Qwen/Qwen2.5-1.5B-Instruct warm
export AMS_DETERMINISTIC=1 TOKENIZERS_PARALLELISM=false
python3 train_v346.py --steps 60 --out ckpt/v346_trained.pt
```

Takes ~335 s on an H200, ~2000–3000 s on a smaller GPU. Checkpoint is bit-reproducible modulo cuDNN determinism flags, which is fine for this measurement.

## 4. Running the spike

After `ckpt/v346_trained.pt` exists in the repo root:

```bash
export HF_HOME=${HF_HOME:-/workspace/.hf_home}  # or whichever HF cache path your env uses
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export AMS_DETERMINISTIC=1
export TOKENIZERS_PARALLELISM=false
export AMS_TRAINED_WEIGHTS=$PWD/ckpt/v346_trained.pt

# Trained, N=10
python3 session_viability.py --mt 30 --n-facts 10 \
  --out reports/session_viability_trained | tee reports/session_viability_trained/stdout.log

# Trained, N=20
python3 session_viability.py --mt 30 --n-facts 20 \
  --out reports/session_viability_trained_20facts | tee reports/session_viability_trained_20facts/stdout.log
```

Verify at start of the stdout log that the loader actually picked up the ckpt:

```
  [AMS_TRAINED_WEIGHTS] loaded=202 skipped=0 shape_errs=0  path=.../ckpt/v346_trained.pt  provenance=AgentMemory/v346-revertE-topk-nonexclusive-7e97
```

If you see `loaded=0` — the ckpt is wrong shape for the current SUT. Do **not** proceed; check whether you accidentally grabbed `ckpt/v344_trained.pt` or `ckpt/v348_stacked.pt` (see §6 of `SPRINT_CLOSEOUT_v3.46.md` — those are intentionally incompatible).

## 5. Updating the PR

### 5.1 Fill in the missing comparison table

Add a new section `§10.7 Trained-ckpt results` at the end of `SPRINT_CLOSEOUT_v3.46.md` using the same table format as §10.3. Use this template:

```markdown
### 10.7 Trained-ckpt results (v3.46-trained, 60 steps, NVIDIA X, mt=30)

#### N=10 (identity-only, `reports/session_viability_trained/`)

| Mode | Hit-rate | avg in-tokens | avg retrieve ms | avg generate ms | Δ fresh |
|---|---:|---:|---:|---:|---:|
| `D_full_history` | ...% | ... | ... | ... | ... |
| `B_flat_cos` | ...% | ... | ... | ... | ... |
| `B_ams_text` | ...% | ... | ... | ... | ... |
| `A_ams_prefix` | ...% | ... | ... | ... | ... |
| `C_ams_hybrid` | ...% | ... | ... | ... | ... |

#### N=20 (+10 distractors, `reports/session_viability_trained_20facts/`)

(same format)

### 10.8 Decision update

Answer these three questions as plain yes/no with one-line justification. Do NOT predict before seeing the data (per §5.6 rule); write the answers *from* the data.

1. Did training lift `A_ams_prefix` hit-rate past `B_ams_text` at N=20?  [yes/no, observed: X% vs 90%]
2. Did training lift `C_ams_hybrid` hit-rate past `B_ams_text` at N=20?  [yes/no, observed: X% vs 90%]
3. Is the A/C generate-time cost still ~5× the text modes?  [yes/no, observed: Xms vs ~4000ms]

If all three are "no" → §10.5 recommendation stands: ship `B_ams_text`, P0–P4 research track is nice-to-have not must-have.
If 1 OR 2 flips to yes → P0–P4 becomes justified; write the specific case that tipped it.
If 3 flips to no → training also improved inference efficiency; note the magnitude.
```

### 5.2 Commit, push, update PR #29 body

```bash
git add reports/session_viability_trained/ reports/session_viability_trained_20facts/ SPRINT_CLOSEOUT_v3.46.md
git commit -m "session_viability: trained-ckpt comparison at N=10, N=20

<fill in your actual numbers and the §10.8 decision outcome>"
git push origin AgentMemory/v346-session-viability-7e97
```

Then update PR #29 with the trained rows added to the existing results tables in the PR body. Keep the fresh-init rows and the §10.5 decision; add `### Trained (N=10)` and `### Trained (N=20)` subsections and a final `## Final decision` section that answers §10.8 questions 1–3 from the data.

## 6. Guardrails (inherited from `SPRINT_CLOSEOUT_v3.46.md`)

- **No Cfg changes** (§5.4). If you find yourself wanting to change `Cfg` to tune some mode's output, stop — that's §3.3 anti-pattern (1).
- **No Trainer loss additions** (§5.4). Same reasoning.
- **No decode-time metric patching** like `rare_keyword_floor_boost` (§3.3 anti-pattern (2)).
- **Don't predict a Δ hit-rate before running the trained comparison.** Report numbers as data per §7.7 norm.
- If you find a real bug in `session_viability.py` (not a tuning opportunity), fix it and document the fix in the commit message. Keep the bar high: bug fix means "the measurement was wrong", not "I don't like the number".

## 7. What success looks like

- Two new report directories committed, with valid `report.json` showing the 5-mode aggregate on trained weights.
- `SPRINT_CLOSEOUT_v3.46.md` §10.7 + §10.8 filled in with actual observed numbers.
- PR #29 body updated with the trained rows and a final-decision paragraph.
- One atomic commit per logical step (e.g., "add trained N=10 run", "add trained N=20 run", "§10.7/§10.8 decision update"). Four commits is fine; one mega-commit is not.

## 8. If you get stuck

- vast.ai unreachable AND you have no other GPU AND you don't have budget to spin one up → commit whatever you got (even just a CPU trained run at `--n-facts 10 --mt 20` takes ~15 min; that's still better than nothing) and update PR #29 noting the constraint. Future agents can then pick up from that state.
- Loader reports `loaded=0` → see §4, you have the wrong ckpt. Re-fetch or retrain per §3.
- Any 5-mode run takes > 30 min on GPU → something's wrong, abort, check `nvidia-smi` and model loading.

Current branch head: `6c2eec6` (at time of handoff). Rebase on origin before starting.
